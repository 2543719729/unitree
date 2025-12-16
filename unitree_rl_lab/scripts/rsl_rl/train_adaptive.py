# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
===============================================================================
自适应训练脚本
===============================================================================

本脚本在标准RSL-RL训练的基础上，添加了自适应PPO参数调整功能。

功能特点:
    - 自动检测训练阶段（通过 adaptive_curriculum 模块）
    - 动态调整PPO参数（learning_rate, entropy_coef, clip_param, desired_kl）
    - 阶段切换时打印详细日志

使用方法:
    python scripts/rsl_rl/train_adaptive.py --task=Unitree-G1-29Dof-Stair-Blind-v0

===============================================================================
"""

"""Launch Isaac Sim Simulator first."""

import isaacsim  # noqa: F401 - must be imported first for pip-installed Isaac Sim

import gymnasium as gym
import pathlib
import sys

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401

sys.path.pop(0)

tasks = []
for task_spec in gym.registry.values():
    if "Unitree" in task_spec.id and "Isaac" not in task_spec.id:
        tasks.append(task_spec.id)

import argparse

import argcomplete

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL (Adaptive Mode).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, choices=tasks, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument(
    "--ppo_update_interval", type=int, default=100, help="Interval (iterations) to check and update PPO params."
)
parser.add_argument(
    "--resume_episode_length", type=float, default=1000.0, 
    help="Estimated episode length for resume training to initialize adaptive stage."
)
parser.add_argument(
    "--resume_terrain_level", type=float, default=1.0,
    help="Estimated terrain level for resume training to initialize adaptive stage."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
argcomplete.autocomplete(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import inspect
import os
import shutil
import statistics
import time
import torch
from collections import deque
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

# 导入自适应模块
from unitree_rl_lab.tasks.locomotion.mdp import (
    get_current_stage,
    get_stage_name,
    get_ppo_params,
    update_ppo_params,
    reset_adaptive_state,
    init_adaptive_state_from_metrics,
    get_adaptive_state_dict,
    load_adaptive_state_dict,
    apply_stage_params,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def adaptive_learn(
    runner: OnPolicyRunner,
    num_learning_iterations: int,
    init_at_random_ep_len: bool = False,
    ppo_update_interval: int = 100,
) -> None:
    """
    自适应训练循环
    
    基于 OnPolicyRunner.learn() 修改，添加了 PPO 参数自适应更新功能。
    
    Args:
        runner: RSL-RL 的 OnPolicyRunner 实例
        num_learning_iterations: 训练迭代次数
        init_at_random_ep_len: 是否随机初始化 episode 长度
        ppo_update_interval: PPO 参数更新检查间隔
    """
    # Initialize writer (same as runner.learn())
    runner._prepare_logging_writer()
    
    env = runner.env
    alg = runner.alg
    
    # Randomize initial episode lengths (for exploration)
    if init_at_random_ep_len:
        env.episode_length_buf = torch.randint_like(
            env.episode_length_buf, high=int(env.max_episode_length)
        )
    
    # Start learning
    obs = env.get_observations().to(runner.device)
    runner.train_mode()  # switch to train mode
    
    # Book keeping (same as runner.learn())
    ep_infos = []
    rewbuffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    cur_reward_sum = torch.zeros(env.num_envs, dtype=torch.float, device=runner.device)
    cur_episode_length = torch.zeros(env.num_envs, dtype=torch.float, device=runner.device)
    
    # Ensure all parameters are in-synced
    if runner.is_distributed:
        print(f"Synchronizing parameters for rank {runner.gpu_global_rank}...")
        alg.broadcast_parameters()
    
    # 跟踪上一次的阶段
    last_stage = get_current_stage()
    print(f"\n[Adaptive Training] 初始阶段: Stage {last_stage} ({get_stage_name(last_stage)})")
    print(f"[Adaptive Training] PPO 参数更新间隔: {ppo_update_interval} 迭代\n")
    
    # 在训练开始时应用当前阶段的所有参数（奖励权重、终止条件、PPO 参数）
    apply_stage_params(env.unwrapped, alg, last_stage)
    
    # Start training
    start_iter = runner.current_learning_iteration
    tot_iter = start_iter + num_learning_iterations
    
    for it in range(start_iter, tot_iter):
        start = time.time()
        
        # Rollout
        with torch.inference_mode():
            for _ in range(runner.num_steps_per_env):
                # Sample actions
                actions = alg.act(obs)
                # Step the environment
                obs, rewards, dones, extras = env.step(actions.to(env.device))
                # Move to device
                obs, rewards, dones = (obs.to(runner.device), rewards.to(runner.device), dones.to(runner.device))
                # Process the step
                alg.process_env_step(obs, rewards, dones, extras)
                
                # Book keeping (same as runner.learn())
                if runner.log_dir is not None:
                    if "episode" in extras:
                        ep_infos.append(extras["episode"])
                    elif "log" in extras:
                        ep_infos.append(extras["log"])
                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0
            
            stop = time.time()
            collection_time = stop - start
            start = stop
            
            # Compute returns
            alg.compute_returns(obs)
        
        # Update policy
        loss_dict = alg.update()
        
        stop = time.time()
        learn_time = stop - start
        runner.current_learning_iteration = it
        
        # ==================== 自适应 PPO 参数状态打印 ====================
        current_stage = get_current_stage()
        ppo_params = get_ppo_params(current_stage)
        
        # 获取算法实际使用的参数
        actual_lr = getattr(alg, 'learning_rate', ppo_params['learning_rate'])
        actual_entropy = getattr(alg, 'entropy_coef', ppo_params['entropy_coef'])
        actual_clip = getattr(alg, 'clip_param', ppo_params['clip_param'])
        actual_kl = getattr(alg, 'desired_kl', ppo_params['desired_kl'])
        
        # 每次迭代打印自适应参数状态
        print(f"[Iter {it}] Stage {current_stage} ({get_stage_name(current_stage)}) | "
              f"lr={actual_lr:.2e} entropy={actual_entropy:.4f} clip={actual_clip:.3f} kl={actual_kl:.4f}")
        
        # 检查阶段是否变化，更新 PPO 参数
        if current_stage != last_stage:
            print(f"\n[Adaptive PPO] 检测到阶段变化: Stage {last_stage} -> Stage {current_stage}")
            
            # 更新 PPO 参数
            ppo_updates = update_ppo_params(alg, current_stage)
            
            if ppo_updates:
                print(f"[Adaptive PPO] PPO 参数已更新:")
                for param_name, vals in ppo_updates.items():
                    if param_name == "learning_rate":
                        print(f"    - {param_name}: {vals['old']:.2e} -> {vals['new']:.2e}")
                    else:
                        print(f"    - {param_name}: {vals['old']:.4f} -> {vals['new']:.4f}")
            
            last_stage = current_stage
            print()
        
        # Log information (using runner.log() method)
        if runner.log_dir is not None and not runner.disable_logs:
            runner.log(locals())
            # Save model and adaptive state
            if it % runner.save_interval == 0:
                model_path = os.path.join(runner.log_dir, f"model_{it}.pt")
                runner.save(model_path)
                # 保存自适应状态到同一目录
                adaptive_state_path = os.path.join(runner.log_dir, f"adaptive_state_{it}.pt")
                torch.save(get_adaptive_state_dict(), adaptive_state_path)
        
        # Clear episode infos
        ep_infos.clear()
    
    # Save final model and adaptive state
    if runner.log_dir is not None and not runner.disable_logs:
        final_iter = runner.current_learning_iteration
        runner.save(os.path.join(runner.log_dir, f"model_{final_iter}.pt"))
        # 保存最终自适应状态
        torch.save(get_adaptive_state_dict(), os.path.join(runner.log_dir, f"adaptive_state_{final_iter}.pt"))


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent (Adaptive Mode)."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)
        
        # 尝试自动加载自适应状态
        # 从 model_XXX.pt 推断 adaptive_state_XXX.pt 的路径
        resume_dir = os.path.dirname(resume_path)
        checkpoint_name = os.path.basename(resume_path)  # e.g., model_400.pt
        iter_num = checkpoint_name.replace("model_", "").replace(".pt", "")  # e.g., 400
        adaptive_state_path = os.path.join(resume_dir, f"adaptive_state_{iter_num}.pt")
        
        if os.path.exists(adaptive_state_path):
            # 自动加载保存的自适应状态
            print(f"[INFO]: Loading adaptive state from: {adaptive_state_path}")
            adaptive_state_dict = torch.load(adaptive_state_path)
            load_adaptive_state_dict(adaptive_state_dict)
        else:
            # 没有找到保存的状态，使用命令行参数或默认值推断
            print(f"[INFO]: No adaptive state found at {adaptive_state_path}")
            print(f"[INFO]: Initializing adaptive state from metrics...")
            init_adaptive_state_from_metrics(args_cli.resume_episode_length, args_cli.resume_terrain_level)
    else:
        # 新训练，重置自适应状态
        reset_adaptive_state()
    
    print("\n" + "=" * 70)
    print("[Adaptive Training] 自适应训练模式已启用")
    print("=" * 70 + "\n")

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    export_deploy_cfg(env.unwrapped, log_dir)
    # copy the environment configuration file to the log directory
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    # 使用自适应训练循环
    adaptive_learn(
        runner,
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
        ppo_update_interval=args_cli.ppo_update_interval,
    )

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
