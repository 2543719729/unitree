"""
===============================================================================
Unitree G1 29DOF MuJoCo PPO 训练脚本

使用 Stable-Baselines3 进行 PPO 训练
仿照 IsaacLab 的训练流程
===============================================================================
"""

import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from envs.g1_29dof_env import G1_29DOF_Env
from configs.unified_env_cfg import (
    UnifiedEnvCfg,
    get_default_config,
    get_flat_blind_config,
    get_flat_sensor_config,
    get_stairs_blind_config,
    get_stairs_sensor_config,
)


def make_env(cfg: UnifiedEnvCfg, rank: int, seed: int = 0):
    """
    创建环境工厂函数
    """
    def _init():
        env = G1_29DOF_Env(
            render_mode=None,
            sim_dt=cfg.simulation.sim_dt,
            control_decimation=cfg.simulation.control_decimation,
            episode_length_s=cfg.simulation.episode_length_s,
            num_modes=cfg.mode.num_modes,
            mode_probabilities=cfg.mode.mode_probabilities,
            cmd_lin_vel_x_range=cfg.commands.limit_lin_vel_x_range,
            cmd_lin_vel_y_range=cfg.commands.limit_lin_vel_y_range,
            cmd_ang_vel_z_range=cfg.commands.limit_ang_vel_z_range,
            history_length=cfg.observations.history_length,
            use_height_scan=cfg.observations.use_height_scan,
            action_scale=cfg.actions.action_scale,
            kp=cfg.pd_control.leg_kp,
            kd=cfg.pd_control.leg_kd,
            domain_randomization=cfg.domain_randomization.enabled,
            terrain_type=cfg.terrain.terrain_type,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    
    set_random_seed(seed)
    return _init


def create_vec_env(cfg: UnifiedEnvCfg, num_envs: int, seed: int = 0):
    """
    创建向量化环境
    """
    if num_envs == 1:
        env = DummyVecEnv([make_env(cfg, 0, seed)])
    else:
        env = SubprocVecEnv([make_env(cfg, i, seed) for i in range(num_envs)])
    
    # 归一化观测和奖励
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )
    
    return env


def get_ppo_config(cfg: UnifiedEnvCfg) -> dict:
    """
    获取 PPO 超参数配置
    
    参考 IsaacLab 的默认设置
    """
    return {
        "learning_rate": 3e-4,
        "n_steps": 24,  # 每个环境的步数
        "batch_size": 24 * 16,  # mini-batch 大小
        "n_epochs": 5,  # 每次更新的 epoch 数
        "gamma": 0.99,  # 折扣因子
        "gae_lambda": 0.95,  # GAE lambda
        "clip_range": 0.2,  # PPO clip range
        "clip_range_vf": None,  # 价值函数 clip (None = 不裁剪)
        "ent_coef": 0.01,  # 熵系数
        "vf_coef": 0.5,  # 价值函数系数
        "max_grad_norm": 1.0,  # 梯度裁剪
        "target_kl": None,  # 目标 KL 散度
        "verbose": 1,
    }


def train(args):
    """
    训练主函数
    """
    # 创建日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"g1_29dof_{args.mode}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"=" * 60)
    print(f"Unitree G1 29DOF MuJoCo PPO Training")
    print(f"=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Num envs: {args.num_envs}")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Log directory: {log_dir}")
    print(f"=" * 60)
    
    # 获取配置
    if args.mode == "unified":
        cfg = get_default_config()
    elif args.mode == "flat_blind":
        cfg = get_flat_blind_config()
    elif args.mode == "flat_sensor":
        cfg = get_flat_sensor_config()
    elif args.mode == "stairs_blind":
        cfg = get_stairs_blind_config()
    elif args.mode == "stairs_sensor":
        cfg = get_stairs_sensor_config()
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    
    # 创建环境
    print("Creating training environment...")
    train_env = create_vec_env(cfg, args.num_envs, seed=args.seed)
    
    print("Creating evaluation environment...")
    eval_env = create_vec_env(cfg, 1, seed=args.seed + 1000)
    
    # 获取 PPO 配置
    ppo_config = get_ppo_config(cfg)
    ppo_config["n_steps"] = args.n_steps
    ppo_config["batch_size"] = args.batch_size
    
    # 网络架构
    policy_kwargs = {
        "net_arch": dict(
            pi=[512, 256, 128],  # 策略网络
            vf=[512, 256, 128],  # 价值网络
        ),
        "activation_fn": torch.nn.ELU,
    }
    
    # 创建 PPO 模型
    print("Creating PPO model...")
    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = PPO.load(
            args.resume,
            env=train_env,
            tensorboard_log=log_dir,
        )
    else:
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_dir,
            device=args.device,
            **ppo_config,
        )
    
    # 回调函数
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="g1_29dof",
        save_vecnormalize=True,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=5,
        deterministic=True,
    )
    
    callbacks = CallbackList([checkpoint_callback, eval_callback])
    
    # 开始训练
    print("Starting training...")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time / 3600:.2f} hours")
    
    # 保存最终模型
    final_model_path = os.path.join(log_dir, "final_model")
    model.save(final_model_path)
    train_env.save(os.path.join(log_dir, "vec_normalize.pkl"))
    
    print(f"Final model saved to: {final_model_path}")
    
    # 清理
    train_env.close()
    eval_env.close()
    
    return model, log_dir


def evaluate(args):
    """
    评估模型
    """
    print(f"Loading model from: {args.model_path}")
    
    # 加载配置
    cfg = get_default_config()
    
    # 创建环境
    env = G1_29DOF_Env(
        render_mode="human",
        sim_dt=cfg.simulation.sim_dt,
        control_decimation=cfg.simulation.control_decimation,
        episode_length_s=cfg.simulation.episode_length_s,
        num_modes=cfg.mode.num_modes,
        mode_probabilities=cfg.mode.mode_probabilities,
        cmd_lin_vel_x_range=cfg.commands.limit_lin_vel_x_range,
        cmd_lin_vel_y_range=cfg.commands.limit_lin_vel_y_range,
        cmd_ang_vel_z_range=cfg.commands.limit_ang_vel_z_range,
        history_length=cfg.observations.history_length,
        use_height_scan=cfg.observations.use_height_scan,
        action_scale=cfg.actions.action_scale,
        domain_randomization=False,
    )
    
    # 加载模型
    model = PPO.load(args.model_path)
    
    # 加载归一化参数
    if args.vec_normalize_path:
        from stable_baselines3.common.vec_env import VecNormalize
        # 需要包装环境
        env = DummyVecEnv([lambda: env])
        env = VecNormalize.load(args.vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
    
    print("Starting evaluation...")
    
    total_rewards = []
    
    for episode in range(args.num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            env.render()
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    print(f"\nMean reward: {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description="G1 29DOF MuJoCo RL Training")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # 训练命令
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--mode", type=str, default="unified",
                              choices=["unified", "flat_blind", "flat_sensor", 
                                       "stairs_blind", "stairs_sensor"],
                              help="Training mode")
    train_parser.add_argument("--num_envs", type=int, default=16,
                              help="Number of parallel environments")
    train_parser.add_argument("--total_timesteps", type=int, default=10_000_000,
                              help="Total training timesteps")
    train_parser.add_argument("--n_steps", type=int, default=24,
                              help="Number of steps per environment per update")
    train_parser.add_argument("--batch_size", type=int, default=384,
                              help="Mini-batch size")
    train_parser.add_argument("--seed", type=int, default=42,
                              help="Random seed")
    train_parser.add_argument("--device", type=str, default="auto",
                              help="Device (auto, cpu, cuda)")
    train_parser.add_argument("--log_dir", type=str, default="./logs",
                              help="Log directory")
    train_parser.add_argument("--save_freq", type=int, default=50000,
                              help="Checkpoint save frequency")
    train_parser.add_argument("--eval_freq", type=int, default=10000,
                              help="Evaluation frequency")
    train_parser.add_argument("--resume", type=str, default=None,
                              help="Path to model to resume training")
    
    # 评估命令
    eval_parser = subparsers.add_parser("eval", help="Evaluate the model")
    eval_parser.add_argument("--model_path", type=str, required=True,
                             help="Path to trained model")
    eval_parser.add_argument("--vec_normalize_path", type=str, default=None,
                             help="Path to VecNormalize stats")
    eval_parser.add_argument("--num_episodes", type=int, default=10,
                             help="Number of evaluation episodes")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train(args)
    elif args.command == "eval":
        evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
