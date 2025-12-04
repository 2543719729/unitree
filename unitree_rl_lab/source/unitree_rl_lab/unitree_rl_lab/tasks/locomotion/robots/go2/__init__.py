"""
===============================================================================
Unitree Go2 四足机器人任务注册
===============================================================================

本文件注册 Unitree Go2 四足机器人的速度跟踪任务。

任务 ID: Unitree-Go2-Velocity

任务描述:
    - 训练 Go2 四足机器人进行速度跟踪
    - 支持平地和复杂地形
    - 使用 PPO 算法训练

使用方法:
    ```python
    import gymnasium as gym
    env = gym.make("Unitree-Go2-Velocity")
    ```
===============================================================================
"""

import gymnasium as gym

# 注册 Go2 速度跟踪任务
gym.register(
    id="Unitree-Go2-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",  # 训练环境配置
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",  # 推理环境配置
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",  # PPO配置
    },
)
