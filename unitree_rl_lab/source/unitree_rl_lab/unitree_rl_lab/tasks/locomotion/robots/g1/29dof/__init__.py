import gymnasium as gym

# ============================================================================
#                       4阶段训练任务注册
# ============================================================================
# 训练顺序：
#   阶段1: Unitree-G1-29dof-Velocity        - 盲走（平地，无 height_scan）
#   阶段2: Unitree-G1-29dof-Velocity-HeightScan - 带传感器行走（平地，有 height_scan）
#   阶段3: Unitree-G1-29dof-Stair-Blind     - 盲爬楼梯（楼梯，无 height_scan）
#   阶段4: Unitree-G1-29dof-Stair           - 带传感器爬楼梯（楼梯，有 height_scan）
# ============================================================================

# ============================================================================
# 阶段1: 盲走（平地，无 height_scan）
# ============================================================================
gym.register(
    id="Unitree-G1-29dof-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# ============================================================================
# 阶段2: 带传感器行走（平地，有 height_scan）
# ============================================================================
gym.register(
    id="Unitree-G1-29dof-Velocity-HeightScan",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotHeightScanEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotHeightScanPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# ============================================================================
# 阶段3: 盲爬楼梯（楼梯，无 height_scan）
# ============================================================================
gym.register(
    id="Unitree-G1-29dof-Stair-Blind",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stair_env_cfg:StairBlindEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.stair_env_cfg:StairBlindPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# ============================================================================
# 阶段4: 带传感器爬楼梯（楼梯，有 height_scan）
# ============================================================================
gym.register(
    id="Unitree-G1-29dof-Stair",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stair_env_cfg:StairClimbEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.stair_env_cfg:StairClimbPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# ============================================================================
#                       统一条件策略任务
# ============================================================================
# 一个策略同时学习4种模式，通过 mode_flag 控制行为
# 部署时可智能切换模式，无需加载多个模型
# ============================================================================
gym.register(
    id="Unitree-G1-29dof-Unified",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.unified_env_cfg:UnifiedEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.unified_env_cfg:UnifiedPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
