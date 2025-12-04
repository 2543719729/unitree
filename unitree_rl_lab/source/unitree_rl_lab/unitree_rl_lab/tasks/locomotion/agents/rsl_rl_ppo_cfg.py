# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
===============================================================================
PPO 算法基础配置文件
===============================================================================

本文件定义了用于机器人运动控制任务的 PPO (Proximal Policy Optimization) 算法配置。
这是一个基础配置类，可被具体任务继承和定制。

主要配置模块:
    1. 训练运行参数: 训练迭代次数、保存间隔等
    2. 策略网络配置: Actor-Critic 神经网络架构
    3. PPO算法参数: 学习率、裁剪参数、折扣因子等核心超参数

使用场景:
    - 作为基类被 velocity_env_cfg, stair_env_cfg 等具体任务配置继承
    - 提供稳定的 PPO 超参数基线
    - 可针对特定任务进行微调
===============================================================================
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    基础 PPO 训练配置类
    
    继承自 RslRlOnPolicyRunnerCfg，定义了用于机器人运动控制的标准 PPO 配置。
    包含训练超参数、神经网络架构和算法参数。
    """
    
    # ======================== 训练运行参数 ========================
    num_steps_per_env = 24  # 每个环境每次 rollout 收集的步数（trajectory 长度）
    max_iterations = 50000  # 最大训练迭代次数
    save_interval = 100  # 每 100 次迭代保存一次模型检查点
    experiment_name = ""  # 实验名称（默认与任务名相同）
    empirical_normalization = False  # 是否使用经验归一化（对观测值进行统计归一化）
    
    # ======================== 策略网络配置 ========================
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,  # 初始动作噪声标准差（用于探索）
        actor_hidden_dims=[512, 256, 128],  # Actor（策略）网络隐藏层维度：3层神经网络
        critic_hidden_dims=[512, 256, 128],  # Critic（价值）网络隐藏层维度：3层神经网络
        activation="elu",  # 激活函数：ELU（指数线性单元），比 ReLU 更平滑
    )
    
    # ======================== PPO 算法参数 ========================
    algorithm = RslRlPpoAlgorithmCfg(
        # --- 价值函数相关 ---
        value_loss_coef=1.0,  # 价值函数损失系数（在总损失中的权重）
        use_clipped_value_loss=True,  # 使用裁剪的价值损失（类似于策略裁剪）
        
        # --- PPO 核心参数 ---
        clip_param=0.2,  # PPO 裁剪参数 ε，限制策略更新幅度（标准值 0.2）
        entropy_coef=0.01,  # 熵正则化系数，鼓励探索（值越大探索越多）
        
        # --- 训练优化参数 ---
        num_learning_epochs=5,  # 每次 rollout 后的训练轮数（重复使用数据）
        num_mini_batches=4,  # 每轮训练的 mini-batch 数量
        learning_rate=1.0e-3,  # 学习率（Adam 优化器）
        schedule="adaptive",  # 学习率调度策略：自适应（基于 KL 散度）
        
        # --- 回报计算参数 ---
        gamma=0.99,  # 折扣因子（discount factor），控制对未来奖励的重视程度
        lam=0.95,  # GAE-Lambda 参数，用于计算优势函数（Advantage）
        
        # --- 训练稳定性参数 ---
        desired_kl=0.01,  # 目标 KL 散度，用于自适应学习率调整
        max_grad_norm=1.0,  # 梯度裁剪最大范数，防止梯度爆炸
    )
