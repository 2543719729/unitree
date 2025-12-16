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


# ============================================================================
#                     盲爬楼梯专用 PPO 配置
# ============================================================================
@configclass
class StairBlindPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    盲爬楼梯任务专用 PPO 配置类
    
    针对盲爬楼梯（Blind Stair Climbing）任务的特点进行优化：
        1. 更长的轨迹收集：覆盖完整的攀爬动作周期（抬腿-迈步-着地-重心转移）
        2. 更大的网络容量：学习从历史观测中提取隐式地形信息
        3. 更保守的策略更新：避免策略崩溃导致的训练失败
        4. 更高的折扣因子：重视长期回报，完成整段楼梯攀爬
    
    盲爬模式特点：
        - 不使用 height_scan（地形高度扫描）
        - 仅依赖本体感知：关节角度、速度、IMU、重力投影
        - 通过 5 帧历史观测隐式学习地形信息
        - 参考 Cassie 盲爬楼梯论文的方法
    
    使用场景：
        - Unitree-G1-29dof-Stair-Blind 任务
        - 从平地模型迁移学习时的微调
        - 需要鲁棒盲爬能力的部署场景
    """
    
    # ======================== 训练运行参数 ========================
    # num_steps_per_env: 每个环境每次 rollout 收集的步数
    #
    # 原值 24 步仅能覆盖约 2/3 的攀爬周期
    # 完整攀爬一级楼梯需要约 36-40 步：
    #   - 抬腿阶段: ~10 步
    #   - 迈步阶段: ~8 步
    #   - 着地阶段: ~5 步
    #   - 重心转移: ~8 步
    #   - 稳定阶段: ~5 步
    #
    # 设置为 48 步可覆盖完整周期 + 后续动作衔接
    num_steps_per_env = 32
    
    # max_iterations: 最大训练迭代次数
    # 盲爬任务比平地行走更复杂，需要更多迭代才能收敛
    max_iterations = 80000
    
    # save_interval: 模型保存间隔
    # 适中的保存频率，便于选择最佳检查点
    save_interval = 200
    
    # experiment_name: 实验名称
    # 留空则使用任务名自动填充
    experiment_name = ""
    
    # empirical_normalization: 经验归一化
    # 启用观测值统计归一化，帮助稳定训练
    # 盲爬任务的观测范围可能较大（如关节速度），归一化有助于学习
    empirical_normalization = True
    
    # ======================== 策略网络配置 ========================
    policy = RslRlPpoActorCriticCfg(
        # init_noise_std: 初始动作噪声标准差
        # 1.0 是标准值，保持适度探索
        init_noise_std=1.0,
        
        # actor_hidden_dims: Actor（策略）网络隐藏层维度
        #
        # 原 [512, 256, 128] 约 200K 参数，适合平地行走
        # 增加到 [512, 256, 256, 128] 约 330K 参数
        #
        # 增加深度的原因：
        #   1. 盲爬需要从 5 帧历史观测中提取隐式地形信息
        #   2. 更深的网络能学习更复杂的时序模式
        #   3. 攀爬动作序列比平地行走更复杂
        actor_hidden_dims=[512, 256, 256, 128],
        
        # critic_hidden_dims: Critic（价值）网络隐藏层维度
        # 与 Actor 保持对称，确保价值估计能力匹配
        critic_hidden_dims=[512, 256, 256, 128],
        
        # activation: 激活函数
        # ELU 比 ReLU 更平滑，有助于梯度流动
        activation="elu",
    )
    
    # ======================== PPO 算法参数 ========================
    algorithm = RslRlPpoAlgorithmCfg(
        # -------------------- 价值函数相关 --------------------
        # value_loss_coef: 价值函数损失系数
        # 1.0 是标准值，确保 Critic 学习充分
        value_loss_coef=1.1,
        
        # use_clipped_value_loss: 使用裁剪的价值损失
        # 类似于策略裁剪，提高训练稳定性
        use_clipped_value_loss=True,
        
        # -------------------- PPO 核心参数 --------------------
        # clip_param: PPO 裁剪参数 ε
        #
        # 原值 0.2 是标准值
        # 降低到 0.18 使策略更新更保守
        #
        # 原因：盲爬任务中，激进的策略更新可能导致机器人摔倒
        #       更保守的更新可以避免策略崩溃
        clip_param=0.2,
        
        # entropy_coef: 熵正则化系数
        #
        # 0.01 是标准值，保持适度探索
        # 盲爬任务需要探索不同的攀爬策略
        entropy_coef=0.012,
        
        # -------------------- 训练优化参数 --------------------
        # num_learning_epochs: 每次 rollout 后的训练轮数
        #
        # 原值 5 轮
        # 增加到 8 轮，更充分利用收集的数据
        #
        # 原因：盲爬数据更宝贵，每条成功轨迹都值得多次学习
        num_learning_epochs=8,
        
        # num_mini_batches: 每轮训练的 mini-batch 数量
        #
        # 原值 4 个
        # 增加到 8 个，提供更稳定的梯度估计
        #
        # 计算：4096 envs × 48 steps / 8 batches = 24576 samples/batch
        num_mini_batches=8,
        
        # learning_rate: 学习率
        #
        # 原值 1e-3
        # 降低到 5e-4，更稳定的学习过程
        #
        # 原因：盲爬任务复杂，较低学习率可以：
        #   1. 避免策略剧烈变化导致的不稳定
        #   2. 更细致地优化动作质量
        #   3. 配合自适应调度，让 KL 约束更有效
        learning_rate=1e-3,
        
        # schedule: 学习率调度策略
        # "adaptive" 基于 KL 散度自动调整学习率：
        #   - 如果 KL > desired_kl，降低学习率
        #   - 如果 KL < desired_kl，提高学习率
        schedule="adaptive",
        
        # -------------------- 回报计算参数 --------------------
        # gamma: 折扣因子
        #
        # 原值 0.99，有效视野约 100 步
        # 提高到 0.995，有效视野约 200 步
        #
        # 有效视野公式：1 / (1 - gamma)
        #   - gamma=0.99  → 视野 100 步
        #   - gamma=0.995 → 视野 200 步
        #   - gamma=0.997 → 视野 333 步
        #
        # 原因：盲爬楼梯需要长期规划
        #   - 一级楼梯约 40 步
        #   - 完整楼梯可能 200+ 步
        #   - 更高 gamma 使策略重视整段楼梯的完成
        gamma=0.995,
        
        # lam: GAE-Lambda 参数
        # 0.95 是标准值，在偏差和方差之间取得平衡
        lam=0.95,
        
        # -------------------- 训练稳定性参数 --------------------
        # desired_kl: 目标 KL 散度
        #
        # 0.01 是标准值
        # 配合 adaptive schedule，当 KL 偏离时自动调整学习率
        desired_kl=0.01,
        
        # max_grad_norm: 梯度裁剪最大范数
        #
        # 原值 1.0
        # 降低到 0.8，更严格的梯度裁剪
        #
        # 原因：防止梯度爆炸导致的策略崩溃
        #       盲爬任务的奖励信号可能有较大波动
        max_grad_norm=0.8,
    )
