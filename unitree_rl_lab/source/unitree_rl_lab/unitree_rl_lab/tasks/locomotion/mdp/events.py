"""
===============================================================================
事件函数模块 - 统一条件策略的模式切换与地形检测
===============================================================================

本模块实现了强化学习训练中的事件处理函数，主要用于：
    1. 条件策略训练中的模式切换
    2. 地形类型检测与自动模式设置
    3. 手动模式控制（用于测试和部署）

核心概念 - 条件策略（Conditional Policy）：
    条件策略是一种能够根据不同条件（如地形类型、传感器可用性）
    自动调整行为的策略。通过在训练时随机切换模式，策略学会：
        - 在不同地形上采用不同的运动策略
        - 在传感器失效时切换到盲模式
        - 根据 mode_flag 观测值调整行为

模式定义（4种模式）：
    ┌─────────┬─────────────┬─────────────┐
    │  模式   │   地形类型   │  传感器状态  │
    ├─────────┼─────────────┼─────────────┤
    │ 模式 0  │   平地      │    盲模式    │
    │ 模式 1  │   平地      │   有传感器   │
    │ 模式 2  │   楼梯      │    盲模式    │
    │ 模式 3  │   楼梯      │   有传感器   │
    └─────────┴─────────────┴─────────────┘

模式编码规则：
    mode = terrain_type * 2 + sensor_available
    其中：
        - terrain_type: 0=平地, 1=楼梯
        - sensor_available: 0=盲模式, 1=有传感器

主要函数：
    - reset_mode_randomly: 随机重置模式（训练时使用）
    - reset_mode_by_terrain: 根据地形自动设置模式
    - set_mode_manually: 手动设置模式（测试/部署时使用）

使用场景：
    - 训练统一的条件策略，能够处理多种场景
    - 实现传感器故障时的优雅降级
    - 在不同地形上自动切换运动策略

与其他模块的关系：
    - observations.py: 提供 mode_flag 观测函数，读取 _current_mode
    - rewards.py: 可以根据模式调整奖励权重
    - 环境配置: 在 events 部分注册这些函数

参考文献：
    - Conditional Policy Learning: 条件策略学习方法
    - Multi-task RL: 多任务强化学习
===============================================================================
"""

from __future__ import annotations  # 启用延迟注解评估，允许在类型提示中使用尚未定义的类型

import torch  # PyTorch 深度学习框架，用于张量计算
from typing import TYPE_CHECKING, Sequence  # 类型提示工具

# 条件导入：仅在类型检查时导入，避免循环导入问题
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv  # 基于管理器的环境基类


def reset_mode_randomly(
    env: ManagerBasedEnv,
    env_ids: Sequence[int],
    num_modes: int = 4,
    mode_probabilities: list[float] | None = None,
):
    """
    随机重置模式标志 - 条件策略训练的核心函数
    
    在每次环境重置时，随机为该环境选择一个运行模式。
    这是条件策略训练的核心机制：通过随机切换模式，让策略学会
    根据 mode_flag 观测值调整行为。
    
    工作流程：
        1. 初始化模式相关的缓冲区（首次调用时）
        2. 根据概率分布采样新模式
        3. 更新模式标志和辅助标志
    
    模式定义：
        ┌─────────┬─────────────┬─────────────┬─────────────────────┐
        │  模式   │   地形类型   │  传感器状态  │       描述          │
        ├─────────┼─────────────┼─────────────┼─────────────────────┤
        │ 模式 0  │   平地      │    盲模式    │ 平地盲走            │
        │ 模式 1  │   平地      │   有传感器   │ 平地带传感器行走    │
        │ 模式 2  │   楼梯      │    盲模式    │ 楼梯盲爬            │
        │ 模式 3  │   楼梯      │   有传感器   │ 楼梯带传感器攀爬    │
        └─────────┴─────────────┴─────────────┴─────────────────────┘
    
    缓冲区说明：
        - _current_mode: 当前模式标志，形状 [num_envs]，值为 0-3
        - _terrain_is_stair: 地形类型标志，形状 [num_envs, 1]，0=平地, 1=楼梯
        - _sensor_available: 传感器可用标志，形状 [num_envs, 1]，0=盲, 1=有传感器
    
    Args:
        env (ManagerBasedEnv): 环境实例，包含 num_envs、device 等属性
        env_ids (Sequence[int]): 需要重置的环境 ID 列表
            通常是在当前时间步结束（terminated 或 truncated）的环境
        num_modes (int): 模式数量，默认 4
            可以扩展到更多模式（如添加斜坡、台阶等）
        mode_probabilities (list[float] | None): 各模式的采样概率
            - None: 等概率采样，每个模式概率为 1/num_modes
            - 列表: 自定义概率，如 [0.3, 0.2, 0.3, 0.2] 表示更多平地训练
    
    Returns:
        None: 直接修改 env 的内部状态
    
    Example:
        在环境配置中使用此函数::
        
            from isaaclab.managers import EventTermCfg as EventTerm
            
            reset_mode = EventTerm(
                func=reset_mode_randomly,
                mode="reset",  # 在环境重置时触发
                params={
                    "num_modes": 4,
                    "mode_probabilities": [0.25, 0.25, 0.25, 0.25],
                }
            )
    
    Note:
        - 此函数会在 env 上创建 _current_mode、_terrain_is_stair、_sensor_available 属性
        - 这些属性可以被 observations.py 中的函数读取，作为策略的输入
        - 训练时应该使用随机模式，部署时可以使用 set_mode_manually 固定模式
    """
    # ==================== 第一步：初始化缓冲区 ====================
    # 首次调用时创建模式相关的缓冲区
    # 这些缓冲区存储在 env 对象上，在整个训练过程中持续存在
    
    # _current_mode: 存储每个环境当前的模式（0-3）
    # 形状: [num_envs]，数据类型: long（整数）
    if not hasattr(env, "_current_mode"):
        env._current_mode = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    
    # _terrain_is_stair: 地形类型标志
    # 形状: [num_envs, 1]，值: 0.0=平地, 1.0=楼梯
    # 使用 [num_envs, 1] 形状是为了方便与其他观测拼接
    if not hasattr(env, "_terrain_is_stair"):
        env._terrain_is_stair = torch.zeros(env.num_envs, 1, device=env.device)
    
    # _sensor_available: 传感器可用标志
    # 形状: [num_envs, 1]，值: 0.0=盲模式, 1.0=有传感器
    # 默认初始化为 1（有传感器），因为大多数情况下传感器是可用的
    if not hasattr(env, "_sensor_available"):
        env._sensor_available = torch.ones(env.num_envs, 1, device=env.device)
    
    # ==================== 第二步：检查是否有环境需要重置 ====================
    # 如果没有环境需要重置，直接返回
    if len(env_ids) == 0:
        return
    
    # 将 env_ids 转换为张量，用于索引操作
    env_ids_tensor = torch.tensor(env_ids, device=env.device, dtype=torch.long)
    
    # ==================== 第三步：设置采样概率 ====================
    # 如果没有指定概率，使用等概率分布
    if mode_probabilities is None:
        mode_probabilities = [1.0 / num_modes] * num_modes
    
    # 将概率列表转换为张量
    probs = torch.tensor(mode_probabilities, device=env.device)
    
    # ==================== 第四步：根据概率采样模式 ====================
    # 使用 torch.multinomial 进行多项式采样
    # 
    # probs.expand(len(env_ids), -1): 将概率扩展为 [num_reset_envs, num_modes]
    # num_samples=1: 每个环境采样一个模式
    # squeeze(-1): 移除最后一个维度，得到 [num_reset_envs]
    sampled_modes = torch.multinomial(
        probs.expand(len(env_ids), -1),
        num_samples=1
    ).squeeze(-1)
    
    # ==================== 第五步：更新模式标志 ====================
    # 将采样的模式写入对应环境的缓冲区
    env._current_mode[env_ids_tensor] = sampled_modes
    
    # ==================== 第六步：根据模式更新辅助标志 ====================
    # 这些辅助标志可以被观测函数直接使用，无需再次解码模式
    
    # 地形类型标志：模式 0/1 是平地（<2），模式 2/3 是楼梯（>=2）
    # sampled_modes >= 2 返回布尔张量，.float() 转换为 0.0/1.0
    # .unsqueeze(1) 添加维度，从 [num_reset_envs] 变为 [num_reset_envs, 1]
    is_stair = (sampled_modes >= 2).float().unsqueeze(1)
    env._terrain_is_stair[env_ids_tensor] = is_stair
    
    # 传感器可用标志：模式 0/2 是盲模式（偶数），模式 1/3 是传感器模式（奇数）
    # sampled_modes % 2 == 1 检查是否为奇数
    has_sensor = (sampled_modes % 2 == 1).float().unsqueeze(1)
    env._sensor_available[env_ids_tensor] = has_sensor


def reset_mode_by_terrain(
    env: ManagerBasedEnv,
    env_ids: Sequence[int],
    stair_terrain_names: list[str] | None = None,
    blind_probability: float = 0.5,
):
    """
    根据当前地形自动设置模式 - 地形感知的模式切换
    
    检测机器人当前所在地形类型，自动设置对应模式。
    这种方法比完全随机更加合理，因为模式与实际地形匹配。
    
    工作流程：
        1. 检测当前地形类型（平地/楼梯）
        2. 随机决定是否使用传感器
        3. 根据地形类型和传感器状态计算模式
    
    模式计算公式：
        mode = is_stair * 2 + use_sensor
        
        示例：
            - 平地 + 盲模式: 0 * 2 + 0 = 0
            - 平地 + 传感器: 0 * 2 + 1 = 1
            - 楼梯 + 盲模式: 1 * 2 + 0 = 2
            - 楼梯 + 传感器: 1 * 2 + 1 = 3
    
    Args:
        env (ManagerBasedEnv): 环境实例
        env_ids (Sequence[int]): 需要重置的环境 ID 列表
        stair_terrain_names (list[str] | None): 楼梯地形的名称列表
            用于识别哪些地形是楼梯，如 ["stairs_up", "stairs_down"]
            目前未实现，保留用于未来扩展
        blind_probability (float): 盲模式的概率，默认 0.5
            - 0.0: 总是使用传感器
            - 1.0: 总是盲模式
            - 0.5: 50% 概率盲模式
    
    Returns:
        None: 直接修改 env 的内部状态
    
    Example:
        在环境配置中使用此函数::
        
            from isaaclab.managers import EventTermCfg as EventTerm
            
            reset_mode = EventTerm(
                func=reset_mode_by_terrain,
                mode="reset",
                params={
                    "stair_terrain_names": ["stairs_up_easy", "stairs_up_medium"],
                    "blind_probability": 0.3,  # 30% 盲模式
                }
            )
    
    Note:
        - 当前实现使用随机地形类型（简化版本）
        - 完整实现应该根据 env.scene.terrain.terrain_levels 判断地形类型
        - 可以通过 stair_terrain_names 参数指定哪些地形名称是楼梯
    
    TODO:
        实现真正的地形类型检测逻辑：
        1. 获取当前环境的地形等级
        2. 根据地形等级查找对应的地形类型
        3. 判断是否为楼梯地形
    """
    # ==================== 第一步：初始化缓冲区 ====================
    if not hasattr(env, "_current_mode"):
        env._current_mode = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    
    # 如果没有环境需要重置，直接返回
    if len(env_ids) == 0:
        return
    
    env_ids_tensor = torch.tensor(env_ids, device=env.device, dtype=torch.long)
    
    # ==================== 第二步：检测地形类型 ====================
    # TODO: 实现地形类型检测逻辑
    # 目前简化为随机选择，实际应该根据 env.scene.terrain 的 terrain_levels 判断
    # 
    # 完整实现示例：
    # terrain = env.scene.terrain
    # terrain_levels = terrain.terrain_levels[env_ids_tensor]
    # terrain_types = terrain.terrain_types[terrain_levels]
    # is_on_stair = torch.isin(terrain_types, stair_terrain_indices)
    
    # 随机决定是否在楼梯上（简化实现）
    # 60% 概率在楼梯上，40% 概率在平地上
    is_on_stair = torch.rand(len(env_ids), device=env.device) > 0.4
    
    # ==================== 第三步：随机决定传感器状态 ====================
    # 根据 blind_probability 决定是否使用传感器
    # torch.rand() 生成 [0, 1) 的均匀分布
    # > blind_probability 表示使用传感器的概率为 (1 - blind_probability)
    use_sensor = torch.rand(len(env_ids), device=env.device) > blind_probability
    
    # ==================== 第四步：计算模式 ====================
    # 模式编码: mode = is_stair * 2 + use_sensor
    # 
    # 真值表：
    # is_stair | use_sensor | mode
    # ---------|------------|------
    #    0     |     0      |   0   (平地盲走)
    #    0     |     1      |   1   (平地传感器)
    #    1     |     0      |   2   (楼梯盲爬)
    #    1     |     1      |   3   (楼梯传感器)
    mode = is_on_stair.long() * 2 + use_sensor.long()
    
    # ==================== 第五步：更新模式标志 ====================
    env._current_mode[env_ids_tensor] = mode


def set_mode_manually(
    env: ManagerBasedEnv,
    env_ids: Sequence[int],
    mode: int,
):
    """
    手动设置指定环境的模式 - 用于测试和部署
    
    将指定环境的模式设置为固定值，不进行随机采样。
    这在以下场景中非常有用：
        - 测试特定模式下的策略行为
        - 部署时根据实际情况选择模式
        - 调试和可视化
    
    使用场景：
        1. 测试盲爬能力：set_mode_manually(env, all_env_ids, mode=2)
        2. 测试传感器模式：set_mode_manually(env, all_env_ids, mode=3)
        3. 混合测试：部分环境盲模式，部分环境传感器模式
    
    Args:
        env (ManagerBasedEnv): 环境实例
        env_ids (Sequence[int]): 需要设置的环境 ID 列表
            可以是所有环境 range(env.num_envs)，也可以是部分环境
        mode (int): 目标模式，取值范围 0-3
            - 0: 平地盲走
            - 1: 平地带传感器
            - 2: 楼梯盲爬
            - 3: 楼梯带传感器
    
    Returns:
        None: 直接修改 env 的内部状态
    
    Example:
        测试盲爬模式::
        
            # 在 play.py 或测试脚本中
            all_env_ids = list(range(env.num_envs))
            set_mode_manually(env, all_env_ids, mode=2)  # 所有环境使用楼梯盲爬模式
        
        在环境配置中使用（固定模式训练）::
        
            from isaaclab.managers import EventTermCfg as EventTerm
            
            reset_mode = EventTerm(
                func=set_mode_manually,
                mode="reset",
                params={"mode": 2}  # 固定为楼梯盲爬模式
            )
    
    Note:
        - 此函数不会更新 _terrain_is_stair 和 _sensor_available 辅助标志
        - 如果需要这些标志，应该在调用后手动更新，或使用 reset_mode_randomly
        - 部署时建议根据实际传感器状态和地形类型动态设置模式
    """
    # ==================== 第一步：初始化缓冲区 ====================
    if not hasattr(env, "_current_mode"):
        env._current_mode = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    
    # 如果没有环境需要设置，直接返回
    if len(env_ids) == 0:
        return
    
    # ==================== 第二步：设置模式 ====================
    env_ids_tensor = torch.tensor(env_ids, device=env.device, dtype=torch.long)
    env._current_mode[env_ids_tensor] = mode
