# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
课程学习（Curriculum Learning）模块 - 用于创建学习环境的课程函数。

本模块提供了用于强化学习训练中课程学习的通用函数。课程学习是一种训练策略，
通过逐步增加任务难度来帮助智能体更有效地学习复杂任务。

主要功能：
    - 根据机器人的表现动态调整地形难度
    - 表现好的机器人会被分配到更难的地形
    - 表现差的机器人会被分配到更简单的地形

使用方式：
    这些函数可以传递给 :class:`isaaclab.managers.CurriculumTermCfg` 对象，
    以启用该函数引入的课程学习机制。

典型应用场景：
    - 四足机器人运动训练
    - 双足机器人行走训练
    - 复杂地形导航训练
"""

from __future__ import annotations  # 启用延迟注解评估，允许在类型提示中使用尚未定义的类型

import torch  # PyTorch 深度学习框架，用于张量计算
from collections.abc import Sequence  # 抽象基类，用于类型提示中表示序列类型
from typing import TYPE_CHECKING  # 用于条件导入，仅在类型检查时导入某些模块

# Isaac Lab 核心模块导入
from isaaclab.assets import Articulation  # 关节体资产类，用于表示机器人等多关节物体
from isaaclab.managers import SceneEntityCfg  # 场景实体配置类，用于指定场景中的实体
from isaaclab.terrains import TerrainImporter  # 地形导入器，用于管理和生成各种地形

# 条件导入：仅在类型检查时导入，避免循环导入问题
# TYPE_CHECKING 在运行时为 False，但在类型检查工具（如 mypy）运行时为 True
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv  # 基于管理器的强化学习环境类


def terrain_levels_vel(
    env: ManagerBasedRLEnv, 
    env_ids: Sequence[int], 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    基于速度命令的地形难度课程学习函数。
    
    该函数根据机器人在被命令以期望速度移动时实际行走的距离来调整地形难度。
    这是一种自适应课程学习策略，能够根据每个机器人的表现动态调整训练难度。
    
    课程学习策略：
        - 升级条件：当机器人行走距离超过地形尺寸的一半时，升级到更难的地形
        - 降级条件：当机器人行走距离不足命令速度要求距离的50%时，降级到更简单的地形
        - 保持条件：介于两者之间时，保持当前地形难度
    
    算法原理：
        1. 计算机器人从起点到当前位置的欧几里得距离
        2. 将该距离与两个阈值进行比较：
            - 上限阈值：地形尺寸的一半（terrain_size / 2）
            - 下限阈值：命令速度 × 最大回合时长 × 0.5
        3. 根据比较结果决定是否调整地形难度
    
    Args:
        env (ManagerBasedRLEnv): 基于管理器的强化学习环境实例。
            包含场景、命令管理器等核心组件。
        env_ids (Sequence[int]): 需要更新地形等级的环境ID序列。
            通常是在当前时间步结束（terminated 或 truncated）的环境。
        asset_cfg (SceneEntityCfg, optional): 场景实体配置，用于指定要评估的机器人资产。
            默认值为 SceneEntityCfg("robot")，即名为 "robot" 的实体。
    
    Returns:
        torch.Tensor: 所有环境的平均地形等级（标量张量）。
            用于监控和记录训练过程中的课程进度。
    
    Note:
        此函数仅适用于 ``generator`` 类型的地形。
        关于不同地形类型的更多信息，请参阅 :class:`isaaclab.terrains.TerrainImporter` 类。
    
    Example:
        在环境配置中使用此课程函数::
        
            from isaaclab.managers import CurriculumTermCfg
            
            curriculum = CurriculumTermCfg(
                func=terrain_levels_vel,
                params={"asset_cfg": SceneEntityCfg("robot")}
            )
    
    See Also:
        - :class:`isaaclab.terrains.TerrainImporter`: 地形导入和管理
        - :class:`isaaclab.managers.CurriculumManager`: 课程管理器
    """
    
    # ==================== 第一步：提取所需的数据 ====================
    # 从场景中获取机器人资产（Articulation 类型）
    # asset_cfg.name 默认为 "robot"，可以通过配置修改
    # 类型注解 `: Articulation` 用于启用 IDE 的类型提示功能
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 从场景中获取地形对象
    # TerrainImporter 负责管理地形的生成、更新和环境原点的分配
    terrain: TerrainImporter = env.scene.terrain
    
    # 从命令管理器获取基础速度命令
    # "base_velocity" 是速度命令的标准名称，包含 [vx, vy, omega_z] 三个分量
    # vx: 前进速度, vy: 侧向速度, omega_z: 偏航角速度
    command = env.command_manager.get_command("base_velocity")
    
    # ==================== 第二步：计算机器人行走距离 ====================
    # 计算机器人从环境原点到当前位置的水平距离（仅考虑 x, y 坐标）
    # 
    # asset.data.root_pos_w[env_ids, :2]: 
    #   - root_pos_w 是机器人根链接在世界坐标系中的位置 [x, y, z]
    #   - [env_ids, :2] 选择指定环境的 x, y 坐标
    #
    # env.scene.env_origins[env_ids, :2]:
    #   - env_origins 是每个环境的原点位置
    #   - 机器人在每个回合开始时会被重置到这个原点附近
    #
    # torch.norm(..., dim=1):
    #   - 计算每个环境中机器人位置与原点之间的欧几里得距离
    #   - dim=1 表示沿着坐标轴维度计算范数
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    
    # ==================== 第三步：确定升级条件 ====================
    # 判断哪些机器人应该升级到更难的地形
    # 
    # 升级条件：行走距离 > 地形尺寸的一半
    # 
    # terrain.cfg.terrain_generator.size[0]:
    #   - 地形生成器配置中的地形尺寸（通常是正方形地形的边长）
    #   - size[0] 取 x 方向的尺寸
    #
    # 设计理念：如果机器人能够穿越地形的一半以上，说明它已经掌握了当前难度，
    # 可以尝试更具挑战性的地形
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    
    # ==================== 第四步：确定降级条件 ====================
    # 判断哪些机器人应该降级到更简单的地形
    # 
    # 降级条件：行走距离 < 命令速度要求距离的 50%
    # 
    # command[env_ids, :2]: 
    #   - 获取指定环境的速度命令的 x, y 分量 [vx, vy]
    #
    # torch.norm(command[env_ids, :2], dim=1):
    #   - 计算命令速度的水平分量大小（速度幅值）
    #
    # env.max_episode_length_s:
    #   - 最大回合时长（秒）
    #
    # 期望距离 = 速度幅值 × 最大回合时长
    # 降级阈值 = 期望距离 × 0.5
    #
    # 设计理念：如果机器人只走了期望距离的一半都不到，说明它在当前难度下
    # 表现不佳，需要在更简单的地形上继续练习
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    
    # 确保升级和降级互斥：已经标记为升级的机器人不能同时被标记为降级
    # ~move_up 是 move_up 的逻辑非（取反）
    # *= 是原地乘法，相当于 move_down = move_down & (~move_up)
    # 这确保了每个机器人最多只会执行一种操作（升级、降级或保持不变）
    move_down *= ~move_up
    
    # ==================== 第五步：更新地形等级 ====================
    # 调用地形导入器的方法来更新环境原点
    # 
    # 该方法会：
    #   1. 对于 move_up=True 的环境，将其地形等级 +1（更难的地形）
    #   2. 对于 move_down=True 的环境，将其地形等级 -1（更简单的地形）
    #   3. 根据新的地形等级，重新分配环境原点到对应难度的地形区域
    #
    # 地形等级通常对应于不同类型或难度的地形，例如：
    #   - 等级 0: 平坦地面
    #   - 等级 1: 轻微起伏
    #   - 等级 2: 台阶
    #   - 等级 3: 斜坡
    #   - 等级 4: 复杂障碍物
    #   - ...
    terrain.update_env_origins(env_ids, move_up, move_down)
    
    # ==================== 第六步：返回平均地形等级 ====================
    # 计算并返回所有环境的平均地形等级
    # 
    # terrain.terrain_levels: 
    #   - 形状为 [num_envs] 的张量，存储每个环境当前的地形等级
    #
    # .float(): 
    #   - 将整数类型转换为浮点数，以便计算平均值
    #
    # torch.mean(...):
    #   - 计算所有环境地形等级的平均值
    #
    # 返回值用途：
    #   - 用于监控训练进度（平均等级上升说明整体表现在提升）
    #   - 可以记录到 TensorBoard 等可视化工具中
    #   - 帮助判断课程学习是否正常工作
    return torch.mean(terrain.terrain_levels.float())
