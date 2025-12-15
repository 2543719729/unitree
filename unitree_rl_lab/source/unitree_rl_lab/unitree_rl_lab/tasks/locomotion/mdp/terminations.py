"""
===============================================================================
自定义终止条件模块
===============================================================================

本模块定义了适用于复杂地形（如楼梯）的终止条件函数。

主要函数:
    - root_height_below_terrain_minimum: 基于相对地形高度的终止检测
    - root_height_below_terrain_minimum_with_grace: 带保护期的高度检测
    - bad_orientation_with_grace: 带保护期的姿态检测

设计原因:
    Isaac Lab 原生的 root_height_below_minimum 使用世界坐标系绝对高度，
    只适用于平地。在楼梯等非平地地形上，机器人摔倒后绝对高度可能仍然
    大于阈值，导致检测失效。本模块提供了基于相对高度的替代方案。

    此外，添加了带保护期（grace period）的终止函数，在 episode 开始的
    前几步不触发终止，给机器人时间稳定初始姿态。

===============================================================================
"""

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def root_height_below_terrain_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """基于相对地形高度的终止条件
    
    当机器人基座相对于其所在地形的高度低于阈值时终止。
    
    与原生 root_height_below_minimum 的区别:
        - 原生函数: 使用世界坐标系绝对高度 (root_pos_w[:, 2] < minimum_height)
        - 本函数: 使用相对于地形原点的高度 (root_pos_w[:, 2] - env_origin_z < minimum_height)
    
    工作原理:
        1. 获取机器人当前世界坐标系 Z 高度
        2. 获取该环境所在地形的原点 Z 高度（来自 env.scene.env_origins）
        3. 计算相对高度 = 机器人高度 - 地形原点高度
        4. 如果相对高度 < minimum_height，则终止
    
    适用场景:
        - 楼梯地形
        - 起伏地形
        - 任何非平面地形
    
    Args:
        env: 环境实例
        minimum_height: 相对于地形的最小高度阈值（米）
                       例如 0.2 表示机器人基座距离地形原点低于 0.2m 时终止
        asset_cfg: 机器人资产配置
    
    Returns:
        布尔张量，形状为 (num_envs,)，True 表示该环境应终止
    
    Example:
        在配置中使用::
        
            base_height = DoneTerm(
                func=mdp.root_height_below_terrain_minimum,
                params={"minimum_height": 0.2},
            )
    
    Note:
        env_origins 的 Z 坐标在课程学习中会随着地形难度变化而更新，
        因此本函数能够自适应不同难度的地形。
    """
    # 获取机器人资产
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 获取机器人世界坐标系 Z 高度
    robot_height_world = asset.data.root_pos_w[:, 2]
    
    # 获取每个环境的地形原点 Z 高度
    # env.scene.env_origins 形状为 (num_envs, 3)，包含 [x, y, z]
    terrain_origin_height = env.scene.env_origins[:, 2]
    
    # 计算相对高度
    relative_height = robot_height_world - terrain_origin_height
    
    # 判断是否低于阈值
    return relative_height < minimum_height


def root_height_below_terrain_minimum_with_grace(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    grace_steps: int = 5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """带保护期的相对地形高度终止条件
    
    在 episode 开始的前 grace_steps 步不触发终止，给机器人时间稳定。
    
    Args:
        env: 环境实例
        minimum_height: 相对于地形的最小高度阈值（米）
        grace_steps: 保护期步数，在此期间不触发终止
        asset_cfg: 机器人资产配置
    
    Returns:
        布尔张量，形状为 (num_envs,)
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 计算相对高度
    robot_height_world = asset.data.root_pos_w[:, 2]
    terrain_origin_height = env.scene.env_origins[:, 2]
    relative_height = robot_height_world - terrain_origin_height
    
    # 高度条件
    height_violation = relative_height < minimum_height
    
    # 保护期条件：episode_length_buf 记录当前 episode 已进行的步数
    past_grace_period = env.episode_length_buf >= grace_steps
    
    # 只有过了保护期且违反高度条件才终止
    return height_violation & past_grace_period


def bad_orientation_with_grace(
    env: ManagerBasedRLEnv,
    limit_angle: float,
    grace_steps: int = 5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """带保护期的姿态异常终止条件
    
    在 episode 开始的前 grace_steps 步不触发终止，给机器人时间稳定。
    
    Args:
        env: 环境实例
        limit_angle: 最大允许倾斜角（弧度）
        grace_steps: 保护期步数
        asset_cfg: 机器人资产配置
    
    Returns:
        布尔张量，形状为 (num_envs,)
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 计算倾斜角度
    orientation_violation = torch.acos(-asset.data.projected_gravity_b[:, 2]).abs() > limit_angle
    
    # 保护期条件
    past_grace_period = env.episode_length_buf >= grace_steps
    
    # 只有过了保护期且违反姿态条件才终止
    return orientation_violation & past_grace_period
