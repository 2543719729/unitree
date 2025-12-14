"""
===============================================================================
课程学习模块 - 速度命令与地形难度自适应调整
===============================================================================

本文件实现了强化学习训练中的课程学习（Curriculum Learning）机制。

核心思想:
    - 训练初期使用较小的速度命令范围和简单地形，降低任务难度
    - 随着策略性能提升，自动扩大速度命令范围和地形难度
    - 最终达到目标的最大速度范围和最难地形

主要函数:
    速度命令课程:
        - lin_vel_cmd_levels: 自适应调整线速度命令范围
        - ang_vel_cmd_levels: 自适应调整角速度命令范围
    
    地形难度课程:
        - terrain_levels_vel: 基于速度跟踪的地形难度调整（原版）
        - terrain_levels_climb: 基于攀爬进度的地形难度调整（盲爬楼梯专用）
        - terrain_levels_height: 基于高度增益的地形难度调整（纯楼梯任务）

调整机制:
    速度命令:
        1. 每个 episode 结束时评估策略在速度跟踪任务上的表现
        2. 如果平均奖励超过阈值（80%权重），则扩大命令范围
        3. 扩大幅度为 ±0.1，直至达到 limit_ranges
    
    地形难度:
        1. 每个 episode 结束时评估机器人的攀爬进度
        2. 表现好则升级到更难地形，表现差则降级到简单地形
        3. 盲爬任务使用高度增益作为主要评估指标

使用场景:
    - 配合 UniformLevelVelocityCommandCfg 使用
    - 在环境配置的 curriculum 部分注册
    - 盲爬楼梯任务使用 terrain_levels_climb 或 terrain_levels_height
===============================================================================
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

# Isaac Lab 核心模块导入
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    """
    线速度命令难度自适应调整
    
    根据机器人在速度跟踪任务上的表现，自动调整线速度（x, y 方向）命令的范围。
    
    工作流程:
        1. 获取当前速度命令配置（ranges 和 limit_ranges）
        2. 计算本轮 episode 的平均速度跟踪奖励
        3. 如果奖励 > 权重*0.8，则扩大命令范围 ±0.1
        4. 确保不超过 limit_ranges 的限制
    
    Args:
        env: 强化学习环境实例
        env_ids: 本次重置的环境 ID 列表
        reward_term_name: 用于评估的奖励项名称，默认 "track_lin_vel_xy"
    
    Returns:
        当前线速度 x 的最大值（用于课程可视化）
    
    示例:
        初始 ranges.lin_vel_x = [-0.5, 0.5]
        limit_ranges.lin_vel_x = [-1.5, 1.5]
        
        经过多次调整:
        [-0.5, 0.5] -> [-0.6, 0.6] -> [-0.7, 0.7] -> ... -> [-1.5, 1.5]
    """
    # 获取速度命令配置
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges  # 当前使用的速度范围
    limit_ranges = command_term.cfg.limit_ranges  # 最大限制范围

    # 计算平均奖励（归一化到每秒）
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    # 每个 episode 结束时检查是否需要调整
    if env.common_step_counter % env.max_episode_length == 0:
        # 如果表现良好（奖励 > 权重的80%），则扩大命令范围
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)  # 扩大 ±0.1
            # 调整 x 方向线速度范围
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],  # 下限
                limit_ranges.lin_vel_x[1],  # 上限
            ).tolist()
            # 调整 y 方向线速度范围
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    # 返回当前最大线速度（用于课程可视化）
    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    """
    角速度命令难度自适应调整
    
    根据机器人在角速度跟踪任务上的表现，自动调整角速度（z 轴旋转）命令的范围。
    
    工作流程:
        与 lin_vel_cmd_levels 类似，但针对角速度（偏航角速度）
    
    Args:
        env: 强化学习环境实例
        env_ids: 本次重置的环境 ID 列表
        reward_term_name: 用于评估的奖励项名称，默认 "track_ang_vel_z"
    
    Returns:
        当前角速度 z 的最大值（用于课程可视化）
    
    示例:
        初始 ranges.ang_vel_z = [-0.5, 0.5] rad/s
        limit_ranges.ang_vel_z = [-2.0, 2.0] rad/s
        
        经过多次调整:
        [-0.5, 0.5] -> [-0.6, 0.6] -> [-0.7, 0.7] -> ... -> [-2.0, 2.0]
    """
    # 获取速度命令配置
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges  # 当前使用的速度范围
    limit_ranges = command_term.cfg.limit_ranges  # 最大限制范围

    # 计算平均奖励（归一化到每秒）
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    # 每个 episode 结束时检查是否需要调整
    if env.common_step_counter % env.max_episode_length == 0:
        # 如果表现良好（奖励 > 权重的80%），则扩大命令范围
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)  # 扩大 ±0.1 rad/s
            # 调整 z 轴角速度范围（偏航角速度）
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],  # 下限
                limit_ranges.ang_vel_z[1],  # 上限
            ).tolist()

    # 返回当前最大角速度（用于课程可视化）
    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


# ============================================================================
#                     地形难度课程学习函数（盲爬楼梯专用）
# ============================================================================

def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    基于速度命令的地形难度课程学习函数（原版，适合平地速度跟踪任务）
    
    该函数根据机器人在被命令以期望速度移动时实际行走的距离来调整地形难度。
    
    课程学习策略：
        - 升级条件：当机器人行走距离超过地形尺寸的一半时，升级到更难的地形
        - 降级条件：当机器人行走距离不足命令速度要求距离的50%时，降级到更简单的地形
        - 保持条件：介于两者之间时，保持当前地形难度
    
    Args:
        env: 基于管理器的强化学习环境实例
        env_ids: 需要更新地形等级的环境ID序列
        asset_cfg: 场景实体配置，用于指定要评估的机器人资产
    
    Returns:
        所有环境的平均地形等级（标量张量）
    
    Note:
        此函数仅适用于 ``generator`` 类型的地形。
        对于盲爬楼梯任务，建议使用 terrain_levels_climb 或 terrain_levels_height。
    """
    # 从场景中获取机器人资产和地形对象
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    
    # 计算机器人从环境原点到当前位置的水平距离（仅考虑 x, y 坐标）
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    
    # 升级条件：行走距离 > 地形尺寸的一半
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    
    # 降级条件：行走距离 < 命令速度要求距离的 50%
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up  # 确保升级和降级互斥
    
    # 更新地形等级
    terrain.update_env_origins(env_ids, move_up, move_down)
    
    return torch.mean(terrain.terrain_levels.float())


def terrain_levels_climb(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_weight: float = 2.0,
    forward_weight: float = 1.0,
    upgrade_threshold_ratio: float = 0.3,
    downgrade_threshold: float = 0.5,
) -> torch.Tensor:
    """
    基于攀爬进度的地形难度课程学习函数（盲爬楼梯专用）
    
    该函数综合考虑机器人的前进距离和高度增益来评估攀爬进度，
    特别适合盲爬楼梯任务，因为它不依赖速度命令，而是关注实际的攀爬表现。
    
    评估指标：
        progress = forward_distance × forward_weight + height_gain × height_weight
    
    课程学习策略：
        - 升级条件：progress > terrain_size × upgrade_threshold_ratio
        - 降级条件：progress < downgrade_threshold（绝对阈值）
        - 保持条件：介于两者之间
    
    设计理念：
        1. 高度增益权重更大（默认 2.0），因为楼梯任务的核心是向上攀爬
        2. 使用绝对阈值作为降级条件，不依赖速度命令
        3. 更宽容的降级条件，允许机器人多次尝试
    
    Args:
        env: 基于管理器的强化学习环境实例
        env_ids: 需要更新地形等级的环境ID序列
        asset_cfg: 场景实体配置，用于指定要评估的机器人资产
        height_weight: 高度增益的权重，默认 2.0（楼梯任务中高度更重要）
        forward_weight: 前进距离的权重，默认 1.0
        upgrade_threshold_ratio: 升级阈值比例，默认 0.3（地形尺寸的 30%）
        downgrade_threshold: 降级阈值（绝对值，米），默认 0.5
    
    Returns:
        所有环境的平均地形等级（标量张量）
    
    Example:
        在环境配置中使用此课程函数::
        
            from isaaclab.managers import CurriculumTermCfg
            
            terrain_levels = CurriculumTermCfg(
                func=terrain_levels_climb,
                params={
                    "height_weight": 2.0,
                    "forward_weight": 1.0,
                    "upgrade_threshold_ratio": 0.3,
                    "downgrade_threshold": 0.5,
                }
            )
    
    Note:
        此函数仅适用于 ``generator`` 类型的地形。
        对于纯楼梯任务（不关心水平移动），可以使用 terrain_levels_height。
    """
    # ==================== 第一步：提取所需的数据 ====================
    # 从场景中获取机器人资产（Articulation 类型）
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 从场景中获取地形对象
    terrain: TerrainImporter = env.scene.terrain
    
    # ==================== 第二步：计算攀爬进度 ====================
    # 计算机器人从环境原点到当前位置的3D位移
    # displacement 形状: [num_env_ids, 3]，包含 [dx, dy, dz]
    displacement = asset.data.root_pos_w[env_ids, :3] - env.scene.env_origins[env_ids, :3]
    
    # 前进距离（x 方向，假设机器人面向 +x 方向）
    # 注意：这里使用 x 方向作为前进方向，如果机器人面向其他方向，需要调整
    forward_distance = displacement[:, 0]
    
    # 高度增益（z 方向）
    # 正值表示机器人向上移动，负值表示向下移动
    height_gain = displacement[:, 2]
    
    # 综合进度评分
    # 高度增益的权重更大，因为楼梯任务的核心是向上攀爬
    progress = forward_distance * forward_weight + height_gain * height_weight
    
    # ==================== 第三步：确定升级条件 ====================
    # 升级条件：综合进度超过地形尺寸的一定比例
    #
    # 设计理念：
    #   - 使用地形尺寸作为参考，而非固定阈值
    #   - 默认 30% 的地形尺寸，比原版的 50% 更宽松
    #   - 因为楼梯任务更难，需要更宽容的升级条件
    upgrade_threshold = terrain.cfg.terrain_generator.size[0] * upgrade_threshold_ratio
    move_up = progress > upgrade_threshold
    
    # ==================== 第四步：确定降级条件 ====================
    # 降级条件：综合进度低于绝对阈值
    #
    # 设计理念：
    #   - 使用绝对阈值，不依赖速度命令
    #   - 默认 0.5 米，只有几乎没有进展才降级
    #   - 这允许机器人在困难地形上多次尝试
    move_down = progress < downgrade_threshold
    
    # 确保升级和降级互斥
    move_down *= ~move_up
    
    # ==================== 第五步：更新地形等级 ====================
    terrain.update_env_origins(env_ids, move_up, move_down)
    
    # ==================== 第六步：返回平均地形等级 ====================
    return torch.mean(terrain.terrain_levels.float())


def terrain_levels_height(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    upgrade_height: float = 0.3,
    downgrade_height: float = 0.0,
) -> torch.Tensor:
    """
    纯基于高度增益的地形难度课程学习函数
    
    该函数仅使用高度增益作为评估指标，特别适合纯楼梯任务，
    不关心机器人的水平移动距离。
    
    课程学习策略：
        - 升级条件：height_gain > upgrade_height
        - 降级条件：height_gain < downgrade_height
        - 保持条件：介于两者之间
    
    设计理念：
        1. 简单直接：只关注高度变化
        2. 精确控制：可以精确设置升级/降级的高度阈值
        3. 适合纯楼梯任务：不考虑水平移动
    
    Args:
        env: 基于管理器的强化学习环境实例
        env_ids: 需要更新地形等级的环境ID序列
        asset_cfg: 场景实体配置，用于指定要评估的机器人资产
        upgrade_height: 升级所需的高度增益（米），默认 0.3（约 2-3 级台阶）
        downgrade_height: 降级阈值（米），默认 0.0（没有任何高度增益才降级）
    
    Returns:
        所有环境的平均地形等级（标量张量）
    
    Example:
        在环境配置中使用此课程函数::
        
            from isaaclab.managers import CurriculumTermCfg
            
            terrain_levels = CurriculumTermCfg(
                func=terrain_levels_height,
                params={
                    "upgrade_height": 0.3,   # 爬升 30cm 才升级
                    "downgrade_height": 0.0, # 没有爬升才降级
                }
            )
    
    Note:
        此函数仅适用于 ``generator`` 类型的地形。
        upgrade_height 的设置应该考虑楼梯的台阶高度：
            - 简单楼梯：8-12cm 台阶，upgrade_height = 0.2-0.3
            - 中等楼梯：10-16cm 台阶，upgrade_height = 0.3-0.4
            - 困难楼梯：14-18cm 台阶，upgrade_height = 0.4-0.5
    """
    # ==================== 第一步：提取所需的数据 ====================
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    
    # ==================== 第二步：计算高度增益 ====================
    # 计算机器人从环境原点到当前位置的高度变化
    # height_gain 形状: [num_env_ids]
    height_gain = asset.data.root_pos_w[env_ids, 2] - env.scene.env_origins[env_ids, 2]
    
    # ==================== 第三步：确定升级条件 ====================
    # 升级条件：高度增益超过阈值
    move_up = height_gain > upgrade_height
    
    # ==================== 第四步：确定降级条件 ====================
    # 降级条件：高度增益低于阈值
    # 默认 0.0，表示只有没有任何高度增益（甚至下降）才降级
    move_down = height_gain < downgrade_height
    
    # 确保升级和降级互斥
    move_down *= ~move_up
    
    # ==================== 第五步：更新地形等级 ====================
    terrain.update_env_origins(env_ids, move_up, move_down)
    
    # ==================== 第六步：返回平均地形等级 ====================
    return torch.mean(terrain.terrain_levels.float())


def terrain_levels_survival(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    survival_ratio_upgrade: float = 0.8,
    survival_ratio_downgrade: float = 0.3,
) -> torch.Tensor:
    """
    基于存活时间的地形难度课程学习函数
    
    该函数使用机器人的存活时间比例作为评估指标，
    适合需要长时间稳定运动的任务。
    
    课程学习策略：
        - 升级条件：存活时间 > max_episode_length × survival_ratio_upgrade
        - 降级条件：存活时间 < max_episode_length × survival_ratio_downgrade
        - 保持条件：介于两者之间
    
    设计理念：
        1. 关注稳定性：能够长时间存活说明策略稳定
        2. 适合困难任务：在困难地形上存活本身就是成功
        3. 可以与其他课程函数组合使用
    
    Args:
        env: 基于管理器的强化学习环境实例
        env_ids: 需要更新地形等级的环境ID序列
        asset_cfg: 场景实体配置（此函数未使用，保留以保持接口一致）
        survival_ratio_upgrade: 升级所需的存活时间比例，默认 0.8（80%）
        survival_ratio_downgrade: 降级阈值的存活时间比例，默认 0.3（30%）
    
    Returns:
        所有环境的平均地形等级（标量张量）
    
    Example:
        在环境配置中使用此课程函数::
        
            from isaaclab.managers import CurriculumTermCfg
            
            terrain_levels = CurriculumTermCfg(
                func=terrain_levels_survival,
                params={
                    "survival_ratio_upgrade": 0.8,   # 存活 80% 时间才升级
                    "survival_ratio_downgrade": 0.3, # 存活不足 30% 时间则降级
                }
            )
    """
    # ==================== 第一步：提取所需的数据 ====================
    terrain: TerrainImporter = env.scene.terrain
    
    # ==================== 第二步：计算存活时间比例 ====================
    # episode_length_buf 存储每个环境当前 episode 的步数
    # max_episode_length 是最大允许的步数
    survival_ratio = env.episode_length_buf[env_ids].float() / env.max_episode_length
    
    # ==================== 第三步：确定升级条件 ====================
    # 升级条件：存活时间比例超过阈值
    move_up = survival_ratio > survival_ratio_upgrade
    
    # ==================== 第四步：确定降级条件 ====================
    # 降级条件：存活时间比例低于阈值
    move_down = survival_ratio < survival_ratio_downgrade
    
    # 确保升级和降级互斥
    move_down *= ~move_up
    
    # ==================== 第五步：更新地形等级 ====================
    terrain.update_env_origins(env_ids, move_up, move_down)
    
    # ==================== 第六步：返回平均地形等级 ====================
    return torch.mean(terrain.terrain_levels.float())
