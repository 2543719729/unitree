"""
===============================================================================
课程学习模块 - 速度命令难度自适应调整
===============================================================================

本文件实现了速度命令的课程学习（Curriculum Learning）机制。

核心思想:
    - 训练初期使用较小的速度命令范围，降低任务难度
    - 随着策略性能提升，自动扩大速度命令范围
    - 最终达到目标的最大速度范围

主要函数:
    - lin_vel_cmd_levels: 自适应调整线速度命令范围
    - ang_vel_cmd_levels: 自适应调整角速度命令范围

调整机制:
    1. 每个 episode 结束时评估策略在速度跟踪任务上的表现
    2. 如果平均奖励超过阈值（80%权重），则扩大命令范围
    3. 扩大幅度为 ±0.1，直至达到 limit_ranges

使用场景:
    - 配合 UniformLevelVelocityCommandCfg 使用
    - 在环境配置的 curriculum 部分注册
===============================================================================
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

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
