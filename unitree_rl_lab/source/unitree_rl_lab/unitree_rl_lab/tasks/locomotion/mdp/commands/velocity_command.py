"""
===============================================================================
速度命令配置模块
===============================================================================

本文件定义了用于课程学习的速度命令配置类。

主要功能:
    - 在基础的 UniformVelocityCommandCfg 基础上扩展
    - 添加 limit_ranges 字段，用于课程学习中的命令难度递增
    - 支持自适应调整速度命令范围

使用场景:
    - 课程学习中逐步提高速度指令的难度
    - 配合 curriculums.py 中的 lin_vel_cmd_levels 和 ang_vel_cmd_levels 函数
===============================================================================
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    """
    分级速度命令配置类
    
    扩展自 UniformVelocityCommandCfg，添加了 limit_ranges 字段用于课程学习。
    
    课程学习机制:
        - ranges: 当前训练使用的速度范围（动态调整）
        - limit_ranges: 最终目标速度范围（上限）
        - 训练过程中，ranges 会根据策略表现逐步接近 limit_ranges
    
    使用示例:
        ```python
        command_cfg = UniformLevelVelocityCommandCfg(
            ranges=Ranges(
                lin_vel_x=(-0.5, 0.5),  # 初始线速度范围
                ang_vel_z=(-0.5, 0.5),  # 初始角速度范围
            ),
            limit_ranges=Ranges(
                lin_vel_x=(-1.0, 1.0),  # 最终线速度范围
                ang_vel_z=(-1.0, 1.0),  # 最终角速度范围
            )
        )
        ```
    """
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING  # 速度命令的最大限制范围（必须设置）
