"""
统一条件策略的事件函数

包含模式切换、地形检测等事件处理函数。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_mode_randomly(
    env: ManagerBasedEnv,
    env_ids: Sequence[int],
    num_modes: int = 4,
    mode_probabilities: list[float] | None = None,
):
    """
    随机重置模式标志
    
    在每次环境重置时，随机为该环境选择一个运行模式。
    这是条件策略训练的核心：让策略学会根据 mode_flag 调整行为。
    
    模式定义：
        - 模式0: 平地盲走 (flat + blind)
        - 模式1: 平地带传感器 (flat + sensor)
        - 模式2: 楼梯盲爬 (stair + blind)
        - 模式3: 楼梯带传感器 (stair + sensor)
    
    Args:
        env: 环境实例
        env_ids: 需要重置的环境 ID 列表
        num_modes: 模式数量，默认4
        mode_probabilities: 各模式的采样概率，None 时等概率
    """
    # 初始化模式缓冲区（如果不存在）
    if not hasattr(env, "_current_mode"):
        env._current_mode = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    
    # 初始化地形类型标志
    if not hasattr(env, "_terrain_is_stair"):
        env._terrain_is_stair = torch.zeros(env.num_envs, 1, device=env.device)
    
    # 初始化传感器可用标志
    if not hasattr(env, "_sensor_available"):
        env._sensor_available = torch.ones(env.num_envs, 1, device=env.device)
    
    if len(env_ids) == 0:
        return
    
    env_ids_tensor = torch.tensor(env_ids, device=env.device, dtype=torch.long)
    
    # 设置采样概率
    if mode_probabilities is None:
        mode_probabilities = [1.0 / num_modes] * num_modes
    
    probs = torch.tensor(mode_probabilities, device=env.device)
    
    # 根据概率采样模式
    sampled_modes = torch.multinomial(
        probs.expand(len(env_ids), -1),
        num_samples=1
    ).squeeze(-1)
    
    # 更新模式标志
    env._current_mode[env_ids_tensor] = sampled_modes
    
    # 根据模式更新辅助标志
    # 模式 0/1 是平地，模式 2/3 是楼梯
    is_stair = (sampled_modes >= 2).float().unsqueeze(1)
    env._terrain_is_stair[env_ids_tensor] = is_stair
    
    # 模式 0/2 是盲模式，模式 1/3 是传感器模式
    has_sensor = (sampled_modes % 2 == 1).float().unsqueeze(1)
    env._sensor_available[env_ids_tensor] = has_sensor


def reset_mode_by_terrain(
    env: ManagerBasedEnv,
    env_ids: Sequence[int],
    stair_terrain_names: list[str] | None = None,
    blind_probability: float = 0.5,
):
    """
    根据当前地形自动设置模式
    
    检测机器人当前所在地形类型，自动设置对应模式：
        - 平地 + 随机盲/传感器
        - 楼梯 + 随机盲/传感器
    
    Args:
        env: 环境实例
        env_ids: 需要重置的环境 ID 列表
        stair_terrain_names: 楼梯地形的名称列表
        blind_probability: 盲模式的概率
    """
    if not hasattr(env, "_current_mode"):
        env._current_mode = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    
    if len(env_ids) == 0:
        return
    
    env_ids_tensor = torch.tensor(env_ids, device=env.device, dtype=torch.long)
    
    # TODO: 实现地形类型检测逻辑
    # 目前简化为随机选择
    # 实际应该根据 env.scene.terrain 的 terrain_levels 判断
    
    # 随机决定是否使用传感器
    use_sensor = torch.rand(len(env_ids), device=env.device) > blind_probability
    
    # 随机决定是否在楼梯上（简化实现，实际应检测地形）
    is_on_stair = torch.rand(len(env_ids), device=env.device) > 0.4  # 60% 楼梯
    
    # 计算模式: mode = is_stair * 2 + use_sensor
    mode = is_on_stair.long() * 2 + use_sensor.long()
    
    env._current_mode[env_ids_tensor] = mode


def set_mode_manually(
    env: ManagerBasedEnv,
    env_ids: Sequence[int],
    mode: int,
):
    """
    手动设置指定环境的模式
    
    用于测试或部署时手动切换模式。
    
    Args:
        env: 环境实例
        env_ids: 需要设置的环境 ID 列表
        mode: 目标模式 (0-3)
    """
    if not hasattr(env, "_current_mode"):
        env._current_mode = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    
    if len(env_ids) == 0:
        return
    
    env_ids_tensor = torch.tensor(env_ids, device=env.device, dtype=torch.long)
    env._current_mode[env_ids_tensor] = mode

