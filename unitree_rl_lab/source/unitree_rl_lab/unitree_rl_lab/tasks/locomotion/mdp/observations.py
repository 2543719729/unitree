from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    """步态相位观测：返回 sin/cos 编码的步态周期"""
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def mode_flag(env: ManagerBasedRLEnv, num_modes: int = 4) -> torch.Tensor:
    """
    模式标志观测：返回 one-hot 编码的当前模式

    用于条件策略，让网络知道当前处于哪种运行模式：
        - 模式0 [1,0,0,0]: 平地盲走
        - 模式1 [0,1,0,0]: 平地带传感器
        - 模式2 [0,0,1,0]: 楼梯盲爬
        - 模式3 [0,0,0,1]: 楼梯带传感器

    Args:
        env: 环境实例
        num_modes: 模式数量，默认4

    Returns:
        shape (num_envs, num_modes) 的 one-hot 张量
    """
    # 初始化模式缓冲区（如果不存在）
    if not hasattr(env, "_current_mode"):
        # 默认模式0，后续由环境根据地形/传感器状态设置
        env._current_mode = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    # 生成 one-hot 编码
    mode_onehot = torch.zeros(env.num_envs, num_modes, device=env.device)
    mode_onehot.scatter_(1, env._current_mode.unsqueeze(1), 1.0)

    return mode_onehot


def terrain_type_flag(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    地形类型标志：返回当前地形是否为楼梯

    用于辅助模式判断：
        - 0: 平地
        - 1: 楼梯

    Returns:
        shape (num_envs, 1) 的张量
    """
    if not hasattr(env, "_terrain_is_stair"):
        env._terrain_is_stair = torch.zeros(env.num_envs, 1, device=env.device)

    return env._terrain_is_stair


def sensor_available_flag(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    传感器可用标志：返回 height_scan 是否可用

    用于辅助模式判断和降级逻辑：
        - 0: 传感器不可用（盲模式）
        - 1: 传感器可用

    Returns:
        shape (num_envs, 1) 的张量
    """
    if not hasattr(env, "_sensor_available"):
        env._sensor_available = torch.ones(env.num_envs, 1, device=env.device)

    return env._sensor_available


def conditional_height_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    条件高度扫描：盲模式时输出置零

    根据 _current_mode 决定是否使用 height_scan:
        - 模式0/2 (盲模式): 输出全零
        - 模式1/3 (传感器模式): 输出真实 height_scan

    这让策略网络学会：当 mode_flag 指示盲模式时，忽略 height_scan 输入

    Args:
        env: 环境实例
        sensor_cfg: 高度扫描器配置

    Returns:
        shape (num_envs, num_rays) 的高度扫描张量
    """
    from isaaclab.sensors import RayCaster

    # 获取传感器
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    # 计算相对高度（与标准 height_scan 相同）
    heights = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - 0.5

    # 初始化模式缓冲区（如果不存在）
    if not hasattr(env, "_current_mode"):
        env._current_mode = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    # 创建盲模式掩码 (模式0和模式2是盲模式)
    blind_mask = (env._current_mode == 0) | (env._current_mode == 2)

    # 盲模式时置零
    heights = torch.where(
        blind_mask.unsqueeze(1).expand_as(heights),
        torch.zeros_like(heights),
        heights
    )

    return heights


# 需要导入 SceneEntityCfg
from isaaclab.managers import SceneEntityCfg
