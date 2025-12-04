"""
===============================================================================
奖励函数模块
===============================================================================

本文件定义了用于机器人运动控制训练的各种奖励函数。

奖励函数分类:
    1. 关节惩罚 (Joint penalties): 能量消耗、站立静止等
    2. 机器人本体奖励 (Robot): 姿态控制、方向对齐等
    3. 足部奖励 (Feet rewards): 接触、抬腿高度、碰撞检测等
    4. 步态奖励 (Feet Gait rewards): 步态同步、腾空时间等
    5. 楼梯攀爬奖励 (Stair climbing): 向上进展、前向进度等

设计原则:
    - 奖励函数返回值应该是归一化的（便于权重调整）
    - 惩罚性奖励应返回正值（在配置中设置负权重）
    - 所有函数返回形状为 (num_envs,) 的张量

使用方法:
    在环境配置的 RewardsCfg 中注册奖励项，并设置权重
===============================================================================
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ============================================================================
#                           关节惩罚函数
# ============================================================================


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    能量消耗惩罚：惩罚机器人关节的能量消耗
    
    能量计算公式: E = Σ|τ| * |ω|
    其中 τ 是关节力矩，ω 是关节角速度
    
    Args:
        env: 环境实例
        asset_cfg: 机器人资产配置
    
    Returns:
        每个环境的能量消耗值
    """
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]  # 关节角速度
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]  # 关节施加的力矩
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    站立静止奖励：在速度命令为零时，奖励保持默认姿态
    
    鼓励机器人在没有移动命令时保持静止站立姿态，避免不必要的晃动。
    
    Args:
        env: 环境实例
        command_name: 速度命令名称
        asset_cfg: 机器人资产配置
    
    Returns:
        关节位置与默认位置的偏差（仅在命令速度 < 0.1 时生效）
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 计算关节位置与默认位置的偏差
    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    # 仅在速度命令很小时应用此奖励
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


# ============================================================================
#                           机器人本体奖励
# ============================================================================


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    姿态对齐奖励：使用 L2 平方核奖励机器人重力方向与期望方向对齐
    
    通过比较机器人感受到的重力方向与期望重力方向（通常为 [0, 0, -1]），
    鼓励机器人保持直立姿态。
    
    Args:
        env: 环境实例
        desired_gravity: 期望的重力方向向量（通常为 [0, 0, -1]）
        asset_cfg: 机器人资产配置
    
    Returns:
        姿态对齐程度，范围 [0, 1]，1 表示完全对齐
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # 余弦距离
    normalized = 0.5 * cos_dist + 0.5  # 从 [-1, 1] 映射到 [0, 1]
    return torch.square(normalized)  # 使用平方核增强奖励


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    向上姿态惩罚：惩罚机器人姿态偏离竖直方向
    
    通过检查重力在机器人 z 轴（向上）方向的分量，惩罚机器人倾倒。
    
    Args:
        env: 环境实例
        asset_cfg: 机器人资产配置
    
    Returns:
        姿态偏差的平方，值越大表示倾斜越严重
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    # projected_gravity_b[:, 2] 应该接近 -1（重力向下）
    # 1 - projected_gravity_b[:, 2] 应该接近 2（完全直立）
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """
    关节位置惩罚：惩罚关节位置偏离默认值
    
    在运动时适度惩罚，在静止时强力惩罚，鼓励机器人在静止时回到默认姿态。
    
    Args:
        env: 环境实例
        asset_cfg: 机器人资产配置
        stand_still_scale: 静止时的惩罚缩放系数（通常 > 1.0）
        velocity_threshold: 判断为运动的速度阈值
    
    Returns:
        关节位置偏差的范数
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)  # 命令速度
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)  # 实际速度
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)  # 关节偏差
    # 如果有命令或在运动，使用标准惩罚；否则使用放大的惩罚
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


# ============================================================================
#                           足部奖励函数
# ============================================================================


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    足部绊倒惩罚：惩罚足部撞击垂直表面
    
    通过比较接触力的水平分量（xy）和垂直分量（z），检测足部是否撞到了垂直障碍物。
    正常行走时，垂直力（z）应该远大于水平力（xy）。
    
    Args:
        env: 环境实例
        sensor_cfg: 接触传感器配置
    
    Returns:
        是否发生绊倒的布尔值（1 表示绊倒，0 表示正常）
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])  # 垂直力
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)  # 水平力
    # 如果水平力 > 4 * 垂直力，说明撞到了垂直表面
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward


"""
Stair climbing rewards.
楼梯攀爬专用奖励函数
"""


def upward_progress(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    向上进展奖励：鼓励机器人向上攀爬楼梯

    计算每步的高度增量，对正向增量给予奖励。
    使用累计高度差避免"跳起来再落回去"骗奖励的情况。

    Args:
        env: 环境实例
        asset_cfg: 机器人资产配置

    Returns:
        奖励张量，形状为 (num_envs,)
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # 获取当前基座高度
    current_height = asset.data.root_pos_w[:, 2]

    # 初始化或获取上一步高度
    if not hasattr(env, "_prev_base_height"):
        env._prev_base_height = current_height.clone()
        env._initial_height = current_height.clone()

    # 形状不一致时进行重置（例如环境数量变化时）
    if env._prev_base_height.shape != current_height.shape:
        env._prev_base_height = current_height.clone()
    if env._initial_height.shape != current_height.shape:
        env._initial_height = current_height.clone()

    # 按回合重置缓存，避免跨回合累计
    if hasattr(env, "reset_buf"):
        reset_mask = env.reset_buf > 0
        if torch.any(reset_mask):
            env._prev_base_height = torch.where(reset_mask, current_height, env._prev_base_height)
            env._initial_height = torch.where(reset_mask, current_height, env._initial_height)

    # 计算高度增量
    height_delta = current_height - env._prev_base_height

    # 更新上一步高度
    env._prev_base_height = current_height.clone()

    # 对正向高度增量给予奖励，负向增量给予较小惩罚
    # 使用 clamp 避免过大的奖励/惩罚
    reward = torch.clamp(height_delta, -0.1, 0.2)

    # 额外奖励：相对于初始高度的总进展
    total_progress = current_height - env._initial_height
    progress_bonus = torch.clamp(total_progress * 0.1, 0, 0.5)

    return reward + progress_bonus


def base_height_adaptive(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    自适应基座高度惩罚：相对于脚下地形的高度

    使用 RayCaster 获取脚下地形高度，计算相对高度偏差。
    这样在楼梯上爬升时不会被错误惩罚。

    Args:
        env: 环境实例
        target_height: 目标相对高度（机器人基座相对于地形的高度）
        sensor_cfg: 高度扫描器配置
        asset_cfg: 机器人资产配置

    Returns:
        惩罚张量，形状为 (num_envs,)
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    # 获取机器人基座世界高度
    base_height_world = asset.data.root_pos_w[:, 2]

    # 从高度扫描器获取脚下地形高度
    # RayCaster 的 ray_hits_w 包含射线击中点的世界坐标
    # 取中心区域的射线作为脚下地形高度估计
    ray_hits = sensor.data.ray_hits_w

    # 获取射线模式的尺寸信息
    num_rays = ray_hits.shape[1]

    # 取中心区域的射线（假设网格中心对应机器人正下方）
    center_idx = num_rays // 2
    # 取中心附近几条射线的平均值
    start_idx = max(0, center_idx - 5)
    end_idx = min(num_rays, center_idx + 5)

    terrain_height = ray_hits[:, start_idx:end_idx, 2].mean(dim=1)

    # 计算相对高度
    relative_height = base_height_world - terrain_height

    # 计算与目标高度的偏差
    height_error = torch.square(relative_height - target_height)

    return height_error


def stair_forward_progress(
    env: ManagerBasedRLEnv,
    stair_direction: list[float] = [1.0, 0.0],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    沿楼梯方向的前进奖励

    Args:
        env: 环境实例
        stair_direction: 楼梯方向单位向量 [x, y]
        asset_cfg: 机器人资产配置

    Returns:
        奖励张量，形状为 (num_envs,)
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # 获取机器人在世界坐标系下的线速度
    lin_vel = asset.data.root_lin_vel_w[:, :2]

    # 楼梯方向向量
    stair_dir = torch.tensor(stair_direction, device=env.device)
    stair_dir = stair_dir / torch.norm(stair_dir)

    # 计算沿楼梯方向的速度分量
    forward_vel = torch.sum(lin_vel * stair_dir, dim=1)

    # 对正向速度给予奖励
    reward = torch.clamp(forward_vel, 0, 1.0)

    return reward
