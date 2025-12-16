"""
===============================================================================
自适应训练课程学习模块
===============================================================================

本模块实现自动检测训练阶段并动态调整参数的机制。

核心功能:
    - 自动检测训练阶段（基于 episode_length 和 terrain_level）
    - 动态调整奖励权重
    - 动态调整终止条件参数
    - 根据阶段切换课程学习策略

训练阶段定义:
    Stage 0: 初始探索期 (mean_episode_length < 100)
    Stage 1: 站立稳定期 (100 ≤ mean_episode_length < 300)
    Stage 2: 行走学习期 (300 ≤ mean_episode_length < 600)
    Stage 3: 楼梯适应期 (mean_episode_length ≥ 600, terrain_level < 3)
    Stage 4: 楼梯精通期 (mean_episode_length ≥ 600, terrain_level ≥ 3)

使用方法:
    在 StairCurriculumCfg 中配置:
    
        terrain_levels = CurrTerm(func=mdp.adaptive_terrain_levels)

===============================================================================
"""

from __future__ import annotations

import torch
from collections import deque
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ============================================================================
#                         全局状态存储
# ============================================================================

class AdaptiveState:
    """自适应训练状态管理器（单例模式）"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.reset()
    
    def reset(self):
        """重置状态"""
        self.episode_lengths = deque(maxlen=100)
        self.terrain_levels = deque(maxlen=100)
        self.current_stage = 0
        self.stage_stable_count = 0
        self.last_update_step = 0
        self.stage_enter_step = 0  # 进入当前阶段的步数
        self.min_stage_duration = 2000  # 每个阶段最少停留步数（从5000降低）


def _get_adaptive_state() -> AdaptiveState:
    """获取全局自适应状态实例"""
    return AdaptiveState()


# ============================================================================
#                         阶段参数配置
# ============================================================================

STAGE_CONFIGS = {
    0: {  # 初始探索期（平地学习）
        "name": "初始探索期",
        # 奖励权重 - 降低 alive 避免静止拿奖励，提高速度跟踪
        "alive": 2.0,
        "track_lin_vel_xy": 3.0,  # 大幅提高速度跟踪
        "upward_progress": 0.0,   # 平地禁用
        "flat_orientation_l2": -1.0,
        "action_rate": -0.01,     # 降低动作惩罚鼓励探索
        # 终止条件
        "limit_angle": 1.4,  # 80°
        # 课程学习参数
        "curriculum_type": "survival",
        "survival_ratio_upgrade": 0.5,
        "survival_ratio_downgrade": 0.1,
        # PPO 参数
        "learning_rate": 1e-3,
        "entropy_coef": 0.02,      # 高探索
        "clip_param": 0.2,
        "desired_kl": 0.015,
    },
    1: {  # 站立稳定期（平地行走）
        "name": "站立稳定期",
        # 奖励权重
        "alive": 2.5,
        "track_lin_vel_xy": 2.5,
        "upward_progress": 0.5,   # 开始引入少量上升奖励
        "flat_orientation_l2": -1.5,
        "action_rate": -0.02,
        # 终止条件
        "limit_angle": 1.3,  # 74°
        # 课程学习参数
        "curriculum_type": "survival",
        "survival_ratio_upgrade": 0.6,
        "survival_ratio_downgrade": 0.15,
        # PPO 参数
        "learning_rate": 8e-4,
        "entropy_coef": 0.015,
        "clip_param": 0.2,
        "desired_kl": 0.012,
    },
    2: {  # 行走学习期（过渡到楼梯）
        "name": "行走学习期",
        # 奖励权重
        "alive": 3.0,
        "track_lin_vel_xy": 2.0,
        "upward_progress": 1.5,
        "flat_orientation_l2": -2.0,
        "action_rate": -0.03,
        # 终止条件
        "limit_angle": 1.2,  # 69°
        # 课程学习参数
        "curriculum_type": "climb",
        "height_weight": 1.5,
        "forward_weight": 1.0,
        # PPO 参数
        "learning_rate": 5e-4,
        "entropy_coef": 0.01,
        "clip_param": 0.18,
        "desired_kl": 0.01,
    },
    3: {  # 楼梯适应期
        "name": "楼梯适应期",
        # 奖励权重
        "alive": 2.5,
        "track_lin_vel_xy": 1.5,
        "upward_progress": 3.0,
        "flat_orientation_l2": -1.5,
        "action_rate": -0.04,
        # 终止条件
        "limit_angle": 1.1,  # 63°
        # 课程学习参数
        "curriculum_type": "climb",
        "height_weight": 2.0,
        "forward_weight": 1.0,
        # PPO 参数
        "learning_rate": 3e-4,
        "entropy_coef": 0.008,
        "clip_param": 0.15,
        "desired_kl": 0.008,
    },
    4: {  # 楼梯精通期
        "name": "楼梯精通期",
        # 奖励权重
        "alive": 2.0,
        "track_lin_vel_xy": 1.0,
        "upward_progress": 4.0,
        "flat_orientation_l2": -1.5,
        "action_rate": -0.04,
        # 终止条件
        "limit_angle": 1.0,  # 57°
        # 课程学习参数
        "curriculum_type": "climb",
        "height_weight": 2.5,
        "forward_weight": 0.5,
        # PPO 参数
        "learning_rate": 1e-4,
        "entropy_coef": 0.005,     # 低探索，精细调整
        "clip_param": 0.12,
        "desired_kl": 0.005,
    },
}


# ============================================================================
#                         阶段检测函数
# ============================================================================

def _detect_stage(mean_length: float, mean_terrain: float) -> int:
    """
    根据训练指标判断当前应处于哪个阶段
    
    Args:
        mean_length: 滑动平均 episode 长度
        mean_terrain: 滑动平均地形等级
    
    Returns:
        阶段编号 (0-4)
    """
    if mean_length < 100:
        return 0  # 初始探索期
    elif mean_length < 300:
        return 1  # 站立稳定期
    elif mean_length < 600:
        return 2  # 行走学习期
    elif mean_terrain < 3:
        return 3  # 楼梯适应期
    else:
        return 4  # 楼梯精通期


# ============================================================================
#                         参数更新函数
# ============================================================================

def _update_reward_weights(env: ManagerBasedRLEnv, stage: int) -> dict:
    """
    更新奖励权重
    
    Args:
        env: 环境实例
        stage: 目标阶段
    
    Returns:
        实际更新的权重字典
    """
    params = STAGE_CONFIGS[stage]
    updated = {}
    
    # 需要更新的奖励项
    reward_names = ["alive", "track_lin_vel_xy", "upward_progress", "flat_orientation_l2", "action_rate"]
    
    for name in reward_names:
        if name in params:
            try:
                cfg = env.reward_manager.get_term_cfg(name)
                old_weight = cfg.weight
                cfg.weight = params[name]
                updated[name] = {"old": old_weight, "new": params[name]}
            except ValueError:
                # 奖励项不存在，跳过
                pass
    
    return updated


def _update_termination_params(env: ManagerBasedRLEnv, stage: int) -> dict:
    """
    更新终止条件参数
    
    Args:
        env: 环境实例
        stage: 目标阶段
    
    Returns:
        实际更新的参数字典
    """
    params = STAGE_CONFIGS[stage]
    updated = {}
    
    if "limit_angle" in params:
        try:
            cfg = env.termination_manager.get_term_cfg("bad_orientation")
            old_value = cfg.params.get("limit_angle", None)
            cfg.params["limit_angle"] = params["limit_angle"]
            updated["limit_angle"] = {"old": old_value, "new": params["limit_angle"]}
        except ValueError:
            pass
    
    return updated


def _log_stage_transition(
    old_stage: int,
    new_stage: int,
    mean_length: float,
    mean_terrain: float,
    reward_updates: dict,
    termination_updates: dict,
):
    """打印阶段切换日志"""
    old_name = STAGE_CONFIGS[old_stage]["name"]
    new_name = STAGE_CONFIGS[new_stage]["name"]
    
    print("\n" + "=" * 70)
    print("[Adaptive Training] 阶段切换!")
    print("-" * 70)
    print(f"  上一阶段: Stage {old_stage} ({old_name})")
    print(f"  新 阶 段: Stage {new_stage} ({new_name})")
    print("-" * 70)
    print("  触发指标:")
    print(f"    - Mean Episode Length: {mean_length:.1f} steps")
    print(f"    - Mean Terrain Level:  {mean_terrain:.2f}")
    print("-" * 70)
    
    if reward_updates:
        print("  奖励权重变化:")
        for name, vals in reward_updates.items():
            print(f"    - {name}: {vals['old']:.2f} -> {vals['new']:.2f}")
    
    if termination_updates:
        print("  终止条件变化:")
        for name, vals in termination_updates.items():
            old_val = vals['old'] if vals['old'] is not None else 0.0
            old_deg = old_val * 57.3
            new_deg = vals['new'] * 57.3
            print(f"    - {name}: {old_val:.2f} rad ({old_deg:.0f}°) -> {vals['new']:.2f} rad ({new_deg:.0f}°)")
    
    print("=" * 70 + "\n")


# ============================================================================
#                         底层课程学习函数调用
# ============================================================================

def _call_survival_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg,
    params: dict,
) -> torch.Tensor:
    """调用基于存活时间的课程学习"""
    terrain: TerrainImporter = env.scene.terrain
    
    # 确保 env_ids 是 tensor
    if not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)
    
    survival_ratio = env.episode_length_buf[env_ids].float() / env.max_episode_length
    
    move_up = survival_ratio > params.get("survival_ratio_upgrade", 0.7)
    move_down = survival_ratio < params.get("survival_ratio_downgrade", 0.2)
    move_down = move_down & ~move_up  # 确保互斥
    
    terrain.update_env_origins(env_ids, move_up, move_down)
    
    return torch.mean(terrain.terrain_levels.float())


def _call_climb_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg,
    params: dict,
) -> torch.Tensor:
    """调用基于攀爬进度的课程学习"""
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    
    # 确保 env_ids 是 tensor
    if not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)
    
    displacement = asset.data.root_pos_w[env_ids, :3] - env.scene.env_origins[env_ids, :3]
    
    forward_distance = displacement[:, 0]
    height_gain = displacement[:, 2]
    
    height_weight = params.get("height_weight", 2.0)
    forward_weight = params.get("forward_weight", 1.0)
    
    progress = forward_distance * forward_weight + height_gain * height_weight
    
    upgrade_threshold = terrain.cfg.terrain_generator.size[0] * 0.3
    move_up = progress > upgrade_threshold
    move_down = progress < 0.5
    move_down = move_down & ~move_up
    
    terrain.update_env_origins(env_ids, move_up, move_down)
    
    return torch.mean(terrain.terrain_levels.float())


# ============================================================================
#                         主入口函数
# ============================================================================

def adaptive_terrain_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    自适应地形难度课程学习（主入口函数）
    
    功能:
        1. 自动检测当前训练阶段
        2. 阶段切换时自动更新奖励权重和终止条件
        3. 根据阶段调用不同的底层课程学习策略
    
    Args:
        env: 强化学习环境实例
        env_ids: 本次重置的环境 ID 列表
        asset_cfg: 机器人资产配置
    
    Returns:
        所有环境的平均地形等级（标量张量）
    
    使用方法:
        在 StairCurriculumCfg 中配置:
        
            terrain_levels = CurrTerm(func=mdp.adaptive_terrain_levels)
    """
    state = _get_adaptive_state()
    
    # ==================== 1. 收集指标 ====================
    # 获取当前 episode 长度（使用重置环境的长度，因为这些是刚结束的 episode）
    if len(env_ids) > 0:
        finished_lengths = env.episode_length_buf[env_ids].float()
        mean_finished_length = finished_lengths.mean().item()
        state.episode_lengths.append(mean_finished_length)
    
    # 获取地形等级
    terrain: TerrainImporter = env.scene.terrain
    current_terrain_level = terrain.terrain_levels.float().mean().item()
    state.terrain_levels.append(current_terrain_level)
    
    # ==================== 2. 阶段检测（每 500 步检查一次）====================
    step_counter = env.common_step_counter
    check_interval = 500  # 从 1000 降低到 500，加快响应
    
    if step_counter - state.last_update_step >= check_interval:
        state.last_update_step = step_counter
        
        # 计算滑动平均
        if len(state.episode_lengths) > 0:
            avg_length = sum(state.episode_lengths) / len(state.episode_lengths)
        else:
            avg_length = 0.0
        
        if len(state.terrain_levels) > 0:
            avg_terrain = sum(state.terrain_levels) / len(state.terrain_levels)
        else:
            avg_terrain = 0.0
        
        # 检测新阶段
        new_stage = _detect_stage(avg_length, avg_terrain)
        
        # 检查是否满足阶段切换条件
        min_duration_met = (step_counter - state.stage_enter_step) >= state.min_stage_duration
        
        if new_stage != state.current_stage and min_duration_met:
            state.stage_stable_count += 1
            
            # 连续 3 次检测到相同的新阶段才切换（从 5 降低到 3）
            if state.stage_stable_count >= 3:
                old_stage = state.current_stage
                
                # 执行参数更新
                reward_updates = _update_reward_weights(env, new_stage)
                termination_updates = _update_termination_params(env, new_stage)
                
                # 打印日志
                _log_stage_transition(
                    old_stage, new_stage, avg_length, avg_terrain,
                    reward_updates, termination_updates
                )
                
                # 更新状态
                state.current_stage = new_stage
                state.stage_stable_count = 0
                state.stage_enter_step = step_counter
        else:
            state.stage_stable_count = 0
    
    # ==================== 3. 调用底层课程学习函数 ====================
    # 如果没有环境需要重置，直接返回当前地形等级
    if len(env_ids) == 0:
        return torch.mean(terrain.terrain_levels.float())
    
    stage = state.current_stage
    params = STAGE_CONFIGS[stage]
    
    curriculum_type = params.get("curriculum_type", "survival")
    
    if curriculum_type == "survival":
        return _call_survival_curriculum(env, env_ids, asset_cfg, params)
    else:  # climb
        return _call_climb_curriculum(env, env_ids, asset_cfg, params)


# ============================================================================
#                         辅助函数（用于外部查询）
# ============================================================================

def get_current_stage() -> int:
    """获取当前训练阶段"""
    return _get_adaptive_state().current_stage


def get_stage_name(stage: int = None) -> str:
    """获取阶段名称"""
    if stage is None:
        stage = get_current_stage()
    return STAGE_CONFIGS.get(stage, {}).get("name", "Unknown")


def reset_adaptive_state():
    """重置自适应状态（用于新训练开始时）"""
    _get_adaptive_state().reset()


def get_ppo_params(stage: int = None) -> dict:
    """
    获取指定阶段的 PPO 参数
    
    Args:
        stage: 阶段编号，如果为 None 则使用当前阶段
    
    Returns:
        包含 PPO 参数的字典
    """
    if stage is None:
        stage = get_current_stage()
    
    config = STAGE_CONFIGS.get(stage, STAGE_CONFIGS[0])
    return {
        "learning_rate": config.get("learning_rate", 1e-3),
        "entropy_coef": config.get("entropy_coef", 0.01),
        "clip_param": config.get("clip_param", 0.2),
        "desired_kl": config.get("desired_kl", 0.01),
    }


def update_ppo_params(alg, stage: int = None) -> dict:
    """
    更新 PPO 算法参数
    
    Args:
        alg: RSL-RL 的 PPO 算法实例 (runner.alg)
        stage: 目标阶段，如果为 None 则使用当前阶段
    
    Returns:
        实际更新的参数字典
    """
    if stage is None:
        stage = get_current_stage()
    
    params = get_ppo_params(stage)
    updates = {}
    
    # 更新学习率
    if hasattr(alg, 'learning_rate'):
        old_lr = alg.learning_rate
        new_lr = params["learning_rate"]
        if abs(old_lr - new_lr) > 1e-8:
            alg.learning_rate = new_lr
            # 同时更新优化器的学习率
            for param_group in alg.optimizer.param_groups:
                param_group['lr'] = new_lr
            updates["learning_rate"] = {"old": old_lr, "new": new_lr}
    
    # 更新熵系数
    if hasattr(alg, 'entropy_coef'):
        old_ent = alg.entropy_coef
        new_ent = params["entropy_coef"]
        if abs(old_ent - new_ent) > 1e-8:
            alg.entropy_coef = new_ent
            updates["entropy_coef"] = {"old": old_ent, "new": new_ent}
    
    # 更新裁剪参数
    if hasattr(alg, 'clip_param'):
        old_clip = alg.clip_param
        new_clip = params["clip_param"]
        if abs(old_clip - new_clip) > 1e-8:
            alg.clip_param = new_clip
            updates["clip_param"] = {"old": old_clip, "new": new_clip}
    
    # 更新目标 KL
    if hasattr(alg, 'desired_kl'):
        old_kl = alg.desired_kl
        new_kl = params["desired_kl"]
        if abs(old_kl - new_kl) > 1e-8:
            alg.desired_kl = new_kl
            updates["desired_kl"] = {"old": old_kl, "new": new_kl}
    
    return updates
