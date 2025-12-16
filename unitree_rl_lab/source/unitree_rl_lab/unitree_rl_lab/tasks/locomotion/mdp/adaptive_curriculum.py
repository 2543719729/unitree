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
        # 课程学习参数 - 使用 survival 以便在平地上也能升级到楼梯
        "curriculum_type": "survival",
        "survival_ratio_upgrade": 0.7,    # 存活 70% 时间升级
        "survival_ratio_downgrade": 0.2,  # 存活不足 20% 时间降级
        # PPO 参数
        "learning_rate": 3e-4,
        "entropy_coef": 0.008,
        "clip_param": 0.15,
        "desired_kl": 0.008,
    },
    4: {  # 楼梯精通期
        "name": "楼梯精通期",
        # 奖励权重
        "alive": 1.0,
        "track_lin_vel_xy": 2.5,
        "upward_progress": 4.0,
        "flat_orientation_l2": -1.5,
        "action_rate": -0.02,
        # 终止条件
        "limit_angle": 1.0,  # 57°
        # 课程学习参数
        "curriculum_type": "climb",
        "height_weight": 2.5,
        "forward_weight": 0.5,
        # PPO 参数
        "learning_rate": 2e-4,
        "entropy_coef": 0.008,     # 保持一定探索，避免策略塌缩到站立不动
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
#                         阶段检测与参数同步更新
# ============================================================================

def _check_and_update_stage(
    env: ManagerBasedRLEnv,
    state: AdaptiveState,
    force_check: bool = False,
) -> bool:
    """
    检查是否需要切换阶段，并同步更新自适应参数
    
    [重要修复] 此函数确保自适应参数与 terrain_levels 同步：
        - 定期检查（每 500 步）
        - terrain_levels 更新后强制检查
    
    Args:
        env: 环境实例
        state: 自适应状态
        force_check: 是否强制检查（terrain_levels 更新后调用时设为 True）
    
    Returns:
        是否发生了阶段切换
    """
    step_counter = env.common_step_counter
    check_interval = 500
    
    # 非强制检查时，遵守检查间隔
    if not force_check and (step_counter - state.last_update_step < check_interval):
        return False
    
    if not force_check:
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
        
        # 连续 3 次检测到相同的新阶段才切换
        # 强制检查时（terrain_level 刚更新），只需 1 次确认即可立即切换
        required_stable_count = 1 if force_check else 3
        
        if state.stage_stable_count >= required_stable_count:
            old_stage = state.current_stage
            
            # 执行参数更新
            reward_updates = _update_reward_weights(env, new_stage)
            termination_updates = _update_termination_params(env, new_stage)
            
            # 打印日志
            sync_note = " [与terrain_level同步]" if force_check else ""
            print(f"\n{'=' * 70}")
            print(f"[Adaptive Training] 阶段切换!{sync_note}")
            print("-" * 70)
            print(f"  上一阶段: Stage {old_stage} ({STAGE_CONFIGS[old_stage]['name']})")
            print(f"  新 阶 段: Stage {new_stage} ({STAGE_CONFIGS[new_stage]['name']})")
            print("-" * 70)
            print("  触发指标:")
            print(f"    - Mean Episode Length: {avg_length:.1f} steps")
            print(f"    - Mean Terrain Level:  {avg_terrain:.2f}")
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
            
            # 更新状态
            state.current_stage = new_stage
            state.stage_stable_count = 0
            state.stage_enter_step = step_counter
            return True
    else:
        state.stage_stable_count = 0
    
    return False


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
        4. [重要] terrain_levels 更新后同步检查阶段切换
    
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
    terrain: TerrainImporter = env.scene.terrain
    
    # ==================== 1. 收集指标 ====================
    # 获取当前 episode 长度（使用重置环境的长度，因为这些是刚结束的 episode）
    if len(env_ids) > 0:
        finished_lengths = env.episode_length_buf[env_ids].float()
        mean_finished_length = finished_lengths.mean().item()
        state.episode_lengths.append(mean_finished_length)
    
    # 获取地形等级（更新前的值）
    old_terrain_level = terrain.terrain_levels.float().mean().item()
    state.terrain_levels.append(old_terrain_level)
    
    # ==================== 2. 定期阶段检测 ====================
    _check_and_update_stage(env, state, force_check=False)
    
    # ==================== 3. 调用底层课程学习函数 ====================
    # 如果没有环境需要重置，直接返回当前地形等级
    if len(env_ids) == 0:
        return torch.mean(terrain.terrain_levels.float())
    
    stage = state.current_stage
    params = STAGE_CONFIGS[stage]
    curriculum_type = params.get("curriculum_type", "survival")
    
    if curriculum_type == "survival":
        result = _call_survival_curriculum(env, env_ids, asset_cfg, params)
    else:  # climb
        result = _call_climb_curriculum(env, env_ids, asset_cfg, params)
    
    # ==================== 4. [重要修复] terrain_levels 更新后同步检查 ====================
    # 获取更新后的地形等级
    new_terrain_level = terrain.terrain_levels.float().mean().item()
    
    # 如果 terrain_level 发生了显著变化，立即更新滑动窗口并检查阶段
    terrain_change = abs(new_terrain_level - old_terrain_level)
    if terrain_change > 0.1:  # 变化超过 0.1 视为显著变化
        # 更新滑动窗口中的最新值
        if len(state.terrain_levels) > 0:
            state.terrain_levels[-1] = new_terrain_level
        else:
            state.terrain_levels.append(new_terrain_level)
        
        # 强制检查阶段切换（确保自适应参数与 terrain_level 同步）
        _check_and_update_stage(env, state, force_check=True)
    
    return result


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


def get_adaptive_state_dict() -> dict:
    """
    获取自适应状态字典（用于保存到 checkpoint）
    
    Returns:
        包含自适应状态的字典
    """
    state = _get_adaptive_state()
    return {
        "current_stage": state.current_stage,
        "episode_lengths": list(state.episode_lengths),
        "terrain_levels": list(state.terrain_levels),
        "stage_stable_count": state.stage_stable_count,
        "last_update_step": state.last_update_step,
        "stage_enter_step": state.stage_enter_step,
    }


def load_adaptive_state_dict(state_dict: dict):
    """
    从字典加载自适应状态（用于从 checkpoint 恢复）
    
    Args:
        state_dict: 包含自适应状态的字典
    """
    state = _get_adaptive_state()
    state.reset()
    
    state.current_stage = state_dict.get("current_stage", 0)
    state.stage_stable_count = state_dict.get("stage_stable_count", 0)
    state.last_update_step = state_dict.get("last_update_step", 0)
    state.stage_enter_step = state_dict.get("stage_enter_step", 0)
    
    # 恢复 deque
    for length in state_dict.get("episode_lengths", []):
        state.episode_lengths.append(length)
    for level in state_dict.get("terrain_levels", []):
        state.terrain_levels.append(level)
    
    # 计算平均值用于打印
    avg_length = sum(state.episode_lengths) / len(state.episode_lengths) if state.episode_lengths else 0
    avg_terrain = sum(state.terrain_levels) / len(state.terrain_levels) if state.terrain_levels else 0
    
    print(f"[Adaptive State] 从 checkpoint 恢复阶段: Stage {state.current_stage} ({get_stage_name(state.current_stage)})")
    print(f"    - mean_episode_length: {avg_length:.1f}")
    print(f"    - mean_terrain_level: {avg_terrain:.2f}")


def init_adaptive_state_from_metrics(mean_episode_length: float, mean_terrain_level: float = 0.0):
    """
    根据当前训练指标初始化自适应状态（用于恢复训练时）
    
    Args:
        mean_episode_length: 当前平均 episode 长度
        mean_terrain_level: 当前平均地形等级
    """
    state = _get_adaptive_state()
    state.reset()
    
    # 根据指标检测应该处于的阶段
    stage = _detect_stage(mean_episode_length, mean_terrain_level)
    state.current_stage = stage
    
    # 预填充 deque 以避免需要重新积累数据
    for _ in range(50):
        state.episode_lengths.append(mean_episode_length)
        state.terrain_levels.append(mean_terrain_level)
    
    print(f"[Adaptive State] 根据指标初始化阶段: Stage {stage} ({get_stage_name(stage)})")
    print(f"    - mean_episode_length: {mean_episode_length:.1f}")
    print(f"    - mean_terrain_level: {mean_terrain_level:.2f}")


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


def apply_stage_params(env, alg, stage: int = None) -> dict:
    """
    应用指定阶段的所有参数（奖励权重、终止条件、PPO 参数）
    
    用于训练开始时初始化，确保所有参数与当前阶段匹配。
    
    Args:
        env: 环境实例（用于更新奖励权重和终止条件）
        alg: PPO 算法实例（用于更新 PPO 参数）
        stage: 目标阶段，如果为 None 则使用当前阶段
    
    Returns:
        包含所有更新信息的字典
    """
    if stage is None:
        stage = get_current_stage()
    
    updates = {}
    
    # 更新奖励权重
    reward_updates = _update_reward_weights(env, stage)
    if reward_updates:
        updates["rewards"] = reward_updates
    
    # 更新终止条件
    termination_updates = _update_termination_params(env, stage)
    if termination_updates:
        updates["terminations"] = termination_updates
    
    # 更新 PPO 参数
    ppo_updates = update_ppo_params(alg, stage)
    if ppo_updates:
        updates["ppo"] = ppo_updates
    
    # 打印应用的参数
    stage_name = get_stage_name(stage)
    print(f"\n[Adaptive] 应用 Stage {stage} ({stage_name}) 的参数:")
    
    if reward_updates:
        print("  奖励权重:")
        for name, vals in reward_updates.items():
            print(f"    - {name}: {vals['old']:.2f} -> {vals['new']:.2f}")
    
    if termination_updates:
        print("  终止条件:")
        for name, vals in termination_updates.items():
            old_val = vals['old'] if vals['old'] is not None else 0.0
            print(f"    - {name}: {old_val:.2f} -> {vals['new']:.2f}")
    
    if ppo_updates:
        print("  PPO 参数:")
        for name, vals in ppo_updates.items():
            if name == "learning_rate":
                print(f"    - {name}: {vals['old']:.2e} -> {vals['new']:.2e}")
            else:
                print(f"    - {name}: {vals['old']:.4f} -> {vals['new']:.4f}")
    
    print()
    return updates
