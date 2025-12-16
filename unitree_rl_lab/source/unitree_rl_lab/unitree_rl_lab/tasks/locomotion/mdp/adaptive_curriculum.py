"""
===============================================================================
自适应训练课程学习模块 (v2.0 - 优化版)
===============================================================================

本模块实现自动检测训练阶段并动态调整参数的机制。

核心功能:
    - 自动检测训练阶段（基于 episode_length 相对比例和 terrain_level）
    - 动态调整奖励权重（平滑过渡，避免跳变）
    - 动态调整终止条件参数
    - 根据阶段切换课程学习策略

训练阶段定义（使用相对比例，自适应 max_episode_length）:
    Stage 0: 初始探索期 (survival_ratio < 0.10)
    Stage 1: 站立稳定期 (0.10 ≤ survival_ratio < 0.30)
    Stage 2: 行走学习期 (0.30 ≤ survival_ratio < 0.60)
    Stage 3: 楼梯适应期 (survival_ratio ≥ 0.60, terrain_level < 3)
    Stage 4: 楼梯精通期 (survival_ratio ≥ 0.60, terrain_level ≥ 3)

v2.0 优化内容:
    - [问题1修复] 阶段检测改用相对比例，自适应 max_episode_length
    - [问题2修复] 奖励权重平滑过渡，避免 Value Function 失效
    - [问题3修复] 移除 PPO 动态参数，依赖 adaptive schedule
    - [问题6修复] min_stage_duration 改用 iteration 计数

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
        
        # [问题6修复] 使用 iteration 计数而非 step 计数
        # 每个 iteration = num_envs * num_steps_per_env 个 samples
        # 50 个 iteration 约等于 4096 * 32 * 50 = 6.5M samples
        self.min_stage_iterations = 50  # 每个阶段最少停留 50 个 iteration
        self.stage_enter_iteration = 0  # 进入当前阶段的 iteration 数
        self.estimated_iteration = 0    # 估计的当前 iteration 数
        
        # [问题2修复] 平滑过渡相关状态
        self.transition_progress = 1.0   # 过渡进度 [0, 1]，1.0 表示过渡完成
        self.transition_start_step = 0   # 过渡开始的 step
        self.transition_duration = 50000 # 过渡持续 50000 步（约 50 iteration）
        self.source_stage = 0            # 过渡起始阶段
        self.target_stage = 0            # 过渡目标阶段
        
        # 环境参数缓存（用于相对比例计算）
        self.max_episode_length = 1000   # 默认值，会在运行时更新
        self.num_envs = 4096             # 默认值
        self.num_steps_per_env = 32      # 默认值


def _get_adaptive_state() -> AdaptiveState:
    """获取全局自适应状态实例"""
    return AdaptiveState()


# ============================================================================
#                         阶段参数配置
# ============================================================================

# [问题1修复] 阶段检测阈值（使用相对比例）
# 这些是 survival_ratio = mean_episode_length / max_episode_length 的阈值
STAGE_THRESHOLDS = {
    "stage_0_max": 0.10,   # Stage 0: survival_ratio < 10%
    "stage_1_max": 0.30,   # Stage 1: 10% <= survival_ratio < 30%
    "stage_2_max": 0.60,   # Stage 2: 30% <= survival_ratio < 60%
    "terrain_threshold": 3, # Stage 3/4 分界: terrain_level
}

# [问题3修复] 移除 PPO 参数，依赖 adaptive schedule 自动调整 learning_rate
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
    },
}


# ============================================================================
#                         阶段检测函数
# ============================================================================

def _detect_stage(mean_length: float, mean_terrain: float, max_episode_length: int) -> int:
    """
    [问题1修复] 根据训练指标判断当前应处于哪个阶段
    
    使用相对比例（survival_ratio）而非固定阈值，自适应不同的 max_episode_length
    
    Args:
        mean_length: 滑动平均 episode 长度
        mean_terrain: 滑动平均地形等级
        max_episode_length: 最大 episode 长度（用于计算相对比例）
    
    Returns:
        阶段编号 (0-4)
    """
    # 计算存活比例
    survival_ratio = mean_length / max(max_episode_length, 1)
    
    if survival_ratio < STAGE_THRESHOLDS["stage_0_max"]:
        return 0  # 初始探索期: 存活 < 10%
    elif survival_ratio < STAGE_THRESHOLDS["stage_1_max"]:
        return 1  # 站立稳定期: 10% <= 存活 < 30%
    elif survival_ratio < STAGE_THRESHOLDS["stage_2_max"]:
        return 2  # 行走学习期: 30% <= 存活 < 60%
    elif mean_terrain < STAGE_THRESHOLDS["terrain_threshold"]:
        return 3  # 楼梯适应期: 存活 >= 60%, 地形 < 3
    else:
        return 4  # 楼梯精通期: 存活 >= 60%, 地形 >= 3


# ============================================================================
#                         参数更新函数
# ============================================================================

# [问题2修复] 平滑过渡相关函数
def _get_blended_weight(param_name: str, state: AdaptiveState) -> float:
    """
    获取混合后的权重（在过渡期间平滑插值）
    
    Args:
        param_name: 参数名称
        state: 自适应状态
    
    Returns:
        混合后的权重值
    """
    if state.transition_progress >= 1.0:
        # 过渡完成，直接返回目标阶段参数
        return STAGE_CONFIGS[state.target_stage].get(param_name, 0.0)
    
    source_val = STAGE_CONFIGS[state.source_stage].get(param_name, 0.0)
    target_val = STAGE_CONFIGS[state.target_stage].get(param_name, 0.0)
    
    # 使用 ease-in-out 插值（更平滑的过渡）
    t = state.transition_progress
    smooth_t = t * t * (3 - 2 * t)  # smoothstep
    
    return source_val + smooth_t * (target_val - source_val)


def _update_transition_progress(state: AdaptiveState, step_counter: int):
    """
    更新过渡进度
    
    Args:
        state: 自适应状态
        step_counter: 当前步数
    """
    if state.transition_progress < 1.0:
        elapsed = step_counter - state.transition_start_step
        state.transition_progress = min(1.0, elapsed / state.transition_duration)


def _update_reward_weights_smooth(env, state: AdaptiveState) -> dict:
    """
    [问题2修复] 平滑更新奖励权重
    
    在过渡期间使用混合权重，避免 Value Function 失效
    
    Args:
        env: 环境实例
        state: 自适应状态
    
    Returns:
        实际更新的权重字典
    """
    updated = {}
    reward_names = ["alive", "track_lin_vel_xy", "upward_progress", "flat_orientation_l2", "action_rate"]
    
    for name in reward_names:
        blended_weight = _get_blended_weight(name, state)
        try:
            cfg = env.reward_manager.get_term_cfg(name)
            old_weight = cfg.weight
            if abs(old_weight - blended_weight) > 1e-6:  # 只有变化时才更新
                cfg.weight = blended_weight
                updated[name] = blended_weight
        except ValueError:
            pass
    
    return updated


def _update_reward_weights(env: ManagerBasedRLEnv, stage: int) -> dict:
    """
    更新奖励权重（立即更新，用于初始化）
    
    Args:
        env: 环境实例
        stage: 目标阶段
    
    Returns:
        实际更新的权重字典
    """
    params = STAGE_CONFIGS[stage]
    updated = {}
    
    reward_names = ["alive", "track_lin_vel_xy", "upward_progress", "flat_orientation_l2", "action_rate"]
    
    for name in reward_names:
        if name in params:
            try:
                cfg = env.reward_manager.get_term_cfg(name)
                old_weight = cfg.weight
                cfg.weight = params[name]
                updated[name] = {"old": old_weight, "new": params[name]}
            except ValueError:
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
    
    [优化版本] 此函数确保自适应参数与 terrain_levels 同步：
        - 定期检查（每 500 步）
        - terrain_levels 更新后强制检查
        - [问题6修复] 使用 iteration 计数而非 step 计数
        - [问题2修复] 阶段切换时启动平滑过渡
    
    Args:
        env: 环境实例
        state: 自适应状态
        force_check: 是否强制检查（terrain_levels 更新后调用时设为 True）
    
    Returns:
        是否发生了阶段切换
    """
    step_counter = env.common_step_counter
    check_interval = 500
    
    # [问题6修复] 估算当前 iteration 数
    # Isaac Lab 的 common_step_counter 通常以“环境步”(每次 env.step 同时推进所有并行环境)递增。
    # 对于 on-policy PPO：1 次训练迭代（policy update）会采样约 num_steps_per_env 个环境步。
    # 因此这里应按 num_steps_per_env 估算 iteration，而不是 num_envs * num_steps_per_env。
    steps_per_iteration = max(int(state.num_steps_per_env), 1)
    state.estimated_iteration = step_counter // steps_per_iteration
    
    # 非强制检查时，遵守检查间隔
    if not force_check and (step_counter - state.last_update_step < check_interval):
        # 但仍然需要更新平滑过渡进度
        _update_transition_progress(state, step_counter)
        if state.transition_progress < 1.0:
            _update_reward_weights_smooth(env, state)
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
    
    # [问题1修复] 检测新阶段（使用相对比例）
    new_stage = _detect_stage(avg_length, avg_terrain, state.max_episode_length)
    
    # [问题6修复] 检查是否满足最小阶段停留时间（使用 iteration 计数）
    min_duration_met = (state.estimated_iteration - state.stage_enter_iteration) >= state.min_stage_iterations
    
    if new_stage != state.current_stage and min_duration_met:
        state.stage_stable_count += 1
        
        # 连续 3 次检测到相同的新阶段才切换
        # 强制检查时（terrain_level 刚更新），只需 1 次确认即可立即切换
        required_stable_count = 1 if force_check else 3
        
        if state.stage_stable_count >= required_stable_count:
            old_stage = state.current_stage
            
            # [问题2修复] 启动平滑过渡（而不是立即跳变）
            state.source_stage = old_stage
            state.target_stage = new_stage
            state.transition_progress = 0.0
            state.transition_start_step = step_counter
            
            # 终止条件立即更新（不需要平滑过渡）
            termination_updates = _update_termination_params(env, new_stage)
            
            # 计算存活比例用于日志
            survival_ratio = avg_length / max(state.max_episode_length, 1)
            
            # 打印日志
            sync_note = " [与terrain_level同步]" if force_check else ""
            print(f"\n{'=' * 70}")
            print(f"[Adaptive Training] 阶段切换!{sync_note} [平滑过渡已启动]")
            print("-" * 70)
            print(f"  上一阶段: Stage {old_stage} ({STAGE_CONFIGS[old_stage]['name']})")
            print(f"  新 阶 段: Stage {new_stage} ({STAGE_CONFIGS[new_stage]['name']})")
            print("-" * 70)
            print("  触发指标:")
            print(f"    - Mean Episode Length: {avg_length:.1f} steps ({survival_ratio*100:.1f}% survival)")
            print(f"    - Mean Terrain Level:  {avg_terrain:.2f}")
            print(f"    - Current Iteration:   ~{state.estimated_iteration}")
            print("-" * 70)
            print("  奖励权重: 平滑过渡中... (约 {:.0f} 步完成)".format(state.transition_duration))
            
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
            state.stage_enter_iteration = state.estimated_iteration
            return True
    else:
        state.stage_stable_count = 0
    
    # [问题2修复] 持续更新平滑过渡
    _update_transition_progress(state, step_counter)
    if state.transition_progress < 1.0:
        _update_reward_weights_smooth(env, state)
    
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
    
    # ==================== 0. 缓存环境参数 ====================
    # [问题1/6修复] 缓存 max_episode_length 用于相对比例计算
    state.max_episode_length = env.max_episode_length
    # 缓存 num_envs 用于 iteration 估算
    state.num_envs = env.num_envs
    
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
        # v2.0 新增字段
        "stage_enter_iteration": state.stage_enter_iteration,
        "estimated_iteration": state.estimated_iteration,
        "max_episode_length": state.max_episode_length,
        "transition_progress": state.transition_progress,
        "source_stage": state.source_stage,
        "target_stage": state.target_stage,
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
    
    # v2.0 新增字段
    state.stage_enter_iteration = state_dict.get("stage_enter_iteration", 0)
    state.estimated_iteration = state_dict.get("estimated_iteration", 0)
    state.max_episode_length = state_dict.get("max_episode_length", 1000)
    state.transition_progress = state_dict.get("transition_progress", 1.0)
    state.source_stage = state_dict.get("source_stage", 0)
    state.target_stage = state_dict.get("target_stage", state.current_stage)
    
    # 恢复 deque
    for length in state_dict.get("episode_lengths", []):
        state.episode_lengths.append(length)
    for level in state_dict.get("terrain_levels", []):
        state.terrain_levels.append(level)
    
    # 计算平均值用于打印
    avg_length = sum(state.episode_lengths) / len(state.episode_lengths) if state.episode_lengths else 0
    avg_terrain = sum(state.terrain_levels) / len(state.terrain_levels) if state.terrain_levels else 0
    survival_ratio = avg_length / max(state.max_episode_length, 1)
    
    print(f"[Adaptive State] 从 checkpoint 恢复阶段: Stage {state.current_stage} ({get_stage_name(state.current_stage)})")
    print(f"    - mean_episode_length: {avg_length:.1f} ({survival_ratio*100:.1f}% survival)")
    print(f"    - mean_terrain_level: {avg_terrain:.2f}")
    print(f"    - estimated_iteration: ~{state.estimated_iteration}")


def init_adaptive_state_from_metrics(
    mean_episode_length: float, 
    mean_terrain_level: float = 0.0,
    max_episode_length: int = 1000,
):
    """
    根据当前训练指标初始化自适应状态（用于恢复训练时）
    
    Args:
        mean_episode_length: 当前平均 episode 长度
        mean_terrain_level: 当前平均地形等级
        max_episode_length: 最大 episode 长度（用于相对比例计算）
    """
    state = _get_adaptive_state()
    state.reset()
    
    # 缓存 max_episode_length
    state.max_episode_length = max_episode_length
    
    # [问题1修复] 根据指标检测应该处于的阶段（使用相对比例）
    stage = _detect_stage(mean_episode_length, mean_terrain_level, max_episode_length)
    state.current_stage = stage
    state.target_stage = stage
    
    # 预填充 deque 以避免需要重新积累数据
    for _ in range(50):
        state.episode_lengths.append(mean_episode_length)
        state.terrain_levels.append(mean_terrain_level)
    
    survival_ratio = mean_episode_length / max(max_episode_length, 1)
    print(f"[Adaptive State] 根据指标初始化阶段: Stage {stage} ({get_stage_name(stage)})")
    print(f"    - mean_episode_length: {mean_episode_length:.1f} ({survival_ratio*100:.1f}% survival)")
    print(f"    - mean_terrain_level: {mean_terrain_level:.2f}")
    print(f"    - max_episode_length: {max_episode_length}")


# ============================================================================
#                     已弃用函数 (v2.0 移除 PPO 动态参数)
# ============================================================================

def get_ppo_params(stage: int = None) -> dict:
    """
    [已弃用] 获取指定阶段的 PPO 参数
    
    v2.0 说明: PPO 参数不再由自适应课程动态调整。
    learning_rate 由 RSL-RL 的 adaptive schedule 根据 KL 散度自动调整。
    其他 PPO 参数保持固定以确保训练稳定性。
    
    此函数保留仅为向后兼容，返回默认值。
    """
    return {
        "learning_rate": 1e-3,
        "entropy_coef": 0.01,
        "clip_param": 0.2,
        "desired_kl": 0.01,
    }


def update_ppo_params(alg, stage: int = None) -> dict:
    """
    [已弃用] 更新 PPO 算法参数
    
    v2.0 说明: PPO 参数不再由自适应课程动态调整。
    此函数保留仅为向后兼容，不执行任何操作。
    """
    return {}


def apply_stage_params(env, alg=None, stage: int = None) -> dict:
    """
    应用指定阶段的所有参数（奖励权重、终止条件）
    
    v2.0 更新: 不再更新 PPO 参数（由 adaptive schedule 自动处理）
    
    用于训练开始时初始化，确保奖励权重和终止条件与当前阶段匹配。
    
    Args:
        env: 环境实例（用于更新奖励权重和终止条件）
        alg: [已弃用] PPO 算法实例，不再使用
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
    
    # 打印应用的参数
    state = _get_adaptive_state()
    survival_ratio = 0.0
    if state.episode_lengths:
        avg_length = sum(state.episode_lengths) / len(state.episode_lengths)
        survival_ratio = avg_length / max(state.max_episode_length, 1)
    
    stage_name = get_stage_name(stage)
    print(f"\n[Adaptive] 应用 Stage {stage} ({stage_name}) 的参数:")
    print(f"  (基于 {survival_ratio*100:.1f}% survival ratio)")
    
    if reward_updates:
        print("  奖励权重:")
        for name, vals in reward_updates.items():
            print(f"    - {name}: {vals['old']:.2f} -> {vals['new']:.2f}")
    
    if termination_updates:
        print("  终止条件:")
        for name, vals in termination_updates.items():
            old_val = vals['old'] if vals['old'] is not None else 0.0
            new_deg = vals['new'] * 57.3
            print(f"    - {name}: {old_val:.2f} -> {vals['new']:.2f} rad ({new_deg:.0f}°)")
    
    print("  PPO 参数: [由 adaptive schedule 自动管理]")
    print()
    return updates


# ============================================================================
#                     调试和监控辅助函数
# ============================================================================

def get_transition_status() -> dict:
    """
    获取当前平滑过渡状态（用于调试和监控）
    
    Returns:
        包含过渡状态的字典
    """
    state = _get_adaptive_state()
    return {
        "in_transition": state.transition_progress < 1.0,
        "progress": state.transition_progress,
        "source_stage": state.source_stage,
        "target_stage": state.target_stage,
        "current_stage": state.current_stage,
    }


def get_stage_info() -> dict:
    """
    获取当前阶段的详细信息（用于调试和监控）
    
    Returns:
        包含阶段信息的字典
    """
    state = _get_adaptive_state()
    avg_length = sum(state.episode_lengths) / len(state.episode_lengths) if state.episode_lengths else 0
    avg_terrain = sum(state.terrain_levels) / len(state.terrain_levels) if state.terrain_levels else 0
    survival_ratio = avg_length / max(state.max_episode_length, 1)
    
    return {
        "current_stage": state.current_stage,
        "stage_name": get_stage_name(state.current_stage),
        "mean_episode_length": avg_length,
        "survival_ratio": survival_ratio,
        "mean_terrain_level": avg_terrain,
        "estimated_iteration": state.estimated_iteration,
        "max_episode_length": state.max_episode_length,
        "thresholds": STAGE_THRESHOLDS,
    }
