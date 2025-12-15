# G1 盲爬楼梯自适应训练参数切换规划

---

## 零、Isaac Lab 框架验证结果

> ⚠️ **重要**: 以下是对 Isaac Lab 源码的验证分析，确保规划的可行性。

### ✅ 奖励权重动态修改 - **可行**

**源码位置**: `IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py`

```python
# 第 164-177 行: set_term_cfg() 方法
def set_term_cfg(self, term_name: str, cfg: RewardTermCfg):
    """Sets the configuration of the specified term into the manager."""
    self._term_cfgs[self._term_names.index(term_name)] = cfg

# 第 179-194 行: get_term_cfg() 方法
def get_term_cfg(self, term_name: str) -> RewardTermCfg:
    """Gets the configuration for the specified term."""
    return self._term_cfgs[self._term_names.index(term_name)]

# 第 143-149 行: compute() 直接使用 term_cfg.weight
value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
```

**验证结论**: `_term_cfgs` 存储的是引用，直接修改 `weight` 属性即可生效：

```python
# ✅ 正确的动态修改方式
term_cfg = env.reward_manager.get_term_cfg("alive")
term_cfg.weight = 3.0  # 下次 compute() 调用时立即生效
```

---

### ✅ 终止条件参数动态修改 - **可行**

**源码位置**: `IsaacLab/source/isaaclab/isaaclab/managers/termination_manager.py`

```python
# 第 215-228 行: set_term_cfg() 方法存在
# 第 230-245 行: get_term_cfg() 方法存在
# 第 167 行: compute() 使用 term_cfg.params
value = term_cfg.func(self._env, **term_cfg.params)
```

**验证结论**: `params` 是字典引用，直接修改即可生效：

```python
# ✅ 正确的动态修改方式
term_cfg = env.termination_manager.get_term_cfg("bad_orientation")
term_cfg.params["limit_angle"] = 1.0  # 下次 compute() 调用时立即生效
```

---

### ⚠️ 课程学习函数切换 - **需要特殊处理**

**源码位置**: `IsaacLab/source/isaaclab/isaaclab/managers/curriculum_manager.py`

```python
# 第 137-139 行: compute() 调用
state = term_cfg.func(self._env, env_ids, **term_cfg.params)
```

**问题**: CurriculumManager **没有** `set_term_cfg()` 方法！

**解决方案**: 创建**统一的自适应课程函数**，在函数内部实现策略分派：

```python
# ✅ 正确的实现方式：在单一函数内实现多策略分派
def adaptive_terrain_levels(env, env_ids, **params):
    stage = _detect_training_stage(env)
    
    if stage <= 1:
        return terrain_levels_survival(env, env_ids, ...)
    elif stage == 2:
        return terrain_levels_hybrid(env, env_ids, ...)
    else:
        return terrain_levels_climb(env, env_ids, ...)
```

---

### ⚠️ 课程学习参数动态修改 - **可行但需注意**

```python
# 第 138 行: params 通过 **term_cfg.params 传递
state = term_cfg.func(self._env, env_ids, **term_cfg.params)
```

**验证结论**: 可以通过修改 `term_cfg.params` 来动态调整参数：

```python
# ✅ 需要直接访问 _term_cfgs 列表（无公开 API）
curriculum_manager = env.curriculum_manager
for i, name in enumerate(curriculum_manager._term_names):
    if name == "terrain_levels":
        curriculum_manager._term_cfgs[i].params["survival_ratio_upgrade"] = 0.8
```

---

### 📊 验证结果汇总

| 功能 | 可行性 | 实现方式 |
|------|--------|----------|
| 修改奖励权重 | ✅ 直接支持 | `get_term_cfg().weight = value` |
| 修改奖励参数 | ✅ 直接支持 | `get_term_cfg().params["key"] = value` |
| 修改终止阈值 | ✅ 直接支持 | `get_term_cfg().params["key"] = value` |
| 切换课程函数 | ⚠️ 需封装 | 创建统一分派函数 |
| 修改课程参数 | ⚠️ 需访问私有属性 | `_term_cfgs[i].params["key"] = value` |

---

## 一、目标概述

实现一个**自动检测训练阶段并动态调整参数**的机制，避免手动调参，让训练过程更加智能和高效。

---

## 二、训练阶段定义

基于关键指标自动判断当前训练阶段：

| 阶段 | 阶段名称 | 判断条件 | 训练目标 |
|------|----------|----------|----------|
| **Stage 0** | 初始探索期 | `mean_episode_length < 100` | 学会基本站立 |
| **Stage 1** | 站立稳定期 | `100 ≤ mean_episode_length < 300` | 稳定站立，开始学走路 |
| **Stage 2** | 行走学习期 | `300 ≤ mean_episode_length < 600` | 学会平地行走 |
| **Stage 3** | 楼梯适应期 | `mean_episode_length ≥ 600` 且 `terrain_level < 3` | 开始适应楼梯 |
| **Stage 4** | 楼梯精通期 | `mean_episode_length ≥ 600` 且 `terrain_level ≥ 3` | 精通楼梯攀爬 |

### 阶段检测指标

```python
# 主要指标
mean_episode_length: float     # 平均存活步数（滑动平均）
terrain_level: float           # 平均地形难度等级
upward_progress: float         # 平均向上进展奖励

# 辅助指标
survival_ratio: float          # 存活时间比例 = mean_episode_length / max_episode_length
track_vel_reward: float        # 速度跟踪奖励
```

---

## 三、各阶段参数配置

### Stage 0: 初始探索期

**目标**: 学会基本站立，避免立即倒下

| 参数类别 | 参数名 | 值 | 说明 |
|----------|--------|-----|------|
| **奖励权重** | alive | 6.0 | 最高优先级，鼓励存活 |
| | upward_progress | 0.5 | 低权重，暂不关注前进 |
| | flat_orientation_l2 | -1.0 | 较宽松，允许摇晃探索 |
| | action_rate | -0.02 | 宽松，允许探索动作 |
| **课程学习** | terrain_func | survival | 基于存活时间 |
| | survival_upgrade | 0.5 | 存活50%即可升级 |
| | survival_downgrade | 0.1 | 仅10%以下降级 |
| **终止条件** | limit_angle | 1.4 | 80°，非常宽松 |

### Stage 1: 站立稳定期

**目标**: 稳定站立，开始尝试移动

| 参数类别 | 参数名 | 值 | 说明 |
|----------|--------|-----|------|
| **奖励权重** | alive | 5.0 | 仍然重要 |
| | upward_progress | 1.0 | 开始关注前进 |
| | track_lin_vel_xy | 1.0 | 开始跟踪速度命令 |
| | flat_orientation_l2 | -1.5 | 略微收紧 |
| | action_rate | -0.03 | 略微收紧 |
| **课程学习** | terrain_func | survival | 基于存活时间 |
| | survival_upgrade | 0.6 | 存活60%升级 |
| | survival_downgrade | 0.15 | 15%以下降级 |
| **终止条件** | limit_angle | 1.3 | 74°，略微收紧 |

### Stage 2: 行走学习期

**目标**: 学会平稳行走，准备挑战楼梯

| 参数类别 | 参数名 | 值 | 说明 |
|----------|--------|-----|------|
| **奖励权重** | alive | 4.0 | 降低，鼓励更多探索 |
| | upward_progress | 2.0 | 提高，鼓励向上 |
| | track_lin_vel_xy | 1.5 | 更注重速度跟踪 |
| | flat_orientation_l2 | -2.0 | 正常强度 |
| | feet_clearance | 1.5 | 提高抬脚奖励 |
| | action_rate | -0.04 | 收紧，要求平滑动作 |
| **课程学习** | terrain_func | hybrid | 混合模式（见下文） |
| | survival_weight | 0.5 | 存活占50% |
| | progress_weight | 0.5 | 进展占50% |
| **终止条件** | limit_angle | 1.2 | 69°，进一步收紧 |

### Stage 3: 楼梯适应期

**目标**: 适应楼梯地形，学会跨越台阶

| 参数类别 | 参数名 | 值 | 说明 |
|----------|--------|-----|------|
| **奖励权重** | alive | 3.0 | 进一步降低 |
| | upward_progress | 3.5 | 高权重，核心目标 |
| | track_lin_vel_xy | 1.0 | 降低，楼梯上速度自然慢 |
| | flat_orientation_l2 | -1.5 | 放宽，允许前倾 |
| | feet_clearance | 2.0 | 高权重，必须抬腿 |
| | feet_clearance.target_height | 0.18 | 提高目标高度 |
| **课程学习** | terrain_func | climb | 基于攀爬进度 |
| | height_weight | 2.0 | 高度增益权重 |
| | forward_weight | 1.0 | 前进距离权重 |
| **终止条件** | limit_angle | 1.1 | 63°，较严格 |
| **域随机化** | push_interval | (5.0, 9.0) | 增加推力频率 |

### Stage 4: 楼梯精通期

**目标**: 稳健、高效地攀爬各种楼梯

| 参数类别 | 参数名 | 值 | 说明 |
|----------|--------|-----|------|
| **奖励权重** | alive | 2.0 | 最低，策略应自主保持平衡 |
| | upward_progress | 5.0 | 最高，核心评价指标 |
| | track_lin_vel_xy | 1.0 | 保持 |
| | energy | -5e-5 | 提高能效惩罚 |
| | feet_clearance.target_height | 0.20 | 最高目标 |
| **课程学习** | terrain_func | height | 纯高度增益 |
| | upgrade_height | 0.4 | 更高阈值 |
| **终止条件** | limit_angle | 1.0 | 57°，最严格 |
| **域随机化** | push_interval | (4.0, 8.0) | 频繁推力 |
| | push_velocity | (-0.5, 0.5) | 更大推力 |

---

## 四、实现架构

### 4.1 核心组件

```
┌─────────────────────────────────────────────────────────────┐
│                    AdaptiveTrainingManager                   │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ StageDetector│  │ParamScheduler│  │ MetricsTracker│      │
│  │              │  │              │  │               │      │
│  │ - 检测当前   │  │ - 管理各阶段 │  │ - 滑动平均    │      │
│  │   训练阶段   │  │   参数配置   │  │ - 指标记录    │      │
│  │ - 阶段转换   │  │ - 平滑切换   │  │ - 趋势分析    │      │
│  └──────────────┘  └──────────────┘  └───────────────┘      │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                  动态参数更新                          │   │
│  │  - reward_manager.set_term_weight()                   │   │
│  │  - curriculum 策略切换                                 │   │
│  │  - termination 阈值调整                               │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 文件结构

```
unitree_rl_lab/tasks/locomotion/mdp/
├── curriculums.py           # 现有课程学习函数
├── adaptive_curriculum.py   # [新建] 自适应课程学习模块
│   ├── class AdaptiveTrainingManager    # 自适应训练管理器
│   ├── class StageDetector              # 阶段检测器
│   ├── class ParamScheduler             # 参数调度器
│   ├── class MetricsTracker             # 指标追踪器
│   └── STAGE_CONFIGS                    # 各阶段参数配置字典
└── __init__.py              # 添加新模块导出
```

---

## 五、核心代码设计

### 5.1 阶段检测器 (StageDetector)

```python
class StageDetector:
    """训练阶段自动检测器"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.episode_lengths = deque(maxlen=window_size)
        self.terrain_levels = deque(maxlen=window_size)
        self.current_stage = 0
        self.stage_stable_count = 0  # 阶段稳定计数，防止频繁切换
        
    def update(self, episode_length: float, terrain_level: float) -> int:
        """更新指标并返回当前阶段"""
        self.episode_lengths.append(episode_length)
        self.terrain_levels.append(terrain_level)
        
        mean_length = np.mean(self.episode_lengths)
        mean_terrain = np.mean(self.terrain_levels)
        
        new_stage = self._determine_stage(mean_length, mean_terrain)
        
        # 阶段切换需要连续稳定 N 次
        if new_stage != self.current_stage:
            self.stage_stable_count += 1
            if self.stage_stable_count >= 10:  # 连续10次才切换
                self.current_stage = new_stage
                self.stage_stable_count = 0
        else:
            self.stage_stable_count = 0
            
        return self.current_stage
    
    def _determine_stage(self, mean_length: float, mean_terrain: float) -> int:
        """根据指标判断应处于哪个阶段"""
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
```

### 5.2 参数调度器 (ParamScheduler)

```python
class ParamScheduler:
    """参数平滑切换调度器"""
    
    def __init__(self, transition_steps: int = 1000):
        self.transition_steps = transition_steps
        self.current_params = None
        self.target_params = None
        self.transition_progress = 0
        
    def set_target_stage(self, stage: int):
        """设置目标阶段，开始平滑过渡"""
        self.target_params = STAGE_CONFIGS[stage]
        self.transition_progress = 0
        
    def step(self) -> dict:
        """每步调用，返回当前应使用的参数"""
        if self.target_params is None:
            return self.current_params
            
        self.transition_progress += 1
        alpha = min(1.0, self.transition_progress / self.transition_steps)
        
        # 线性插值
        interpolated = {}
        for key, target_val in self.target_params.items():
            if self.current_params and key in self.current_params:
                current_val = self.current_params[key]
                interpolated[key] = current_val + alpha * (target_val - current_val)
            else:
                interpolated[key] = target_val
                
        if alpha >= 1.0:
            self.current_params = self.target_params
            self.target_params = None
            
        return interpolated
```

### 5.3 自适应课程学习函数

```python
def adaptive_terrain_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    自适应地形难度课程学习
    
    根据当前训练阶段自动选择合适的课程学习策略：
    - Stage 0-1: 使用 terrain_levels_survival
    - Stage 2: 使用混合策略
    - Stage 3-4: 使用 terrain_levels_climb
    """
    # 获取或创建全局管理器
    manager = _get_adaptive_manager(env)
    
    # 更新指标并获取当前阶段
    stage = manager.update_and_get_stage(env, env_ids)
    
    # 根据阶段选择课程策略
    if stage <= 1:
        return terrain_levels_survival(env, env_ids, asset_cfg,
            survival_ratio_upgrade=manager.params["survival_upgrade"],
            survival_ratio_downgrade=manager.params["survival_downgrade"])
    elif stage == 2:
        return terrain_levels_hybrid(env, env_ids, asset_cfg, manager.params)
    else:
        return terrain_levels_climb(env, env_ids, asset_cfg,
            height_weight=manager.params["height_weight"],
            forward_weight=manager.params["forward_weight"])
```

### 5.4 奖励权重动态调整（已验证可行）

```python
def update_reward_weights(env: ManagerBasedRLEnv, params: dict):
    """
    动态更新奖励权重
    
    ✅ 已验证：RewardManager.get_term_cfg() 返回配置引用，
       直接修改 weight 属性会在下次 compute() 时生效
    """
    reward_manager = env.reward_manager
    
    # 奖励项映射：参数名 -> 奖励项名
    weight_mappings = {
        "alive": "alive",
        "upward_progress": "upward_progress",
        "track_lin_vel_xy": "track_lin_vel_xy",
        "flat_orientation_l2": "flat_orientation_l2",
        "feet_clearance": "feet_clearance",
        "action_rate": "action_rate",
    }
    
    for param_key, reward_name in weight_mappings.items():
        if param_key in params:
            try:
                term_cfg = reward_manager.get_term_cfg(reward_name)
                term_cfg.weight = params[param_key]  # ✅ 直接修改引用
            except ValueError:
                pass  # 奖励项不存在，跳过


def update_termination_params(env: ManagerBasedRLEnv, params: dict):
    """
    动态更新终止条件参数
    
    ✅ 已验证：TerminationManager.get_term_cfg() 返回配置引用，
       直接修改 params 字典会在下次 compute() 时生效
    """
    termination_manager = env.termination_manager
    
    # 终止条件参数映射
    if "limit_angle" in params:
        try:
            term_cfg = termination_manager.get_term_cfg("bad_orientation")
            term_cfg.params["limit_angle"] = params["limit_angle"]  # ✅ 直接修改
        except ValueError:
            pass
```

---

## 六、配置文件修改

### 6.1 StairCurriculumCfg 修改

```python
@configclass
class StairCurriculumCfg:
    """自适应课程学习配置"""
    
    # 使用自适应地形课程学习
    terrain_levels = CurrTerm(
        func=mdp.adaptive_terrain_levels,
        params={
            "enable_adaptive": True,
            "stage_transition_steps": 1000,  # 阶段切换平滑步数
            "metrics_window_size": 100,      # 指标滑动窗口大小
        }
    )
    
    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)
```

### 6.2 StairBlindEnvCfg 添加回调

```python
def __post_init__(self):
    # ... 现有代码 ...
    
    # 启用自适应训练参数调整
    self.enable_adaptive_training = True
```

---

## 七、日志和监控

### 7.1 TensorBoard 日志

在训练过程中记录以下指标：

```python
# 阶段相关
"Adaptive/current_stage"          # 当前训练阶段 (0-4)
"Adaptive/stage_transition"       # 阶段转换事件

# 指标追踪
"Adaptive/mean_episode_length"    # 滑动平均存活步数
"Adaptive/mean_terrain_level"     # 滑动平均地形等级
"Adaptive/mean_upward_progress"   # 滑动平均向上进展

# 参数变化
"Adaptive/alive_weight"           # alive 奖励当前权重
"Adaptive/upward_progress_weight" # upward_progress 当前权重
"Adaptive/limit_angle"            # 当前姿态限制角度
```

### 7.2 控制台输出

```
================================================================================
[Adaptive Training] Stage Transition Detected!
--------------------------------------------------------------------------------
Previous Stage: 1 (站立稳定期)
New Stage:      2 (行走学习期)
--------------------------------------------------------------------------------
Metrics:
  - Mean Episode Length: 312.5 steps
  - Mean Terrain Level:  1.2
  - Survival Ratio:      31.25%
--------------------------------------------------------------------------------
Parameter Changes:
  - alive:            5.0 -> 4.0
  - upward_progress:  1.0 -> 2.0
  - limit_angle:      1.3 -> 1.2 rad
  - terrain_func:     survival -> hybrid
================================================================================
```

---

## 八、安全机制

### 8.1 阶段切换保护

1. **稳定性检查**: 新阶段条件需连续满足 10 次才触发切换
2. **回退机制**: 如果切换后性能大幅下降，自动回退到上一阶段
3. **最小停留时间**: 每个阶段至少停留 5000 步

### 8.2 参数边界限制

```python
PARAM_BOUNDS = {
    "alive": (1.0, 8.0),
    "upward_progress": (0.5, 6.0),
    "flat_orientation_l2": (-3.0, -0.5),
    "limit_angle": (0.8, 1.5),
}
```

### 8.3 异常处理

- 如果指标出现 NaN，保持当前阶段不变
- 如果连续 1000 步无进展，触发探索模式（降低惩罚权重）

---

## 九、实现步骤

### Phase 1: 基础框架 (预计 1-2 小时)

- [ ] 创建 `adaptive_curriculum.py` 文件
- [ ] 实现 `StageDetector` 类
- [ ] 实现 `MetricsTracker` 类
- [ ] 实现 `STAGE_CONFIGS` 配置字典

### Phase 2: 参数调度 (预计 1-2 小时)

- [ ] 实现 `ParamScheduler` 类
- [ ] 实现 `AdaptiveTrainingManager` 类
- [ ] 实现 `adaptive_terrain_levels` 函数
- [ ] 实现 `update_reward_weights` 函数

### Phase 3: 集成测试 (预计 2-3 小时)

- [ ] 修改 `stair_single_blind_cfg.py` 使用自适应课程
- [ ] 添加 TensorBoard 日志
- [ ] 短期训练测试（1000 iterations）
- [ ] 验证阶段切换正确性

### Phase 4: 优化调整 (预计 1-2 小时)

- [ ] 根据测试结果调整阶段阈值
- [ ] 优化参数平滑过渡
- [ ] 添加安全机制
- [ ] 完善文档

---

## 十、预期效果

### 训练曲线预期

```
Episode Length
    ^
1000|                                    ┌──────────
    |                               ┌────┘
 600|                          ┌────┘
    |                     ┌────┘
 300|                ┌────┘
    |           ┌────┘
 100|      ┌────┘
    | ─────┘
    └──────────────────────────────────────────> Iterations
         Stage0  Stage1  Stage2   Stage3   Stage4
         0-5k    5k-15k  15k-35k  35k-60k  60k+
```

### 与手动调参对比

| 指标 | 手动调参 | 自适应调参 |
|------|----------|------------|
| 达到 Stage 4 所需迭代 | ~80,000 | ~60,000 (预期) |
| 调参人工介入次数 | 3-5 次 | 0 次 |
| 陷入局部最优风险 | 高 | 低 |
| 可复现性 | 依赖经验 | 完全自动化 |

---

## 十一、修正后的精简实现方案

基于验证结果，以下是**最精简可行**的实现方案：

### 11.1 核心思路

**不需要**复杂的 Manager 类，只需：
1. 一个**自适应课程函数**（在内部分派不同策略）
2. 一个**参数更新辅助函数**（在课程函数中调用）

### 11.2 最小实现代码

```python
# ==================== adaptive_curriculum.py ====================

from collections import deque
import torch
from typing import Sequence
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

# ==================== 全局状态存储 ====================
_adaptive_state = {
    "episode_lengths": deque(maxlen=100),
    "current_stage": 0,
    "stage_stable_count": 0,
    "last_update_step": 0,
}

# ==================== 阶段参数配置 ====================
STAGE_CONFIGS = {
    0: {  # 初始探索期
        "alive": 6.0, "upward_progress": 0.5, "flat_orientation_l2": -1.0,
        "limit_angle": 1.4, "survival_upgrade": 0.5, "survival_downgrade": 0.1,
    },
    1: {  # 站立稳定期
        "alive": 5.0, "upward_progress": 1.0, "flat_orientation_l2": -1.5,
        "limit_angle": 1.3, "survival_upgrade": 0.6, "survival_downgrade": 0.15,
    },
    2: {  # 行走学习期
        "alive": 4.0, "upward_progress": 2.0, "flat_orientation_l2": -2.0,
        "limit_angle": 1.2, "height_weight": 1.5, "forward_weight": 1.0,
    },
    3: {  # 楼梯适应期
        "alive": 3.0, "upward_progress": 3.5, "flat_orientation_l2": -1.5,
        "limit_angle": 1.1, "height_weight": 2.0, "forward_weight": 1.0,
    },
    4: {  # 楼梯精通期
        "alive": 2.0, "upward_progress": 5.0, "flat_orientation_l2": -1.5,
        "limit_angle": 1.0, "height_weight": 2.5, "forward_weight": 0.5,
    },
}


def _detect_stage(mean_length: float, mean_terrain: float) -> int:
    """根据指标判断训练阶段"""
    if mean_length < 100:
        return 0
    elif mean_length < 300:
        return 1
    elif mean_length < 600:
        return 2
    elif mean_terrain < 3:
        return 3
    else:
        return 4


def _update_params(env: ManagerBasedRLEnv, stage: int):
    """更新奖励和终止条件参数"""
    params = STAGE_CONFIGS[stage]
    
    # 更新奖励权重
    for name in ["alive", "upward_progress", "flat_orientation_l2"]:
        if name in params:
            try:
                cfg = env.reward_manager.get_term_cfg(name)
                cfg.weight = params[name]
            except ValueError:
                pass
    
    # 更新终止条件
    if "limit_angle" in params:
        try:
            cfg = env.termination_manager.get_term_cfg("bad_orientation")
            cfg.params["limit_angle"] = params["limit_angle"]
        except ValueError:
            pass


def adaptive_terrain_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    自适应地形难度课程学习（单一入口函数）
    
    ✅ 已验证可行：
    - 内部根据阶段分派不同策略
    - 自动更新奖励权重和终止条件
    """
    global _adaptive_state
    state = _adaptive_state
    
    # 1. 更新指标
    mean_length = env.episode_length_buf.float().mean().item()
    state["episode_lengths"].append(mean_length)
    
    # 2. 检测阶段（每 1000 步检查一次）
    if env.common_step_counter - state["last_update_step"] >= 1000:
        state["last_update_step"] = env.common_step_counter
        
        avg_length = sum(state["episode_lengths"]) / len(state["episode_lengths"])
        terrain_level = env.scene.terrain.terrain_levels.float().mean().item()
        
        new_stage = _detect_stage(avg_length, terrain_level)
        
        # 阶段切换稳定性检查
        if new_stage != state["current_stage"]:
            state["stage_stable_count"] += 1
            if state["stage_stable_count"] >= 5:  # 连续 5 次才切换
                print(f"\n[Adaptive] Stage {state['current_stage']} -> {new_stage}")
                state["current_stage"] = new_stage
                state["stage_stable_count"] = 0
                _update_params(env, new_stage)
        else:
            state["stage_stable_count"] = 0
    
    # 3. 根据阶段调用不同的课程策略
    stage = state["current_stage"]
    params = STAGE_CONFIGS[stage]
    
    if stage <= 1:
        from .curriculums import terrain_levels_survival
        return terrain_levels_survival(
            env, env_ids, asset_cfg,
            survival_ratio_upgrade=params.get("survival_upgrade", 0.7),
            survival_ratio_downgrade=params.get("survival_downgrade", 0.2),
        )
    else:
        from .curriculums import terrain_levels_climb
        return terrain_levels_climb(
            env, env_ids, asset_cfg,
            height_weight=params.get("height_weight", 2.0),
            forward_weight=params.get("forward_weight", 1.0),
        )
```

### 11.3 配置文件修改

```python
# stair_single_blind_cfg.py 中只需修改一行

@configclass
class StairCurriculumCfg:
    # 替换原来的 terrain_levels_survival
    terrain_levels = CurrTerm(func=mdp.adaptive_terrain_levels)
    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)
```

### 11.4 __init__.py 导出

```python
# mdp/__init__.py 添加
from .adaptive_curriculum import adaptive_terrain_levels
```

---

## 十二、后续扩展

1. **多任务适配**: 将框架扩展到其他任务（平地行走、下楼梯等）
2. **超参数自动优化**: 使用贝叶斯优化自动调整阶段阈值
3. **分布式训练支持**: 在多 GPU 环境下同步阶段状态
4. **可视化工具**: 开发实时监控阶段切换的 Web 界面

---

*文档创建时间: 2024年12月*
*版本: v1.1 (已验证)*
