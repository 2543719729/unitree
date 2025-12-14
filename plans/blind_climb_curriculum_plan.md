# 盲爬楼梯课程学习修改规划

## 1. 问题分析

### 1.1 当前 `terrain_levels_vel` 函数的局限性

| 问题 | 说明 | 影响 |
|------|------|------|
| **只计算水平距离** | `distance = torch.norm(root_pos_w[:, :2] - env_origins[:, :2])` | 忽略了 z 方向的高度变化 |
| **升级条件不适合楼梯** | `move_up = distance > terrain_size / 2` | 机器人可能爬了很高但水平距离不够 |
| **降级条件依赖速度命令** | `move_down = distance < cmd_vel * time * 0.5` | 盲爬任务不以速度为主要目标 |
| **原地挣扎误判** | 机器人在楼梯前尝试攀爬时距离小 | 会被错误降级到简单地形 |

### 1.2 盲爬任务的特点

```
盲爬楼梯任务特点：
├── 无地形感知（不使用 height_scan）
├── 只依赖本体感知（关节角度、速度、IMU）
├── 目标是成功攀爬楼梯，而非跟踪速度
├── 高度增益是关键评估指标
└── 需要更宽容的降级条件（允许多次尝试）
```

### 1.3 当前配置中的课程学习

从 `stair_single_blind_cfg.py` 中可以看到：

```python
@configclass
class StairCurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)  # 问题所在
    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)  # 可以保留
```

---

## 2. 解决方案设计

### 2.1 新增课程学习函数

需要在 `unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/curriculums.py` 中添加新函数：

| 函数名 | 用途 | 评估指标 |
|--------|------|----------|
| `terrain_levels_climb` | 基于攀爬进度的地形难度调整 | 前进距离 + 高度增益 |
| `terrain_levels_height` | 纯基于高度增益的地形难度调整 | 仅高度变化 |

### 2.2 设计思路

```
新的课程学习策略：
├── 综合评估指标
│   ├── 前进距离（x 方向）
│   ├── 高度增益（z 方向）- 权重更高
│   └── 可选：存活时间
├── 升级条件
│   ├── 方案A：progress > threshold（综合进度）
│   └── 方案B：height_gain > stair_height * n（爬过 n 级台阶）
├── 降级条件
│   ├── 使用绝对阈值，不依赖速度命令
│   └── 更宽容：只有几乎没有进展才降级
└── 保持条件
    └── 介于升级和降级之间
```

---

## 3. 新函数详细设计

### 3.1 `terrain_levels_climb` - 综合攀爬进度课程学习

```python
def terrain_levels_climb(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_weight: float = 2.0,           # 高度增益权重
    forward_weight: float = 1.0,          # 前进距离权重
    upgrade_threshold_ratio: float = 0.3, # 升级阈值（地形尺寸的比例）
    downgrade_threshold: float = 0.5,     # 降级阈值（绝对值，米）
) -> torch.Tensor:
    """
    基于攀爬进度的地形难度课程学习函数（适合盲爬楼梯任务）
    
    评估指标：
        progress = forward_distance * forward_weight + height_gain * height_weight
    
    课程策略：
        - 升级：progress > terrain_size * upgrade_threshold_ratio
        - 降级：progress < downgrade_threshold
        - 保持：介于两者之间
    """
```

**核心算法：**

```python
# 计算3D位移
displacement = asset.data.root_pos_w[env_ids, :3] - env.scene.env_origins[env_ids, :3]

# 前进距离（x 方向，假设机器人面向 +x）
forward_distance = displacement[:, 0]

# 高度增益（z 方向）
height_gain = displacement[:, 2]

# 综合进度评分
progress = forward_distance * forward_weight + height_gain * height_weight

# 升级条件：综合进度超过阈值
move_up = progress > terrain.cfg.terrain_generator.size[0] * upgrade_threshold_ratio

# 降级条件：几乎没有进展（使用绝对阈值）
move_down = progress < downgrade_threshold
move_down *= ~move_up

# 更新地形等级
terrain.update_env_origins(env_ids, move_up, move_down)
```

### 3.2 `terrain_levels_height` - 纯高度课程学习

```python
def terrain_levels_height(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    upgrade_height: float = 0.3,    # 升级所需高度增益（米）
    downgrade_height: float = 0.0,  # 降级阈值（米）
) -> torch.Tensor:
    """
    纯基于高度增益的地形难度课程学习函数
    
    适用场景：
        - 纯上楼梯任务（不关心水平移动）
        - 需要精确控制难度递进的场景
    
    课程策略：
        - 升级：height_gain > upgrade_height
        - 降级：height_gain < downgrade_height
    """
```

### 3.3 参数设计说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `height_weight` | 2.0 | 高度增益的权重，楼梯任务中高度更重要 |
| `forward_weight` | 1.0 | 前进距离的权重 |
| `upgrade_threshold_ratio` | 0.3 | 升级阈值为地形尺寸的 30% |
| `downgrade_threshold` | 0.5 | 降级阈值为 0.5 米（绝对值） |
| `upgrade_height` | 0.3 | 约 2-3 级台阶的高度 |
| `downgrade_height` | 0.0 | 没有任何高度增益才降级 |

---

## 4. 实现步骤

### 4.1 修改文件清单

```
需要修改的文件：
├── unitree_rl_lab/.../mdp/curriculums.py
│   ├── 添加 terrain_levels_climb 函数
│   └── 添加 terrain_levels_height 函数
├── unitree_rl_lab/.../mdp/__init__.py
│   └── 导出新函数
└── unitree_rl_lab/.../robots/g1/29dof/stair_single_blind_cfg.py
    └── 更新 StairCurriculumCfg 使用新函数
```

### 4.2 详细实现步骤

#### 步骤 1：在 curriculums.py 中添加新函数

```python
# 在 unitree_rl_lab/.../mdp/curriculums.py 中添加

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

def terrain_levels_climb(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_weight: float = 2.0,
    forward_weight: float = 1.0,
    upgrade_threshold_ratio: float = 0.3,
    downgrade_threshold: float = 0.5,
) -> torch.Tensor:
    """基于攀爬进度的地形难度课程学习函数"""
    # ... 实现代码
```

#### 步骤 2：更新 __init__.py 导出

```python
# 在 unitree_rl_lab/.../mdp/__init__.py 中添加
from .curriculums import terrain_levels_climb, terrain_levels_height
```

#### 步骤 3：更新盲爬配置

```python
# 在 stair_single_blind_cfg.py 中修改
@configclass
class StairCurriculumCfg:
    # 使用新的攀爬进度课程学习
    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_climb,
        params={
            "height_weight": 2.0,
            "forward_weight": 1.0,
            "upgrade_threshold_ratio": 0.3,
            "downgrade_threshold": 0.5,
        }
    )
    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)
```

---

## 5. 测试验证

### 5.1 验证指标

| 指标 | 期望结果 |
|------|----------|
| 平均地形等级 | 随训练逐步上升 |
| 升级/降级比例 | 升级 > 降级 |
| 高度增益分布 | 高等级地形上高度增益更大 |
| 训练稳定性 | 不会频繁在简单/困难地形间震荡 |

### 5.2 调试建议

1. **记录课程学习指标**：在 TensorBoard 中记录 `terrain_level_mean`、`height_gain_mean`、`forward_distance_mean`
2. **可视化地形分布**：观察机器人在不同难度地形上的分布
3. **参数调优**：根据训练曲线调整 `height_weight` 和阈值参数

---

## 6. 可选增强

### 6.1 基于存活时间的课程学习

```python
def terrain_levels_survival(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    survival_ratio_upgrade: float = 0.8,  # 存活 80% 时间才升级
    survival_ratio_downgrade: float = 0.3, # 存活不足 30% 时间则降级
) -> torch.Tensor:
    """基于存活时间的课程学习"""
    # 计算存活时间比例
    survival_ratio = env.episode_length_buf[env_ids] / env.max_episode_length
    
    move_up = survival_ratio > survival_ratio_upgrade
    move_down = survival_ratio < survival_ratio_downgrade
    # ...
```

### 6.2 混合课程学习

```python
def terrain_levels_hybrid(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    # 综合考虑：攀爬进度 + 存活时间 + 奖励
) -> torch.Tensor:
    """混合多指标的课程学习"""
    # 可以根据训练阶段动态调整各指标权重
```

---

## 7. 总结

### 7.1 核心改进

| 原函数 | 新函数 | 改进点 |
|--------|--------|--------|
| `terrain_levels_vel` | `terrain_levels_climb` | 添加高度增益评估 |
| 水平距离评估 | 3D 进度评估 | 更适合楼梯任务 |
| 速度命令相关阈值 | 绝对阈值 | 不依赖速度跟踪 |
| 严格降级条件 | 宽容降级条件 | 允许多次尝试 |

### 7.2 预期效果

```
训练流程：
├── 初期：机器人在简单地形（平地、低台阶）学习基础平衡
├── 中期：逐步升级到中等难度楼梯
├── 后期：在困难楼梯上精炼技能
└── 最终：能够盲爬各种难度的楼梯
```

---

## 8. 下一步行动

1. **切换到 Code 模式**
2. **实现 `terrain_levels_climb` 函数**
3. **实现 `terrain_levels_height` 函数**
4. **更新 `__init__.py` 导出**
5. **更新 `stair_single_blind_cfg.py` 配置**
6. **测试验证**
