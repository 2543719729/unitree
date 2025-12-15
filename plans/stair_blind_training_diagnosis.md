# 盲爬楼梯训练失败诊断报告（已修复）

> **状态**: ✅ 已修复
> **修复日期**: 2025-12-14
> **修复文件**: `stair_single_blind_cfg.py`

## 📊 训练日志分析

```
Learning iteration 41/80000
Mean episode length: 1.00           ❌ 极短！机器人立即死亡
Mean reward: -0.11                  ❌ 负奖励

Episode_Termination/base_height: 0.8499    ❌ 85%因高度过低终止
Episode_Termination/bad_orientation: 0.1503 ⚠️ 15%姿态异常

Curriculum/terrain_levels: 4.3114   ❌ 在中等难度地形上！
```

## 🔍 根本原因分析

### 问题 1: 课程学习初始化错误 [严重]

**现象**: `terrain_levels: 4.31` 表示机器人被放置在第4-5级难度地形

**根因**: 在 [`stair_single_blind_cfg.py:119`](unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/stair_single_blind_cfg.py:119)

```python
terrain = TerrainImporterCfg(
    ...
    max_init_terrain_level=STAIR_TERRAIN_CFG.num_rows - 1,  # = 10 - 1 = 9 !!!
```

**影响**: 机器人可能从最难的楼梯地形（14-18cm 阶高）开始训练！

**对比**: marching_env_cfg 使用相同设置但地形是平地，所以没问题

---

### 问题 2: 平地比例太少 [严重]

| 配置 | 平地比例 | 问题 |
|------|----------|------|
| `stair_single_blind_cfg` | **15%** | ❌ 太少 |
| `marching_env_cfg` | **30%** | ✅ 合适 |

**根因**: 在 [`stair_single_blind_cfg.py:66`](unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/stair_single_blind_cfg.py:66)

```python
"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.15),  # 只有15%是平地
```

**影响**: 
- 85% 机器人被放置在楼梯上
- 机器人还没学会站立就要学爬楼梯
- 必然导致训练失败

---

### 问题 3: 初始关节速度扰动 [严重]

| 配置 | 初始关节速度 | 问题 |
|------|-------------|------|
| `stair_single_blind_cfg` | **(-0.5, 0.5)** | ❌ 在楼梯上会导致立即失衡 |
| `marching_env_cfg` | **(0.0, 0.0)** | ✅ 安全 |

**根因**: 在 [`stair_single_blind_cfg.py:226`](unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/stair_single_blind_cfg.py:226)

```python
reset_robot_joints = EventTerm(
    func=mdp.reset_joints_by_scale,
    params={
        "position_range": (1.0, 1.0),
        "velocity_range": (-0.5, 0.5),  # 危险！
    },
)
```

**影响**: 在楼梯不平地面上，初始速度扰动会导致机器人立即失衡摔倒

---

### 问题 4: 缺少基座高度奖励 [中等]

| 配置 | base_height 奖励 | 问题 |
|------|-----------------|------|
| `stair_single_blind_cfg` | **无** | ⚠️ 无站立激励 |
| `marching_env_cfg` | **-5.0 weight, target=0.72m** | ✅ 有站立激励 |

**根因**: 在 [`stair_single_blind_cfg.py:455-458`](unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/stair_single_blind_cfg.py:455-458)

```python
# 盲爬模式：移除 base_height_l2
# 原因：在楼梯上，机器人的绝对高度会随着攀爬而增加
# 固定目标高度 0.78m 会在高处产生错误惩罚
```

**问题**: 虽然理由合理，但完全移除导致机器人没有"站起来"的动机

---

### 问题 5: 缺少步态相位观测 [轻微]

| 配置 | gait_phase 观测 | 问题 |
|------|----------------|------|
| `stair_single_blind_cfg` | **无** | ⚠️ 缺少步态信息 |
| `marching_env_cfg` | **有** | ✅ 有步态相位 |

**根因**: 在 [`stair_single_blind_cfg.py:346`](unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/stair_single_blind_cfg.py:346)

```python
# 策略观测中没有 gait_phase
def __post_init__(self):
    self.history_length = 5
    # ... 缺少 gait_phase 观测
```

**影响**: 策略需要从历史帧中隐式学习步态节奏，增加学习难度

---

## 📋 修复方案

### 修复 1: 降低初始地形难度

```python
# stair_single_blind_cfg.py, 第 119 行
terrain = TerrainImporterCfg(
    ...
    max_init_terrain_level=0,  # 从最简单的地形开始！原值是 9
```

### 修复 2: 增加平地比例

```python
# stair_single_blind_cfg.py, 第 64-98 行
sub_terrains={
    # 基础平地（50%）- 大幅增加！
    "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.50),  # 原值 0.15

    # 简单上楼梯（25%）
    "stairs_up_easy": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        proportion=0.25,  # 原值 0.35
        ...
    ),

    # 中等上楼梯（15%）
    "stairs_up_medium": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        proportion=0.15,  # 原值 0.30
        ...
    ),

    # 困难上楼梯（10%）
    "stairs_up_hard": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        proportion=0.10,  # 原值 0.20
        ...
    ),
},
```

### 修复 3: 移除初始关节速度扰动

```python
# stair_single_blind_cfg.py, 第 221-228 行
reset_robot_joints = EventTerm(
    func=mdp.reset_joints_by_scale,
    mode="reset",
    params={
        "position_range": (0.9, 1.1),  # 轻微位置扰动即可
        "velocity_range": (0.0, 0.0),  # 零初始速度！原值 (-0.5, 0.5)
    },
)
```

### 修复 4: 使用存活时间课程学习

```python
# stair_single_blind_cfg.py, 第 559-571 行
@configclass
class StairCurriculumCfg:
    # 使用基于存活时间的课程学习（更适合早期训练）
    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_survival,  # 改用 survival！原值 terrain_levels_climb
        params={
            "survival_ratio_upgrade": 0.7,   # 存活 70% 时间才升级
            "survival_ratio_downgrade": 0.2, # 存活不足 20% 时间则降级
        }
    )
    
    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)
```

### 修复 5: 添加步态相位观测

```python
# stair_single_blind_cfg.py, PolicyCfg 中添加
class PolicyCfg(ObsGroup):
    # ... 现有观测 ...
    
    # 添加步态相位观测
    gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 1.0})
    
    # 盲爬模式：不使用 height_scan
```

### 修复 6: 可选 - 添加相对高度奖励

```python
# 如果膝关节惩罚不足以保持站立，可以添加：
# (但这需要自定义一个相对高度奖励函数，暂不实现)
```

---

## 🔧 修复优先级

| 优先级 | 修复项 | 原因 |
|--------|--------|------|
| **P0** | 初始地形难度 | 最直接的原因 |
| **P0** | 平地比例 | 需要大量简单地形学习基础 |
| **P0** | 初始关节速度 | 消除不必要的扰动 |
| **P1** | 课程学习函数 | 使用存活时间更稳定 |
| **P2** | 步态相位观测 | 有助于步态学习 |

---

## 📝 修复后的完整配置对比

| 配置项 | 修复前 | 修复后 |
|--------|--------|--------|
| `max_init_terrain_level` | 9 | **0** |
| 平地比例 | 15% | **50%** |
| 初始关节速度 | (-0.5, 0.5) | **(0.0, 0.0)** |
| 课程学习函数 | terrain_levels_climb | **terrain_levels_survival** |
| 步态相位观测 | 无 | **有** |

---

## 🎯 预期效果

修复后，训练应该呈现以下特征：

1. **Episode length**: 逐渐增长（从 10+ 到 500+）
2. **terrain_levels**: 从 0 开始，缓慢上升
3. **Termination/base_height**: 比例降低（从 85% 到 < 30%）
4. **Mean reward**: 从负值逐渐变为正值

---

## ⚠️ 注意事项

1. 修复后需要**从头开始训练**，不能从之前的检查点恢复
2. 前期训练主要在平地上，可能看不到楼梯行为
3. 需要耐心等待课程学习逐步升级地形难度
4. 如果仍然不稳定，可以进一步增加平地比例到 60-70%
