# `base_height_relative` 集成到 `stair_single_blind_cfg.py` 规划

## 📊 当前配置分析

### 现有高度相关惩罚

| 奖励项 | 权重 | 作用 | 问题 |
|--------|------|------|------|
| `flat_orientation_l2` | -2.0 | 保持躯干直立 | 间接约束，不直接控制高度 |
| `joint_deviation_knees` | -0.5 | 防止膝关节过度弯曲 | 间接约束，效果有限 |

### 问题

1. **缺乏直接高度约束**：机器人可能学会以低姿态行走
2. **间接约束不够强**：姿态正确但身体重心可能偏低
3. **楼梯任务特殊性**：需要区分"蹲下"和"爬升"

---

## ✅ 推荐方案：添加 `base_height_relative`

### 方案 A：补充现有奖励（推荐）

在 `StairBlindRewardsCfg` 中添加 `base_height_relative`，与现有奖励协同工作：

```python
# ====================== 姿态奖励 ======================
# 增强姿态惩罚，保持躯干直立
flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)

# 新增：相对高度惩罚（只惩罚下降，不惩罚爬楼梯导致的高度增加）
# 这是直接的高度约束，比间接的膝关节惩罚更有效
base_height = RewTerm(
    func=mdp.base_height_relative,
    weight=-1.5,  # 中等权重，与 flat_orientation_l2 配合
    params={
        "target_offset": 0.0,         # 保持初始高度
        "only_penalize_drop": True,   # 关键：只惩罚下降，不惩罚爬楼梯
    },
)

# 保留：膝关节弯曲惩罚（降低权重，因为有了直接高度约束）
joint_deviation_knees = RewTerm(
    func=mdp.joint_deviation_l1,
    weight=-0.3,  # 从 -0.5 降低到 -0.3
    params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_joint"])
    },
)
```

### 方案 B：替换现有奖励

如果觉得奖励项太多，可以用 `base_height_relative` 替换 `joint_deviation_knees`：

```python
# ====================== 姿态奖励 ======================
flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)

# 用 base_height_relative 替换 joint_deviation_knees
# 直接约束高度比间接约束膝关节更有效
base_height = RewTerm(
    func=mdp.base_height_relative,
    weight=-2.0,  # 权重稍高，因为没有其他高度约束
    params={
        "target_offset": 0.0,
        "only_penalize_drop": True,
    },
)

# 移除 joint_deviation_knees
```

---

## 🔧 权重调整建议

### 奖励权重平衡分析

```
任务奖励（正向激励）:
  - track_lin_vel_xy: 1.0      # 速度跟踪
  - track_ang_vel_z: 0.5       # 角速度跟踪
  - alive: 2.0                 # 存活奖励
  - upward_progress: 1.5       # 向上进展
  - gait: 0.5                  # 步态奖励
  - feet_clearance: 1.2        # 抬腿高度
  总计: ~6.7

惩罚（负向抑制）:
  - base_linear_velocity: -0.5  # Z轴速度
  - base_angular_velocity: -0.05 # XY轴角速度
  - joint_vel: -0.001           # 关节速度
  - joint_acc: -2.5e-7          # 关节加速度
  - action_rate: -0.05          # 动作率
  - dof_pos_limits: -1.0        # 关节限制
  - energy: -2e-5               # 能量
  - joint_deviation_*: ~-2.4    # 关节偏差
  - flat_orientation_l2: -2.0   # 姿态
  - feet_slide: -0.3            # 脚滑动
  - undesired_contacts: -0.8    # 不期望接触
  总计: ~-7.1

新增 base_height:
  - 建议权重: -1.5 到 -2.0
  - 不会显著改变奖励平衡
```

### 推荐权重配置

| 场景 | `base_height` 权重 | `joint_deviation_knees` 权重 | 说明 |
|------|-------------------|------------------------------|------|
| 保守方案 | -1.0 | -0.5（保持不变） | 最小改动 |
| 推荐方案 | -1.5 | -0.3（降低） | 平衡配合 |
| 激进方案 | -2.0 | 0（移除） | 完全替换 |

---

## 📝 具体修改步骤

### Step 1: 在 `StairBlindRewardsCfg` 中添加奖励项

位置：第 462-478 行（姿态奖励部分）

```python
# ====================== 姿态奖励 ======================
# 增强姿态惩罚，保持躯干直立
flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)

# 新增：相对高度惩罚
# 只惩罚高度下降（蹲下/摔倒），不惩罚高度增加（爬楼梯）
# 这解决了传统 base_height_l2 在楼梯上的问题
base_height = RewTerm(
    func=mdp.base_height_relative,
    weight=-1.5,
    params={
        "target_offset": 0.0,
        "only_penalize_drop": True,
    },
)

# 膝关节弯曲惩罚（降低权重，与 base_height 配合）
joint_deviation_knees = RewTerm(
    func=mdp.joint_deviation_l1,
    weight=-0.3,  # 从 -0.5 降低
    params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_joint"])
    },
)
```

### Step 2: 验证 mdp 模块导出

确保 `base_height_relative` 已在 `mdp/__init__.py` 中导出。

---

## ⚠️ 注意事项

### 1. 与 `upward_progress` 的交互

```
upward_progress (weight=1.5):
  - 奖励高度增加
  - 每步高度增量 × 1.5

base_height_relative (weight=-1.5):
  - 惩罚高度下降
  - 只在 current_height < initial_height 时生效

交互关系：互补而非冲突
  - upward_progress: 鼓励向上爬
  - base_height_relative: 惩罚向下掉
```

### 2. 与 `flat_orientation_l2` 的交互

```
flat_orientation_l2 (weight=-2.0):
  - 惩罚躯干倾斜
  - 间接保持站立姿态

base_height_relative (weight=-1.5):
  - 直接惩罚高度下降
  - 更精确的高度控制

交互关系：互补
  - flat_orientation: 控制姿态角度
  - base_height: 控制质心高度
```

### 3. 训练建议

1. **逐步增加权重**：如果训练不稳定，可以从 `-1.0` 开始，逐步增加到 `-1.5`
2. **监控日志**：观察 `base_height` 奖励项的值，确保不会过大
3. **对比实验**：可以对比有无 `base_height` 的训练效果

---

## 📊 预期效果

| 指标 | 无 `base_height` | 有 `base_height` |
|------|-----------------|-----------------|
| 站立高度一致性 | 可能偏低 | 更稳定 |
| 爬楼梯时惩罚 | 无误惩罚 | 无误惩罚（only_penalize_drop） |
| 蹲下时惩罚 | 间接惩罚 | 直接惩罚 |
| 摔倒检测 | 依赖终止条件 | 提前预警 |

---

## ✅ 总结

推荐使用**方案 A（补充现有奖励）**：

1. 添加 `base_height_relative` 奖励，权重 `-1.5`
2. 设置 `only_penalize_drop=True`（关键！）
3. 降低 `joint_deviation_knees` 权重到 `-0.3`
4. 保留 `flat_orientation_l2` 不变

这样可以：
- ✅ 直接约束高度，防止蹲着走
- ✅ 不惩罚爬楼梯导致的高度增加
- ✅ 与现有奖励协同工作
- ✅ 最小化对训练稳定性的影响
