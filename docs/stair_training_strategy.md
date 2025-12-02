# G1 上下楼梯训练策略说明

> **任务名称**: `Unitree-G1-29dof-Stair`  
> **更新日期**: 2025-12-02  
> **基于配置**: `stair_env_cfg.py`

---

## 1. 策略概述

本策略基于强化学习（PPO 算法）训练 Unitree G1 29DOF 人形机器人在混合楼梯地形上稳定行走。核心思路来自 Cassie 盲爬楼梯研究（RSS 2021）：

> **"只需在现有行走框架中引入楼梯地形随机化，无需改变奖励函数就能学会爬楼梯"**

在此基础上，我们增加了：
- **高度扫描观测**：让策略感知前方地形
- **向上进展奖励**：额外激励爬升行为
- **课程学习**：从简单楼梯逐步过渡到困难楼梯

---

## 2. 训练环境架构

```
┌─────────────────────────────────────────────────────────────┐
│                    StairClimbEnvCfg                         │
├─────────────────────────────────────────────────────────────┤
│  场景 (StairSceneCfg)                                       │
│    ├── 地形: STAIR_TERRAIN_CFG (混合楼梯)                   │
│    ├── 机器人: Unitree G1 29DOF                             │
│    ├── 高度扫描器: RayCaster (160 条射线)                   │
│    └── 接触力传感器: ContactSensor                          │
├─────────────────────────────────────────────────────────────┤
│  MDP 组件                                                   │
│    ├── 观测: 本体感知 + height_scan (共 256+ 维)            │
│    ├── 动作: 关节位置控制 (29 维)                           │
│    ├── 命令: 前向速度为主 (0.1-0.6 m/s)                     │
│    ├── 奖励: 速度跟踪 + 向上进展 + 稳定性惩罚               │
│    ├── 终止: 跌倒 / 超时                                    │
│    └── 课程: 地形难度 + 速度范围递增                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 地形配置

### 3.1 混合地形比例

| 地形类型 | 比例 | 阶高范围 | 踏面宽度 | 用途 |
|---------|------|---------|---------|------|
| 平地 (flat) | 15% | - | - | 基础平衡 |
| 简单上楼梯 (stairs_easy) | 30% | 8-12 cm | 35 cm | 初期学习 |
| 中等上楼梯 (stairs_medium) | 25% | 12-18 cm | 30 cm | 标准室内楼梯 |
| 困难上楼梯 (stairs_hard) | 15% | 16-20 cm | 28 cm | 能力上限 |
| 下楼梯 (stairs_down) | 15% | 10-15 cm | 32 cm | 下楼能力 |

### 3.2 地形网格布局

- **尺寸**: 每块 8m × 8m
- **行数**: 10 行（对应难度等级 0.0 → 1.0）
- **列数**: 20 列（同难度下的变体）
- **课程学习**: 机器人表现好时向更难的行迁移

---

## 4. 观测空间

### 4.1 策略网络输入 (PolicyCfg)

| 观测项 | 维度 | 噪声 | 说明 |
|--------|------|------|------|
| base_ang_vel | 3 | ±0.2 | 基座角速度 |
| projected_gravity | 3 | ±0.05 | 重力投影方向 |
| velocity_commands | 3 | - | 目标速度命令 |
| joint_pos_rel | 29 | ±0.01 | 关节位置偏差 |
| joint_vel_rel | 29 | ±1.5 | 关节速度 |
| last_action | 29 | - | 上一步动作 |
| **height_scan** | **160** | **±0.1** | **前方地形高度扫描** |

- **历史帧数**: 5 帧
- **总观测维度**: 约 256 × 5 = 1280 维（含历史）

### 4.2 评论家网络输入 (CriticCfg)

- 包含策略所有观测
- 额外添加 `base_lin_vel`（特权信息）
- `height_scan` 无噪声版本

### 4.3 高度扫描器配置

```python
height_scanner = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link",
    offset=RayCasterCfg.OffsetCfg(pos=(0.2, 0.0, 0.8)),  # 前移 20cm，高度 80cm
    pattern_cfg=patterns.GridPatternCfg(
        resolution=0.1,      # 10cm 分辨率
        size=[1.6, 1.0]      # 前后 1.6m × 左右 1.0m
    ),
)
```

---

## 5. 奖励函数设计

### 5.1 任务奖励（正向激励）

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| track_lin_vel_xy | 1.0 | 跟踪目标前进/横向速度 |
| track_ang_vel_z | 0.5 | 跟踪目标转向速度 |
| alive | 0.15 | 存活奖励 |
| **upward_progress** | **1.5** | **向上攀爬进展（核心）** |
| gait | 0.5 | 步态周期性 |
| feet_clearance | 1.2 | 抬脚高度（目标 15cm） |

### 5.2 正则化惩罚（负向）

| 惩罚项 | 权重 | 说明 |
|--------|------|------|
| base_linear_velocity (z) | -1.0 | 垂直方向震荡 |
| base_angular_velocity (xy) | -0.05 | 横滚/俯仰角速度 |
| flat_orientation_l2 | -3.0 | 身体倾斜（允许轻微前倾） |
| **base_height_relative** | **-5.0** | **相对地形的高度偏差** |
| joint_vel | -0.001 | 关节速度 |
| action_rate | -0.05 | 动作变化率 |
| feet_slide | -0.3 | 脚部滑动 |
| undesired_contacts | -0.8 | 非脚部接触 |

### 5.3 关键奖励函数说明

#### `upward_progress`（向上进展）

```python
# 计算每步高度增量
height_delta = current_height - prev_height  # 限制在 [-0.1, 0.2]

# 额外奖励：本回合总爬升高度
total_progress = current_height - initial_height  # × 0.1，上限 0.5

reward = height_delta + progress_bonus
```

- 每回合开始时自动重置 `prev_height` 和 `initial_height`
- 避免跨回合累积导致的奖励污染

#### `base_height_relative`（相对高度惩罚）

- 使用 RayCaster 获取脚下地形高度
- 计算：`relative_height = base_height - terrain_height`
- 惩罚：`(relative_height - 0.78m)²`
- 这样在楼梯上爬升时不会被错误惩罚

---

## 6. 命令空间

### 6.1 速度范围

| 阶段 | lin_vel_x | lin_vel_y | ang_vel_z |
|------|-----------|-----------|-----------|
| 初始 | 0.1 ~ 0.3 m/s | ±0.05 m/s | ±0.1 rad/s |
| 最终 | 0.2 ~ 0.6 m/s | ±0.1 m/s | ±0.15 rad/s |

### 6.2 设计依据

- 楼梯攀爬以**前进**为主
- 横向/转向速度过大容易踩空
- 速度不宜过快，保证稳定性

---

## 7. 域随机化

| 随机化项 | 参数范围 | 模式 |
|----------|---------|------|
| 摩擦系数 | 0.4 ~ 1.0 | startup |
| 基座附加质量 | -1.0 ~ +3.0 kg | startup |
| 初始位置 | x,y: ±0.5m, yaw: ±0.5rad | reset |
| 初始关节速度 | ±0.5 rad/s | reset |
| 外部推力 | ±0.3 m/s (间隔 8-12s) | interval |

---

## 8. 课程学习

### 8.1 难度进阶

```
训练开始
    ↓
[阶段 1] 难度 0.0-0.2: 平地 + 简单楼梯 (8-12cm)
    ↓ 成功率提升
[阶段 2] 难度 0.3-0.5: 中等楼梯 (12-18cm)
    ↓ 成功率提升
[阶段 3] 难度 0.6-0.8: 困难楼梯 (16-20cm)
    ↓ 成功率提升
[阶段 4] 难度 0.8-1.0: 最高难度 + 下楼梯
    ↓
训练完成
```

### 8.2 课程配置

```python
curriculum = StairCurriculumCfg(
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel),  # 地形难度
    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels),  # 速度范围
)
```

---

## 9. 终止条件

| 条件 | 参数 | 说明 |
|------|------|------|
| time_out | 20s | 单回合最大时长 |
| base_height | < 0.25m | 跌倒判定 |
| bad_orientation | > 0.7 rad | 姿态异常 |

---

## 10. 训练与测试

### 10.1 启动训练

```bash
cd e:\Aunitree
E:\isaacenv\python.exe scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Stair
```

### 10.2 可视化测试

```bash
E:\isaacenv\python.exe scripts/rsl_rl/play.py --task Unitree-G1-29dof-Stair --num_envs 16
```

### 10.3 恢复训练

```bash
E:\isaacenv\python.exe scripts/rsl_rl/train.py --task Unitree-G1-29dof-Stair --resume
```

### 10.4 训练参数建议

| 参数 | 建议值 | 说明 |
|------|--------|------|
| num_envs | 4096 | 并行环境数 |
| max_iterations | 3000-5000 | 训练轮数 |
| learning_rate | 1e-3 → 1e-5 | 学习率衰减 |
| episode_length | 20s | 单回合时长 |

---

## 11. 预期训练曲线

| 阶段 | 迭代次数 | 预期行为 |
|------|---------|---------|
| 0-500 | 初期 | 学会站立、基础行走 |
| 500-1500 | 中期 | 开始尝试上简单楼梯 |
| 1500-3000 | 后期 | 稳定上中等楼梯，开始适应困难楼梯 |
| 3000+ | 收敛 | 能够稳定上下大部分楼梯 |

---

## 12. 与平地任务的关键差异

| 配置项 | Velocity 任务 | Stair 任务 |
|--------|--------------|-----------|
| 地形 | 纯平地 | 混合楼梯 |
| height_scan | 未启用 | 启用 (160维) |
| 前进速度 | 0.0-1.0 m/s | 0.1-0.6 m/s |
| 横向速度 | ±0.5 m/s | ±0.1 m/s |
| 转向速度 | ±1.0 rad/s | ±0.15 rad/s |
| 基座高度奖励 | 绝对高度 | 相对地形高度 |
| 抬脚高度目标 | 8cm | 15cm |
| 向上进展奖励 | 无 | 权重 1.5 |

---

## 13. 调试建议

1. **观测检查**: 确认 `height_scan` 数值在 [-1, 1] 内有变化（不是恒定值）
2. **奖励监控**: 观察 `upward_progress` 是否在爬楼时给出正奖励
3. **可视化**: 开启 `debug_vis=True` 查看射线分布
4. **分阶段验证**: 先在简单楼梯上验证，再逐步增加难度

---

## 14. 参考资料

1. **Blind Bipedal Stair Traversal via Sim-to-Real RL** (RSS 2021)
2. **Learning Vision-Based Bipedal Locomotion** (2023)
3. **Humanoid-Gym** (RobotEra)
4. **legged_gym** (ETH Zurich)
5. Isaac Lab 官方文档

---

**文档结束**
