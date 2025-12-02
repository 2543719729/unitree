# Unitree G1 上楼梯任务实现报告

> **项目**: Unitree RL Lab  
> **任务**: Unitree-G1-29dof-Stair  
> **日期**: 2025-12-01  
> **基于**: docs/stair.md 规划文档

---

## 1. 实现概述

本报告记录了根据 `docs/stair.md` 规划文档，为 Unitree G1 29DOF 人形机器人实现上楼梯强化学习任务的完整过程。

### 1.1 实现目标

| 目标 | 状态 |
|------|------|
| 在 Isaac Lab 中引入标准楼梯地形 | ✅ 已完成 |
| 观测空间加入高度扫描信息 | ✅ 已完成 |
| 实现楼梯专用奖励函数 | ✅ 已完成 |
| 调整命令空间专注前进 | ✅ 已完成 |
| 课程学习从低阶梯到高阶梯 | ✅ 已完成 |
| 注册新任务并测试 | ✅ 已完成 |

### 1.2 修改/新增文件清单

| 文件路径 | 操作 | 说明 |
|----------|------|------|
| `tasks/locomotion/robots/g1/29dof/stair_env_cfg.py` | 新增 | 楼梯环境完整配置（627行） |
| `tasks/locomotion/mdp/rewards.py` | 修改 | 新增楼梯专用奖励函数 |
| `tasks/locomotion/robots/g1/29dof/__init__.py` | 修改 | 注册新任务 |

---

## 2. 楼梯地形配置 (STAIR_TERRAIN_CFG)

### 2.1 地形生成器参数

```python
STAIR_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),              # 每个地形块 8m × 8m
    border_width=20.0,            # 20m 边界宽度
    num_rows=10,                  # 10 行（难度等级）
    num_cols=20,                  # 20 列（每个难度的变体）
    horizontal_scale=0.1,         # 0.1m 水平分辨率
    vertical_scale=0.005,         # 0.005m 垂直分辨率
    slope_threshold=0.75,         # 斜坡阈值
    difficulty_range=(0.0, 1.0),  # 难度范围 0-100%
    use_cache=False,              # 不使用缓存，保证地形多样性
)
```

### 2.2 子地形配置

| 子地形类型 | 比例 | 阶高范围 | 踏面宽度 | 用途 |
|-----------|------|---------|---------|------|
| `flat` (平地) | 15% | - | - | 基础平衡学习 |
| `stairs_easy` (简单楼梯) | 30% | 8-12cm | 35cm | 初期学习 |
| `stairs_medium` (中等楼梯) | 25% | 12-18cm | 30cm | 标准室内楼梯 |
| `stairs_hard` (困难楼梯) | 15% | 16-20cm | 28cm | 能力上限提升 |
| `stairs_down` (下楼梯) | 15% | 10-15cm | 32cm | 下楼能力训练 |

### 2.3 楼梯参数设计依据

根据《机器人上楼梯配置调研》：
- **标准室内楼梯阶高**: 15-20cm
- **标准室内楼梯踏面**: 25-30cm
- **G1 机器人腿长约束**: 最大抬腿高度约 25cm

采用渐进式难度设计：从 8cm 低阶梯开始，逐步增加到 20cm 高阶梯。

---

## 3. 高度扫描观测配置

### 3.1 RayCaster 传感器配置

```python
height_scanner = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link",
    offset=RayCasterCfg.OffsetCfg(pos=(0.2, 0.0, 20.0)),  # 略微前移
    ray_alignment="yaw",
    pattern_cfg=patterns.GridPatternCfg(
        resolution=0.1,      # 10cm 分辨率
        size=[1.6, 1.0]      # 1.6m × 1.0m 扫描范围
    ),
    debug_vis=False,
    mesh_prim_paths=["/World/ground"],
)
```

### 3.2 观测空间设计

**策略网络观测 (PolicyCfg)**:
| 观测项 | 维度 | 噪声 | 说明 |
|--------|------|------|------|
| base_ang_vel | 3 | ±0.2 | 基座角速度 |
| projected_gravity | 3 | ±0.05 | 重力投影 |
| velocity_commands | 3 | - | 速度命令 |
| joint_pos_rel | 29 | ±0.01 | 关节位置偏差 |
| joint_vel_rel | 29 | ±1.5 | 关节速度 |
| last_action | 29 | - | 上一步动作 |
| **height_scan** | **160** | **±0.1** | **楼梯高度扫描** |

**评论家网络观测 (CriticCfg)**:
- 包含所有策略观测
- 额外添加 `base_lin_vel`（特权信息）
- `height_scan` 无噪声版本

### 3.3 历史观测

```python
history_length = 5  # 保留 5 帧历史
```

---

## 4. 楼梯专用奖励函数

### 4.1 新增奖励函数

#### 4.1.1 `upward_progress` - 向上进展奖励

**目的**: 鼓励机器人向上攀爬楼梯

```python
def upward_progress(env, asset_cfg):
    # 计算每步高度增量
    height_delta = current_height - prev_height
    
    # 正向增量给予奖励，负向增量小惩罚
    reward = clamp(height_delta, -0.1, 0.2)
    
    # 额外奖励：相对初始高度的总进展
    progress_bonus = clamp(total_progress * 0.1, 0, 0.5)
    
    return reward + progress_bonus
```

**权重**: 1.5（核心奖励）

#### 4.1.2 `base_height_adaptive` - 自适应高度惩罚

**目的**: 相对于脚下地形计算高度偏差，避免在楼梯上被错误惩罚

```python
def base_height_adaptive(env, target_height, sensor_cfg, asset_cfg):
    # 使用 RayCaster 获取地形高度
    terrain_height = ray_hits[:, center_rays, 2].mean()
    
    # 计算相对高度
    relative_height = base_height_world - terrain_height
    
    # 与目标高度比较
    height_error = (relative_height - target_height)²
    
    return height_error
```

**权重**: -5.0（惩罚过低/过高姿态）

### 4.2 奖励权重调整

| 奖励项 | 权重 | 调整说明 |
|--------|------|---------|
| track_lin_vel_xy | 1.0 | 保持 |
| track_ang_vel_z | 0.5 | 保持 |
| alive | 0.15 | 保持 |
| **upward_progress** | **1.5** | **新增** |
| base_linear_velocity | -1.0 | 减小 Z 轴惩罚 |
| flat_orientation_l2 | -3.0 | 减小，允许前倾 |
| **base_height_relative** | **-5.0** | **替换原 base_height** |
| feet_clearance | 1.2 | target_height 提高到 15cm |
| feet_slide | -0.3 | 增大滑动惩罚 |
| joint_deviation_waists | -0.8 | 允许略微前倾 |

---

## 5. 命令空间配置

### 5.1 速度命令范围

**初始范围**（训练初期）:
```python
ranges = Ranges(
    lin_vel_x=(0.1, 0.3),     # 前进为主
    lin_vel_y=(-0.05, 0.05),  # 限制横向
    ang_vel_z=(-0.1, 0.1),    # 限制转向
)
```

**最终范围**（课程学习后）:
```python
limit_ranges = Ranges(
    lin_vel_x=(0.2, 0.6),     # 前进速度
    lin_vel_y=(-0.1, 0.1),    # 小幅横向
    ang_vel_z=(-0.15, 0.15),  # 小幅转向
)
```

### 5.2 设计依据

- 楼梯攀爬主要是**前进**任务
- 过大的横向/转向速度容易导致踩空
- 速度不宜过快，保证稳定性

---

## 6. 事件配置（域随机化）

### 6.1 启动时事件

| 事件 | 参数 | 说明 |
|------|------|------|
| physics_material | friction: 0.4-1.0 | 楼梯摩擦变化更大 |
| add_base_mass | -1.0 ~ +3.0 kg | 质量随机化 |

### 6.2 重置时事件

| 事件 | 参数 | 说明 |
|------|------|------|
| reset_base | yaw: ±0.5 rad | 减小偏航角范围 |
| reset_robot_joints | velocity: ±0.5 | 减小初始关节速度 |

### 6.3 间隔事件

| 事件 | 间隔 | 参数 | 说明 |
|------|------|------|------|
| push_robot | 8-12s | ±0.3 m/s | 降低推力频率和强度 |

**设计依据**: 楼梯上推力干扰更危险，需要降低强度避免过早终止。

---

## 7. 终止条件配置

| 终止条件 | 参数 | 说明 |
|----------|------|------|
| time_out | 20s | 单回合最大时长 |
| base_height | < 0.25m | 略微提高阈值 |
| bad_orientation | > 0.7 rad | 略微减小，更早终止不稳定状态 |

---

## 8. 课程学习配置

### 8.1 课程策略

```python
@configclass
class StairCurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)
```

### 8.2 课程学习流程

```
训练开始
    ↓
[阶段 1] 平地 + 简单楼梯 (8-12cm)
    ↓ 成功率 > 80%
[阶段 2] 中等楼梯 (12-18cm)
    ↓ 成功率 > 80%
[阶段 3] 困难楼梯 (16-20cm)
    ↓ 成功率 > 80%
[阶段 4] 混合地形 + 下楼梯
    ↓
训练完成
```

### 8.3 地形难度与行号对应

| 行号 | 难度 | 主要地形 |
|------|------|---------|
| 0-2 | 0.0-0.2 | 平地 + 简单楼梯 |
| 3-5 | 0.3-0.5 | 中等楼梯 |
| 6-7 | 0.6-0.7 | 困难楼梯 |
| 8-9 | 0.8-1.0 | 最高难度楼梯 |

---

## 9. 环境配置类结构

### 9.1 类继承关系

```
ManagerBasedRLEnvCfg
        ↓
  StairClimbEnvCfg (训练用)
        ↓
  StairClimbPlayEnvCfg (演示用)
```

### 9.2 StairClimbEnvCfg 主要参数

```python
@configclass
class StairClimbEnvCfg(ManagerBasedRLEnvCfg):
    # 场景
    scene: StairSceneCfg = StairSceneCfg(num_envs=4096, env_spacing=2.5)

    # MDP 组件
    observations: StairObservationsCfg
    actions: StairActionsCfg
    commands: StairCommandsCfg
    rewards: StairRewardsCfg
    terminations: StairTerminationsCfg
    events: StairEventCfg
    curriculum: StairCurriculumCfg

    # 仿真参数
    decimation = 4              # 控制频率 = 200Hz / 4 = 50Hz
    episode_length_s = 20.0     # 单回合 20 秒
    sim.dt = 0.005              # 仿真步长 5ms
```

### 9.3 StairClimbPlayEnvCfg 参数

```python
@configclass
class StairClimbPlayEnvCfg(StairClimbEnvCfg):
    scene.num_envs = 32                    # 减少环境数量
    terrain_generator.num_rows = 3         # 减少地形行数
    terrain_generator.num_cols = 8         # 减少地形列数
    commands.ranges = limit_ranges         # 使用完整速度范围
```

---

## 10. 任务注册

### 10.1 注册代码

```python
# __init__.py
gym.register(
    id="Unitree-G1-29dof-Stair",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stair_env_cfg:StairClimbEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.stair_env_cfg:StairClimbPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "...rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
```

### 10.2 使用方法

**训练**:
```powershell
E:\isaacenv\python.exe scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Stair
```

**可视化测试**:
```powershell
E:\isaacenv\python.exe scripts/rsl_rl/play.py --task Unitree-G1-29dof-Stair
```

**恢复训练**:
```powershell
E:\isaacenv\python.exe scripts/rsl_rl/train.py --task Unitree-G1-29dof-Stair --resume
```

---

## 11. 代码文件详解

### 11.1 stair_env_cfg.py 结构 (627 行)

```
第 1-38 行     : 文件头注释和导入
第 40-98 行    : STAIR_TERRAIN_CFG 地形配置
第 100-152 行  : StairSceneCfg 场景配置
第 154-216 行  : StairEventCfg 事件配置
第 218-270 行  : StairCommandsCfg 命令配置
第 272-298 行  : StairActionsCfg 动作配置
第 300-393 行  : StairObservationsCfg 观测配置
第 395-528 行  : StairRewardsCfg 奖励配置
第 530-560 行  : StairTerminationsCfg 终止条件
第 562-575 行  : StairCurriculumCfg 课程学习
第 577-608 行  : StairClimbEnvCfg 主环境配置
第 610-627 行  : StairClimbPlayEnvCfg 演示配置
```

### 11.2 rewards.py 新增函数 (152 行新增)

```
第 227-267 行  : upward_progress() - 向上进展奖励
第 270-323 行  : base_height_adaptive() - 自适应高度惩罚
第 326-360 行  : stair_forward_progress() - 前进奖励
```

---

## 12. 与原 Velocity 任务对比

| 配置项 | Velocity 任务 | Stair 任务 |
|--------|--------------|-----------|
| 地形 | 平地 | 混合楼梯 |
| height_scan 观测 | 未启用 | 启用 (160维) |
| 前进速度范围 | 0.0-1.0 m/s | 0.2-0.6 m/s |
| 横向速度范围 | ±0.5 m/s | ±0.1 m/s |
| 转向速度范围 | ±1.0 rad/s | ±0.15 rad/s |
| 基座高度奖励 | 绝对高度 | 相对地形高度 |
| 脚部离地高度目标 | 8cm | 15cm |
| 推力干扰 | ±0.5 m/s | ±0.3 m/s |
| 向上进展奖励 | 无 | 权重 1.5 |

---

## 13. 后续优化建议

### 13.1 短期优化

1. **调整 upward_progress 权重**: 根据实际训练效果调整 1.0-2.0 之间
2. **增加脚部接触奖励**: 鼓励脚掌完整接触台阶
3. **调整历史长度**: 尝试 history_length=3 或 10

### 13.2 中期优化

1. **添加视觉观测**: 使用深度相机替代 RayCaster
2. **多阶段训练**: 先平地预训练，再楼梯微调
3. **添加扶手检测**: 为复杂场景做准备

### 13.3 长期目标

1. **Sim2Real 验证**: 在真实 G1 机器人上测试
2. **户外楼梯适应**: 不规则台阶、湿滑表面
3. **动态障碍规避**: 楼梯上避开行人

---

## 14. 参考资料

1. `docs/stair.md` - 原始规划文档
2. `docs/机器人上楼梯配置调研.md` - 技术调研报告
3. Cassie Blind Stair Climbing (RSS 2021) - 盲上楼梯研究
4. Isaac Lab Documentation - 官方文档
5. `velocity_env_cfg.py` - 原始速度跟踪任务配置

---

## 附录 A: 完整文件路径

```
unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/
├── tasks/
│   └── locomotion/
│       ├── mdp/
│       │   ├── __init__.py
│       │   └── rewards.py              # 修改：新增楼梯奖励函数
│       └── robots/
│           └── g1/
│               └── 29dof/
│                   ├── __init__.py     # 修改：注册新任务
│                   ├── velocity_env_cfg.py
│                   └── stair_env_cfg.py  # 新增：楼梯环境配置
```

---

## 附录 B: 地形可视化示意图

```
     ┌─────────────────────────────────────────────────────────┐
     │                      地形网格布局                        │
     │                  (10 行 × 20 列 × 8m × 8m)               │
     ├─────────────────────────────────────────────────────────┤
     │                                                         │
行 0 │  ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢  │ 难度 0.0  │
行 1 │  ▤ ▤ ▤ ▤ ▤ ▤ ▤ ▤ ▤ ▤ ▤ ▤ ▤ ▤ ▤ ▤ ▤ ▤ ▤ ▤  │ 难度 0.1  │
行 2 │  ▤ ▤ ▤ ▤ ▤ ▤ ▤ ▤ ▤ ▤ ▤ ▤ ▤ ▤ ▤ ▤ ▤ ▤ ▤ ▤  │ 难度 0.2  │
     │  ...                                                    │
行 5 │  ▥ ▥ ▥ ▥ ▥ ▥ ▥ ▥ ▥ ▥ ▥ ▥ ▥ ▥ ▥ ▥ ▥ ▥ ▥ ▥  │ 难度 0.5  │
     │  ...                                                    │
行 9 │  ▦ ▦ ▦ ▦ ▦ ▦ ▦ ▦ ▦ ▦ ▦ ▦ ▦ ▦ ▦ ▦ ▦ ▦ ▦ ▦  │ 难度 1.0  │
     │                                                         │
     │  图例: ▢=平地  ▤=简单楼梯  ▥=中等楼梯  ▦=困难楼梯      │
     └─────────────────────────────────────────────────────────┘
```

---

**报告结束**

