# Unitree G1 上楼梯任务训练方案

> **训练策略**: 4阶段渐进式训练
> **算法**: RSL-RL PPO
> **预计总训练时间**: 24-48 小时（RTX 4090）
> **目标**: 在 Isaac Lab 仿真中实现 G1 机器人稳定上楼梯

---

## 1. 4阶段训练任务概览

| 阶段 | 任务 ID | 地形 | 传感器 | 预计时间 |
|------|---------|------|--------|---------|
| 1 | `Unitree-G1-29dof-Velocity` | 平地 | 无 height_scan | 4-8h |
| 2 | `Unitree-G1-29dof-Velocity-HeightScan` | 平地 | 有 height_scan | 2-4h |
| 3 | `Unitree-G1-29dof-Stair-Blind` | 楼梯 | 无 height_scan | 8-16h |
| 4 | `Unitree-G1-29dof-Stair` | 楼梯 | 有 height_scan | 8-16h |

**训练逻辑**:
- 阶段1→2: 学习使用 height_scan 传感器
- 阶段2→3: 迁移到楼梯地形（盲爬）
- 阶段3→4: 结合传感器提升楼梯攀爬能力

---

## 2. 训练环境准备

### 2.1 硬件要求

| 配置项 | 最低要求 | 推荐配置 |
|--------|---------|---------|
| GPU | RTX 3080 (10GB) | RTX 4090 (24GB) |
| CPU | 8核 | 16核+ |
| 内存 | 32GB | 64GB |
| 硬盘 | 50GB SSD | 100GB NVMe |

### 2.2 验证任务注册

```powershell
cd E:\Aunitree\unitree_rl_lab

# 列出所有 G1 任务
python -c "import gymnasium as gym; import unitree_rl_lab.tasks; print([t for t in gym.registry.keys() if 'G1-29dof' in t])"

# 预期输出:
# ['Unitree-G1-29dof-Velocity', 'Unitree-G1-29dof-Velocity-HeightScan',
#  'Unitree-G1-29dof-Stair-Blind', 'Unitree-G1-29dof-Stair']
```

---

## 3. 4阶段训练命令

### 3.1 阶段1：盲走（平地，无 height_scan）

**目标**: 学会基本行走和速度跟踪

```powershell
cd E:\Aunitree\unitree_rl_lab

# 训练命令
python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity --max_iterations 20000

# 测试命令
python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity --num_envs 16
```

**关键指标**:
- `episode_length` > 800 steps
- `reward/track_lin_vel_xy` > 0.5
- `reward/alive` 稳定在 0.15

**通过标准**: 机器人能稳定行走，跟踪速度命令

---

### 3.2 阶段2：带传感器行走（平地，有 height_scan）

**目标**: 学习使用 height_scan 传感器，为楼梯任务做准备

```powershell
# 从阶段1的模型继续训练（可选）
python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity-HeightScan --max_iterations 15000

# 或从头训练
python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity-HeightScan --max_iterations 20000

# 测试命令
python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity-HeightScan --num_envs 16
```

**关键指标**:
- 与阶段1相似的行走性能
- `height_scan` 观测值有合理变化（不是恒定值）

**通过标准**: 机器人能稳定行走，且 height_scan 观测正常工作

---

### 3.3 阶段3：盲爬楼梯（楼梯，无 height_scan）

**目标**: 学会在楼梯上行走，只依赖本体感知

```powershell
# 训练命令
python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Stair-Blind --max_iterations 30000

# 测试命令
python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Stair-Blind --num_envs 16
```

**关键指标**:
- `episode_length` > 500 steps
- `reward/upward_progress` > 0.3
- `reward/feet_clearance` > 0.2
- `terrain_level` 提升到 3+

**通过标准**: 机器人能在简单楼梯上行走，不频繁跌倒

---

### 3.4 阶段4：带传感器爬楼梯（楼梯，有 height_scan）

**目标**: 结合传感器信息，提升楼梯攀爬能力

```powershell
# 训练命令
python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Stair --max_iterations 50000

# 测试命令
python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Stair --num_envs 16
```

**关键指标**:
- `episode_length` > 800 steps
- `reward/upward_progress` > 0.5
- `terrain_level` 达到 6+

**通过标准**: 机器人能稳定攀爬中等难度楼梯

---

## 4. TensorBoard 监控

```powershell
# 新开终端，启动 TensorBoard
python -m tensorboard.main --logdir logs/rsl_rl

# 浏览器访问 http://localhost:6006
```

**关键曲线**:
- `Loss/value_function`: 应稳定下降
- `Policy/mean_noise_std`: 应逐渐下降（从 1.0 到 0.3-0.5）
- `Perf/total_fps`: 应稳定（4096 envs 约 50000-80000 fps）
- `reward/*`: 各奖励分量变化趋势

---

## 5. 恢复训练

```powershell
# 从最新 checkpoint 恢复
python scripts/rsl_rl/train.py --headless --task <TASK_NAME> --resume

# 从指定 run 恢复
python scripts/rsl_rl/train.py --headless --task <TASK_NAME> --resume --load_run 2025-12-01_10-30-00

# 从指定 checkpoint 恢复
python scripts/rsl_rl/train.py --headless --task <TASK_NAME> --resume --checkpoint model_5000.pt
```

---

## 6. 统一条件策略训练（推荐）

### 6.1 概述

统一条件策略让**一个网络同时学会4种模式**，部署时只需一个模型文件。

```
┌─────────────────────────────────────────────────────┐
│                    统一策略网络                      │
│  观测输入:                                          │
│  ├── 本体感知 (IMU, 关节)                           │
│  ├── 速度命令                                       │
│  ├── mode_flag [1,0,0,0] / [0,1,0,0] / ...         │ ← 模式标志
│  └── height_scan (盲模式时置零)                     │
│                        ↓                            │
│              ┌─────────────────┐                    │
│              │   共享 MLP 网络  │                    │
│              └─────────────────┘                    │
│                        ↓                            │
│              ┌─────────────────┐                    │
│              │    动作输出      │                    │
│              └─────────────────┘                    │
└─────────────────────────────────────────────────────┘
```

### 6.2 训练命令

```powershell
cd E:\Aunitree\unitree_rl_lab

# 统一策略训练（推荐）
python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Unified --max_iterations 50000

# 测试
python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Unified --num_envs 16
```

### 6.3 模式说明

| mode_flag | 模式 | 地形 | 传感器 | 说明 |
|-----------|------|------|--------|------|
| `[1,0,0,0]` | 0 | 平地 | 盲 | height_scan 置零 |
| `[0,1,0,0]` | 1 | 平地 | 有 | 使用真实 height_scan |
| `[0,0,1,0]` | 2 | 楼梯 | 盲 | height_scan 置零 |
| `[0,0,0,1]` | 3 | 楼梯 | 有 | 使用真实 height_scan |

### 6.4 部署优势

- **单模型部署**: 只需部署1个网络文件
- **智能切换**: 根据传感器状态和地形自动切换模式
- **安全降级**: 传感器失效时自动切换到盲模式

---

## 7. PPO 超参数配置

### 4.1 当前默认配置

```python
# unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py
@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24        # 每环境每迭代步数
    max_iterations = 50000        # 最大迭代次数
    save_interval = 100           # 保存间隔
    
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        learning_rate=1.0e-3,
        entropy_coef=0.01,
        gamma=0.99,
        lam=0.95,
        clip_param=0.2,
        num_learning_epochs=5,
        num_mini_batches=4,
    )
```

### 4.2 楼梯任务推荐调整

| 参数 | 默认值 | 楼梯任务推荐 | 说明 |
|------|--------|-------------|------|
| `num_steps_per_env` | 24 | 32 | 增加轨迹长度，更好学习长期行为 |
| `learning_rate` | 1e-3 | 5e-4 | 降低学习率，提高稳定性 |
| `entropy_coef` | 0.01 | 0.005 | 减少探索，加速收敛 |
| `gamma` | 0.99 | 0.995 | 更重视长期回报 |
| `init_noise_std` | 1.0 | 0.8 | 略微降低初始噪声 |

### 4.3 创建楼梯专用 PPO 配置（可选）

如需单独调整楼梯任务的 PPO 参数，可创建专用配置文件：

```python
# unitree_rl_lab/tasks/locomotion/agents/stair_ppo_cfg.py
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class StairPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 50000
    save_interval = 100

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        learning_rate=5.0e-4,
        entropy_coef=0.005,
        gamma=0.995,
        lam=0.95,
        clip_param=0.2,
        num_learning_epochs=5,
        num_mini_batches=4,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        desired_kl=0.01,
        max_grad_norm=1.0,
        schedule="adaptive",
    )
```

---

## 5. 奖励权重调优指南

### 5.1 当前奖励配置

| 奖励项 | 权重 | 类型 | 调优方向 |
|--------|------|------|---------|
| `track_lin_vel_xy` | 1.0 | 任务 | 核心，保持不变 |
| `track_ang_vel_z` | 0.5 | 任务 | 楼梯任务可降低 |
| `alive` | 0.15 | 存活 | 保持 |
| `upward_progress` | **1.5** | 楼梯 | 核心，可增加到 2.0 |
| `feet_clearance` | 1.2 | 步态 | 上楼梯关键 |
| `base_height_relative` | -5.0 | 姿态 | 过大会限制动作 |
| `flat_orientation_l2` | -3.0 | 姿态 | 楼梯上可降低到 -2.0 |
| `feet_slide` | -0.3 | 安全 | 楼梯上增大到 -0.5 |

### 5.2 常见问题与调优

#### 问题 1: 机器人不愿意前进
```python
# 增加前进奖励
track_lin_vel_xy.weight = 1.5
upward_progress.weight = 2.0
```

#### 问题 2: 机器人频繁摔倒
```python
# 增加存活和平衡奖励
alive.weight = 0.2
flat_orientation_l2.weight = -5.0
base_height_relative.weight = -8.0
```

#### 问题 3: 机器人抬腿不够高
```python
# 增加脚部离地奖励
feet_clearance.weight = 1.5
feet_clearance.params["target_height"] = 0.18  # 提高目标高度
```

#### 问题 4: 机器人脚步滑动
```python
# 增加滑动惩罚
feet_slide.weight = -0.5
gait.weight = 0.8
```

---

## 6. 测试与验证

### 6.1 Play 模式测试

```powershell
# 使用最新模型测试
E:\isaacenv\python.exe scripts/rsl_rl/play.py --task Unitree-G1-29dof-Stair

# 指定环境数量
E:\isaacenv\python.exe scripts/rsl_rl/play.py --task Unitree-G1-29dof-Stair --num_envs 16

# 使用指定 checkpoint
E:\isaacenv\python.exe scripts/rsl_rl/play.py --task Unitree-G1-29dof-Stair --checkpoint model_10000.pt
```

### 6.2 验证检查清单

| 检查项 | 通过标准 | 验证方法 |
|--------|---------|---------|
| 平地行走 | 稳定行走 30s+ | Play 模式观察 |
| 简单楼梯 (10cm) | 成功率 > 90% | 多次测试统计 |
| 中等楼梯 (15cm) | 成功率 > 70% | 多次测试统计 |
| 困难楼梯 (18cm) | 成功率 > 50% | 多次测试统计 |
| 下楼梯 | 能下 2-3 级 | Play 模式观察 |
| 抗干扰 | 轻推不倒 | 增加 push_robot 测试 |

### 6.3 录制演示视频

```powershell
# 录制 play 视频
E:\isaacenv\python.exe scripts/rsl_rl/play.py --task Unitree-G1-29dof-Stair --video --video_length 500
```

---

## 7. 常见问题排查

### 7.1 训练不收敛

**症状**: `reward` 长期不增长或震荡

**排查步骤**:
1. 检查 `episode_length`：如果很短，说明频繁终止
2. 检查 `termination/*`：找出主要终止原因
3. 检查 `reward/*`：确认各奖励分量是否合理

**解决方案**:
```python
# 1. 降低学习率
learning_rate = 1e-4

# 2. 增加 alive 奖励
alive.weight = 0.3

# 3. 减小惩罚项
flat_orientation_l2.weight = -1.0
```

### 7.2 显存不足 (OOM)

**症状**: CUDA out of memory

**解决方案**:
```powershell
# 减少环境数量
--num_envs 1024

# 减小 batch size（需修改配置）
num_mini_batches = 8  # 增加 mini-batch 数量
```

### 7.3 训练速度慢

**症状**: fps < 30000

**排查**:
1. 确认使用 `--headless` 模式
2. 检查 GPU 利用率：`nvidia-smi`
3. 确认没有其他 GPU 占用进程

### 7.4 height_scan 观测异常

**症状**: `height_scan` 值全部接近 1 或 -1

**排查**:
```python
# 检查 RayCaster z 偏移
height_scanner.offset.pos = (0.2, 0.0, 0.8)  # 应为 0.8，不是 20.0

# 开启可视化调试
height_scanner.debug_vis = True
```

---

## 8. 训练日志结构

```
logs/rsl_rl/unitree_g1_29dof_stair/
├── 2025-12-01_10-30-00/           # 训练 run 目录
│   ├── params/
│   │   ├── env.yaml               # 环境配置快照
│   │   ├── agent.yaml             # PPO 配置快照
│   │   └── stair_env_cfg.py       # 环境配置代码
│   ├── model_*.pt                 # 模型 checkpoint
│   ├── events.out.tfevents.*      # TensorBoard 日志
│   └── videos/                    # 录制的视频
└── 2025-12-02_14-00-00/           # 另一次训练
```

---

## 9. 训练时间估算

| 配置 | 环境数 | 迭代次数 | 预计时间 | 预计效果 |
|------|--------|---------|---------|---------|
| 快速验证 | 1024 | 5000 | 30 min | 基础平衡 |
| 标准训练 | 4096 | 30000 | 8-12 h | 中等楼梯 |
| 完整训练 | 4096 | 50000 | 16-24 h | 困难楼梯 |
| 精细调优 | 4096 | 80000 | 30+ h | 高鲁棒性 |

---

## 10. 下一步建议

### 10.1 训练完成后

1. **保存最佳模型**: 复制表现最好的 checkpoint
2. **导出部署配置**: 检查 `deploy_cfg.yaml` 是否正确生成
3. **Sim2Sim 测试**: 在 MuJoCo 中验证策略

### 10.2 进阶优化

1. **多阶段训练**: 先平地预训练 → 楼梯微调
2. **域随机化增强**: 增加更多物理参数随机化
3. **课程学习调优**: 调整难度提升速度

### 10.3 Sim2Real 准备

1. 确认控制频率与真机一致 (50Hz)
2. 检查观测空间与真机传感器匹配
3. 准备真机部署脚本

---

## 附录 A: 完整训练命令速查

```powershell
# ==================== 基础命令 ====================
# 标准训练
E:\isaacenv\python.exe scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Stair

# 快速验证（少量环境）
E:\isaacenv\python.exe scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Stair --num_envs 512 --max_iterations 2000

# 恢复训练
E:\isaacenv\python.exe scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Stair --resume

# ==================== 测试命令 ====================
# Play 测试
E:\isaacenv\python.exe scripts/rsl_rl/play.py --task Unitree-G1-29dof-Stair --num_envs 16

# 录制视频
E:\isaacenv\python.exe scripts/rsl_rl/play.py --task Unitree-G1-29dof-Stair --video

# ==================== 监控命令 ====================
# TensorBoard
E:\isaacenv\python.exe -m tensorboard.main --logdir logs/rsl_rl/unitree_g1_29dof_stair

# GPU 监控
nvidia-smi -l 1
```

---

**文档结束**

