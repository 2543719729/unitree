# Unitree G1 Unified 策略快速开始指南

**5分钟快速上手** | 从零到运行训练

---

## 🚀 快速开始（3步）

### 步骤 1：环境检查

**Linux/Mac (Bash)**:
```bash
# 确保在 unitree_rl_lab 目录
cd /path/to/unitree_rl_lab

# 激活环境
conda activate isaacenv

# 验证环境
python scripts/list_envs.py | grep "Unitree-G1-29dof-Unified"
```

**Windows (PowerShell)**:
```powershell
# 确保在 unitree_rl_lab 目录
cd e:\Aunitree\unitree_rl_lab

# 激活环境
conda activate isaacenv

# 验证环境
python scripts/list_envs.py | Select-String "Unitree-G1-29dof-Unified"
```

**预期输出**：
```
✓ Unitree-G1-29dof-Unified
```

### 步骤 2：启动训练（标准配置）

**Linux/Mac (Bash)**:
```bash
# 一键启动训练（使用默认参数）
python scripts/rsl_rl/train.py \
    --task Unitree-G1-29dof-Unified \
    --num_envs 4096 \
    --headless \
    --max_iterations 30000
```

**Windows (PowerShell)** - 推荐单行命令:
```powershell
# 单行命令（推荐）
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Unified --num_envs 4096 --headless --max_iterations 30000

# 或使用反引号 ` 续行
python scripts/rsl_rl/train.py `
    --task Unitree-G1-29dof-Unified `
    --num_envs 4096 `
    --headless `
    --max_iterations 30000
```

> ⚠️ **PowerShell 注意**: 不要使用反斜杠 `\`，要用反引号 `` ` `` 或直接写成单行

**训练将开始！** 预计 4-6 小时完成（RTX 3090）

### 步骤 3：监控训练

```bash
# 新开一个终端，启动 TensorBoard
tensorboard --logdir=logs/rsl_rl
```

浏览器打开：`http://localhost:6006`

---

## 📊 关键指标速查

**训练正常的标志**：

| Iteration | Mean Reward | Episode Length | 速度 (m/s) | 楼梯成功率 |
|-----------|-------------|----------------|------------|------------|
| 1000 | 20-50 | 200-400 | - | - |
| 5000 | 80-120 | 500-700 | 0.5+ | 30%+ |
| 10000 | 120-160 | 700-850 | 0.8+ | 60%+ |
| 20000 | 160-200 | 850-950 | 1.0+ | 80%+ |
| 30000 | 200+ | 950+ | 1.2+ | 85%+ |

---

## ⚙️ 常用配置变体

### 🖥️ 低显存配置 (4GB 显存)

```bash
python scripts/rsl_rl/train.py \
    --task Unitree-G1-29dof-Unified \
    --num_envs 1024 \        # 减少到 1024
    --headless \
    --max_iterations 30000
```

### 🐛 调试配置 (带可视化)

```bash
python scripts/rsl_rl/train.py \
    --task Unitree-G1-29dof-Unified \
    --num_envs 64 \           # 小规模
    --max_iterations 1000 \   # 短时间
    --enable_cameras          # 开启可视化
```

### 🔄 从检查点恢复

```bash
python scripts/rsl_rl/train.py \
    --task Unitree-G1-29dof-Unified \
    --resume \
    --load_run logs/rsl_rl/Unitree-G1-29dof-Unified/YYYY-MM-DD_HH-MM-SS
```

---

## 🎯 测试训练好的策略

```bash
# 评估模式（Play）
python scripts/rsl_rl/play.py \
    --task Unitree-G1-29dof-Unified \
    --num_envs 16 \
    --checkpoint logs/rsl_rl/.../model_30000.pt \
    --enable_cameras
```

---

## ❓ 常见问题快速解决

### Q: CUDA out of memory?
**A**: 降低 `--num_envs`
```bash
--num_envs 2048  # 或 1024, 512
```

### Q: 训练速度很慢 (FPS < 5000)?
**A**: 
1. 检查 `--headless` 是否启用
2. 降低环境数量
3. 关闭其他 GPU 程序

### Q: 机器人频繁摔倒 (Episode < 200)?
**A**: 检查配置，或查看完整指南：
```bash
# 查看详细文档
cat docs/unified_training_guide.md
```

### Q: TensorBoard 显示奖励为负数?
**A**: 正常！前 1000-3000 iterations 奖励可能为负，之后会上升。

---

## 📁 关键文件位置

```
unitree_rl_lab/
├── scripts/rsl_rl/
│   ├── train.py              # 训练脚本
│   └── play.py               # 评估脚本
│
├── source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/
│   ├── agents/
│   │   └── rsl_rl_ppo_cfg.py      # PPO 参数配置 🔧
│   └── robots/g1/29dof/
│       └── unified_env_cfg.py     # 环境配置 🔧
│
└── logs/rsl_rl/
    └── Unitree-G1-29dof-Unified/  # 训练日志和模型
        └── YYYY-MM-DD_HH-MM-SS/
            ├── model_*.pt         # 模型检查点
            └── events.out.tfevents.*  # TensorBoard 数据
```

---

## 🛠️ 基础调参示例

### 修改训练迭代次数

编辑 `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py`:

```python
max_iterations = 50000  # 从 30000 改为 50000
```

### 调整奖励权重

编辑 `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/unified_env_cfg.py`:

```python
# 找到 RewardsCfg 类
@configclass
class RewardsCfg:
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,  # 🔧 从 1.5 提高到 2.0（加速）
        params={"std": 0.5}
    )
```

---

## 📚 进阶学习路径

1. ✅ **快速开始** ← 你在这里
2. 📖 **完整训练指南** → `unified_training_guide.md`
3. 🔧 **调参实战** → `unified_hyperparameter_tuning.md`
4. 🏔️ **楼梯专项** → `stair_training_guide.md`

---

## 🎓 核心概念 30 秒速览

**Unified 策略**：一个网络掌握 4 种模式
- 模式 0：平地盲走
- 模式 1：平地+传感器
- 模式 2：楼梯盲爬
- 模式 3：楼梯+传感器

**关键机制**：
- `mode_flag`: 告诉网络当前模式
- `conditional_height_scan`: 盲模式时为 0

**训练流程**：
```
启动训练 → 监控 TensorBoard → 等待收敛 → 评估性能 → 部署
```

---

## ✨ 下一步

**训练进行中？** 
→ 打开 TensorBoard 监控进度

**训练完成？**
→ 运行 `play.py` 测试策略

**想要调优？**
→ 阅读 `unified_hyperparameter_tuning.md`

**遇到问题？**
→ 查看 `unified_training_guide.md` 第 9 章：常见问题

---

**祝训练顺利！🎉**

*提示：首次训练建议使用默认参数，不要急于修改配置*
