# 训练中断恢复指南

**适用任务**: 所有 Unitree RL 训练任务  
**最后更新**: 2025-12-03

---

## 📋 目录

1. [快速恢复（3步）](#快速恢复3步)
2. [恢复方式详解](#恢复方式详解)
3. [常见场景](#常见场景)
4. [检查点选择策略](#检查点选择策略)
5. [注意事项](#注意事项)
6. [故障排除](#故障排除)
7. [最佳实践](#最佳实践)

---

## 🔄 快速恢复（3步）

### 步骤 1：找到训练日志目录

**PowerShell**:
```powershell
# 查看所有训练记录
ls logs/rsl_rl/Unitree-G1-29dof-Unified/
```

**输出示例**：
```
2025-12-03_21-30-45  ← 你的训练目录（时间戳格式）
2025-12-02_15-20-30
```

### 步骤 2：检查已保存的检查点

```powershell
# 查看某次训练的检查点
ls logs/rsl_rl/Unitree-G1-29dof-Unified/2025-12-03_21-30-45/
```

**输出示例**：
```
model_100.pt
model_200.pt
...
model_8000.pt   ← 假设训练在这里中断
model_8100.pt
config.yaml
events.out.tfevents.*
```

### 步骤 3：从检查点恢复训练

**推荐方法：自动加载最新检查点**

```powershell
# 单行命令（PowerShell）
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Unified --num_envs 4096 --headless --resume --load_run logs/rsl_rl/Unitree-G1-29dof-Unified/2025-12-03_21-30-45

# 多行格式（使用反引号）
python scripts/rsl_rl/train.py `
    --task Unitree-G1-29dof-Unified `
    --num_envs 4096 `
    --headless `
    --resume `
    --load_run logs/rsl_rl/Unitree-G1-29dof-Unified/2025-12-03_21-30-45
```

**Linux/Mac (Bash)**:
```bash
python scripts/rsl_rl/train.py \
    --task Unitree-G1-29dof-Unified \
    --num_envs 4096 \
    --headless \
    --resume \
    --load_run logs/rsl_rl/Unitree-G1-29dof-Unified/2025-12-03_21-30-45
```

---

## 🔧 恢复方式详解

### 自动保存机制

训练过程中会**自动保存检查点**：

```python
# 默认配置（在 rsl_rl_ppo_cfg.py 中）
save_interval = 100  # 每 100 iterations 保存一次
```

**保存内容**：
- ✅ 策略网络参数 (Actor)
- ✅ 价值网络参数 (Critic)
- ✅ 优化器状态 (Adam)
- ✅ 当前 iteration 数
- ✅ 观测归一化参数（running mean/std）
- ✅ 训练配置信息

**保存位置**：
```
logs/rsl_rl/[任务名]/[时间戳]/
├── model_100.pt      # 第 100 次迭代
├── model_200.pt      # 第 200 次迭代
├── model_300.pt      # ...
└── config.yaml       # 训练配置
```

---

### 方式 1：`--resume` + `--load_run`（✅ 推荐）

**命令格式**：
```powershell
python scripts/rsl_rl/train.py `
    --task [任务名] `
    --resume `
    --load_run [训练目录路径] `
    [其他参数...]
```

**特点**：
| 特性 | 说明 |
|------|------|
| ✅ **自动加载** | 自动找到目录中最新的检查点 |
| ✅ **继续计数** | 从中断的 iteration 继续（如从 8100 继续） |
| ✅ **保持配置** | 继承原有超参数配置 |
| ✅ **归一化延续** | 保持观测归一化统计信息 |
| ✅ **同一实验** | TensorBoard 曲线平滑连接 |

**示例**：
```powershell
# 原训练在 8100 iteration 中断，目标是 30000
python scripts/rsl_rl/train.py `
    --task Unitree-G1-29dof-Unified `
    --resume `
    --load_run logs/rsl_rl/Unitree-G1-29dof-Unified/2025-12-03_21-30-45 `
    --num_envs 4096 `
    --headless

# 结果：从 8100 继续训练到 30000
```

---

### 方式 2：`--checkpoint`（指定检查点）

**命令格式**：
```powershell
python scripts/rsl_rl/train.py `
    --task [任务名] `
    --checkpoint [具体检查点文件路径] `
    --max_iterations [新的迭代次数] `
    [其他参数...]
```

**特点**：
| 特性 | 说明 |
|------|------|
| ✅ **精确控制** | 可以选择任意检查点作为起点 |
| ⚠️ **重新计数** | Iteration 从 1 重新开始 |
| ⚠️ **新实验** | TensorBoard 中显示为新的训练曲线 |
| ✅ **灵活调整** | 可以修改训练配置 |

**适用场景**：
- 想从某个特定的好检查点重新开始
- 需要回退到更早的稳定状态
- 想改变训练超参数

**示例**：
```powershell
# 从 iteration 5000 的检查点重新开始
python scripts/rsl_rl/train.py `
    --task Unitree-G1-29dof-Unified `
    --checkpoint logs/rsl_rl/Unitree-G1-29dof-Unified/2025-12-03_21-30-45/model_5000.pt `
    --max_iterations 40000 `
    --num_envs 4096 `
    --headless

# 结果：加载 5000 的权重，但从 iteration 1 开始计数
```

---

### 方式对比表

| 对比项 | `--resume --load_run` | `--checkpoint` |
|--------|-----------------------|----------------|
| **命令复杂度** | 简单（目录路径） | 需要完整文件路径 |
| **Iteration 计数** | 继续原有计数 | 从 1 重新开始 |
| **TensorBoard** | 曲线连续 | 新的曲线 |
| **配置继承** | 自动继承 | 可以修改 |
| **典型用途** | 意外中断恢复 | 精细控制、回退 |
| **推荐度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

---

## 🎯 常见场景

### 场景 1：意外中断，继续训练 ⭐ 最常用

**情况**：训练在 8000 iteration 时崩溃或被手动停止，原计划训练 30000 iterations

**解决方案**：
```powershell
python scripts/rsl_rl/train.py `
    --task Unitree-G1-29dof-Unified `
    --resume `
    --load_run logs/rsl_rl/Unitree-G1-29dof-Unified/2025-12-03_21-30-45 `
    --num_envs 4096 `
    --headless
```

**结果**：
- 从最近的检查点（如 8100）继续
- 自动训练到原定的 30000 iterations
- TensorBoard 曲线无缝连接

---

### 场景 2：延长训练时间

**情况**：原计划 30000 iterations 已完成，但想继续训练到 60000

**解决方案**：
```powershell
python scripts/rsl_rl/train.py `
    --task Unitree-G1-29dof-Unified `
    --resume `
    --load_run logs/rsl_rl/Unitree-G1-29dof-Unified/2025-12-03_21-30-45 `
    --max_iterations 60000 `
    --num_envs 4096 `
    --headless
```

**结果**：
- 从 30000 继续训练到 60000
- 保持原有学习率调度和配置

---

### 场景 3：性能回退，从好的检查点重新训练

**情况**：
- Iteration 15000 时性能最好（奖励 200）
- Iteration 20000 后性能下降（奖励降到 150）
- 想从 15000 重新开始

**解决方案**：
```powershell
# 方法 A：继续原有 iteration 计数
python scripts/rsl_rl/train.py `
    --task Unitree-G1-29dof-Unified `
    --checkpoint logs/rsl_rl/Unitree-G1-29dof-Unified/2025-12-03_21-30-45/model_15000.pt `
    --max_iterations 50000 `
    --num_envs 4096 `
    --headless

# 方法 B：同时降低学习率（防止再次崩溃）
# 先修改 rsl_rl_ppo_cfg.py: learning_rate = 5e-4
# 然后运行上面的命令
```

**注意**：此时 iteration 会从 1 重新计数，但网络权重是从 15000 加载的

---

### 场景 4：训练崩溃，回退到稳定检查点

**情况**：
- Iteration 12000 后训练崩溃（奖励暴跌、NaN 出现）
- 需要回退到 10000 的稳定状态

**解决方案**：
```powershell
# 1. 回退到稳定检查点
python scripts/rsl_rl/train.py `
    --task Unitree-G1-29dof-Unified `
    --checkpoint logs/rsl_rl/Unitree-G1-29dof-Unified/2025-12-03_21-30-45/model_10000.pt `
    --max_iterations 30000 `
    --num_envs 4096 `
    --headless

# 2. 同时调整超参数防止再次崩溃
# 修改 rsl_rl_ppo_cfg.py:
#   learning_rate = 5e-4  (从 1e-3 降低)
#   clip_param = 0.15     (从 0.2 降低)
```

---

### 场景 5：切换硬件，调整环境数量

**情况**：原来在 RTX 3090 上训练（4096 envs），现在换到 RTX 3060（显存不足）

**解决方案**：
```powershell
# 降低环境数量，其他保持不变
python scripts/rsl_rl/train.py `
    --task Unitree-G1-29dof-Unified `
    --resume `
    --load_run logs/rsl_rl/Unitree-G1-29dof-Unified/2025-12-03_21-30-45 `
    --num_envs 2048 `
    --headless
```

**说明**：
- ✅ 可以改变 `--num_envs`
- ✅ 训练会正常继续
- ⚠️ 训练速度会相应降低

---

## 🔍 检查点选择策略

### 方法 1：查看 TensorBoard 确定最佳检查点

```powershell
# 启动 TensorBoard
tensorboard --logdir=logs/rsl_rl/Unitree-G1-29dof-Unified/2025-12-03_21-30-45
```

**关键指标**：

| 指标 | 位置 | 评判标准 |
|------|------|----------|
| **Mean Reward** | `Policy/mean_reward` | 越高越好 |
| **Episode Length** | `Policy/mean_episode_length` | 越长越稳定 |
| **Success Rate** | `Success/...` | 越高越好 |
| **Policy Loss** | `Loss/surrogate` | 平稳，不震荡 |
| **Value Loss** | `Loss/value_function` | 逐渐下降 |

**选择原则**：
1. ✅ 奖励高且稳定的点（不是峰值）
2. ✅ Episode 长度接近最大值的点
3. ✅ Loss 平稳收敛的点
4. ⚠️ 避免选择震荡剧烈的点

---

### 方法 2：使用 Play 模式测试检查点

**测试多个检查点**：

```powershell
# 测试 iteration 10000
python scripts/rsl_rl/play.py `
    --task Unitree-G1-29dof-Unified `
    --num_envs 16 `
    --checkpoint logs/rsl_rl/Unitree-G1-29dof-Unified/2025-12-03_21-30-45/model_10000.pt

# 观察：平地速度、楼梯成功率、动作平滑度

# 测试 iteration 15000
python scripts/rsl_rl/play.py `
    --task Unitree-G1-29dof-Unified `
    --num_envs 16 `
    --checkpoint logs/rsl_rl/Unitree-G1-29dof-Unified/2025-12-03_21-30-45/model_15000.pt

# 对比性能，选择最好的
```

**评估标准**：

| 性能指标 | 优秀标准 | 测试方法 |
|----------|----------|----------|
| 平地速度 | > 1.0 m/s | 观察机器人前进速度 |
| 楼梯成功率 | > 80% | 计算成功攀爬次数 |
| 动作平滑度 | 无明显抖动 | 视觉观察 |
| 摔倒频率 | < 5% | 统计 episode 长度 |

---

### 方法 3：自动选择脚本（高级）

```python
# evaluate_checkpoints.py
import torch
import numpy as np

def evaluate_checkpoint(checkpoint_path, num_episodes=50):
    """评估单个检查点的性能"""
    # 加载检查点并运行评估
    rewards = []
    episode_lengths = []
    
    for _ in range(num_episodes):
        # 运行一个 episode
        reward, length = run_episode(checkpoint_path)
        rewards.append(reward)
        episode_lengths.append(length)
    
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(episode_lengths),
        'success_rate': np.sum(np.array(episode_lengths) > 800) / num_episodes
    }

# 评估所有检查点
checkpoints = [5000, 10000, 15000, 20000, 25000, 30000]
results = {}

for ckpt in checkpoints:
    path = f"logs/.../model_{ckpt}.pt"
    results[ckpt] = evaluate_checkpoint(path)
    print(f"Checkpoint {ckpt}: {results[ckpt]}")

# 选择最佳检查点
best_ckpt = max(results, key=lambda k: results[k]['mean_reward'])
print(f"Best checkpoint: {best_ckpt}")
```

---

## ⚠️ 注意事项

### ✅ 可以做的操作

| 操作 | 说明 | 示例 |
|------|------|------|
| ✅ **改变环境数量** | 根据硬件调整 | `--num_envs 2048` |
| ✅ **延长训练** | 增加 max_iterations | `--max_iterations 60000` |
| ✅ **切换显卡** | 只要显存够用 | - |
| ✅ **添加可视化** | 调试时开启 | `--enable_cameras` |
| ✅ **更换实验名** | 创建新的日志目录 | `--experiment_name new_exp` |

---

### ❌ 不建议/不能做的操作

| 操作 | 问题 | 结果 |
|------|------|------|
| ❌ **改变任务类型** | 观测/动作维度不匹配 | 加载失败 |
| ❌ **修改网络架构** | 权重形状不一致 | 崩溃 |
| ❌ **删除 config.yaml** | 无法读取原始配置 | 配置丢失 |
| ❌ **手动编辑 .pt 文件** | 破坏模型结构 | 无法加载 |
| ❌ **混用不同任务检查点** | 维度不匹配 | 错误 |

---

### ⚠️ 需要小心的操作

#### 1. 恢复时修改超参数

**场景**：想降低学习率防止崩溃

**方法**：
```python
# 1. 修改源码配置文件
# 编辑: source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py

@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    algorithm = RslRlPpoAlgorithmCfg(
        learning_rate=5e-4,  # 🔧 从 1e-3 改为 5e-4
        # 其他参数...
    )
```

```powershell
# 2. 然后恢复训练（新配置会生效）
python scripts/rsl_rl/train.py `
    --resume `
    --load_run logs/rsl_rl/Unitree-G1-29dof-Unified/2025-12-03_21-30-45 `
    --num_envs 4096 `
    --headless
```

**警告**：
- ⚠️ 改变学习率可能导致训练不稳定
- ⚠️ 建议先小范围测试（如 1000 iterations）
- ⚠️ 最好创建新的实验名，避免覆盖原始数据

---

#### 2. 跨版本恢复

**场景**：更新了 Isaac Lab 或代码库版本

**风险**：
- ⚠️ API 可能变化
- ⚠️ 检查点格式可能不兼容

**建议**：
```powershell
# 1. 先备份重要检查点
cp -r logs/rsl_rl/Unitree-G1-29dof-Unified/2025-12-03_21-30-45 backup/

# 2. 尝试加载
python scripts/rsl_rl/train.py --resume --load_run backup/2025-12-03_21-30-45 --max_iterations 100

# 3. 如果失败，从头训练
```

---

## 🛠️ 故障排除

### 问题 1：找不到检查点文件

**症状**：
```
Error: No checkpoint found in logs/rsl_rl/...
```

**诊断**：
```powershell
# 检查目录是否存在
ls logs/rsl_rl/Unitree-G1-29dof-Unified/
```

**可能原因**：
1. ❌ 训练时间太短（< save_interval）
2. ❌ 训练从未成功保存
3. ❌ 路径错误（时间戳不对）

**解决方案**：
```powershell
# 1. 确认正确的时间戳
ls logs/rsl_rl/Unitree-G1-29dof-Unified/ | Sort-Object -Descending | Select-Object -First 1

# 2. 检查是否有 .pt 文件
ls logs/rsl_rl/Unitree-G1-29dof-Unified/[时间戳]/*.pt

# 3. 如果确实没有，只能重新训练
```

---

### 问题 2：加载检查点后立即崩溃

**症状**：
```
RuntimeError: Error(s) in loading state_dict...
或
CUDA out of memory
```

**可能原因**：

| 原因 | 诊断方法 | 解决方案 |
|------|----------|----------|
| **网络配置改变** | 检查是否修改过网络架构 | 使用原始配置或重新训练 |
| **检查点损坏** | 文件大小异常或无法读取 | 使用更早的检查点 |
| **显存不足** | `nvidia-smi` 查看显存 | 降低 `--num_envs` |
| **版本不兼容** | 更新了代码库 | 回退版本或重新训练 |

**解决步骤**：
```powershell
# 1. 尝试更早的检查点
python scripts/rsl_rl/train.py `
    --checkpoint logs/.../model_5000.pt `
    --num_envs 4096 `
    --headless

# 2. 如果仍失败，降低环境数量
--num_envs 2048

# 3. 最后手段：从头训练
python scripts/rsl_rl/train.py `
    --task Unitree-G1-29dof-Unified `
    --num_envs 4096 `
    --headless
```

---

### 问题 3：恢复后性能突然下降

**症状**：
- 恢复前奖励 150
- 恢复后奖励降到 80

**可能原因**：

1. **域随机化参数改变**
   ```python
   # 检查是否修改了：
   # - 摩擦系数范围
   # - 质量随机化
   # - 外力扰动
   ```

2. **学习率调度问题**
   ```python
   # 使用 --checkpoint 会重置学习率调度
   # 解决：使用 --resume --load_run
   ```

3. **观测归一化重置**
   ```python
   # --checkpoint 不会加载归一化参数
   # 解决：使用 --resume --load_run
   ```

**解决方案**：
```powershell
# 确保使用 --resume 而不是 --checkpoint
python scripts/rsl_rl/train.py `
    --task Unitree-G1-29dof-Unified `
    --resume `
    --load_run logs/rsl_rl/Unitree-G1-29dof-Unified/2025-12-03_21-30-45 `
    --num_envs 4096 `
    --headless
```

---

### 问题 4：TensorBoard 曲线不连续

**症状**：恢复训练后，TensorBoard 中曲线从头开始

**原因**：使用了 `--checkpoint` 而不是 `--resume`

**解决方案**：
```powershell
# 使用 --resume 保证曲线连续
python scripts/rsl_rl/train.py `
    --resume `
    --load_run logs/.../2025-12-03_21-30-45 `
    --num_envs 4096 `
    --headless
```

---

### 问题 5：多次恢复后日志混乱

**症状**：同一个目录下有多个训练的混合数据

**预防方法**：
```powershell
# 方法 1：每次恢复使用相同命令（推荐）
python scripts/rsl_rl/train.py --resume --load_run logs/.../original_run

# 方法 2：创建新的实验分支
python scripts/rsl_rl/train.py `
    --checkpoint logs/.../model_10000.pt `
    --experiment_name exp_v2_from_10k `
    --max_iterations 30000
```

---

## 📊 训练状态监控

### 恢复成功的标志

**终端输出示例**（使用 `--resume`）：
```
[INFO] Loading checkpoint from: logs/.../model_8100.pt
[INFO] Resuming training from iteration 8100
[INFO] Target iterations: 30000
[INFO] Estimated remaining time: 3.2 hours

Iteration: 8200
  Mean Reward: 145.2
  Mean Episode Length: 875.3
  FPS: 12453
  Policy Loss: 0.042
  Value Loss: 0.234
  Learning Rate: 0.000850
```

**关键确认点**：
- ✅ "Resuming from iteration X" 而不是 "Starting training"
- ✅ 初始 Mean Reward 接近中断前的值
- ✅ Learning Rate 已经经过调度（不是初始的 1e-3）

---

### TensorBoard 监控

**正常情况**（使用 `--resume`）：
```
Policy/mean_reward 曲线：
  ^
  |     /----------- 恢复点，曲线平滑连接
  |    /
  |   /
  +---+---+---+---+---+---+> Iteration
  0   5k  10k 15k 20k 25k 30k
```

**异常情况**（使用 `--checkpoint` 或配置改变）：
```
Policy/mean_reward 曲线：
  ^
  |                  /---- 新曲线，从头开始
  |     /---+       /
  |    /    |      /
  |   /     ×     /
  +---+---+---+---+---+---+> Iteration
  0   5k  10k 0   5k  10k
        原训练   新训练
```

---

## 💾 备份和检查点管理

### 最佳实践

#### 1. 保留重要检查点

```powershell
# 备份关键检查点
mkdir backup
cp logs/rsl_rl/Unitree-G1-29dof-Unified/2025-12-03_21-30-45/model_15000.pt backup/unified_best_15k.pt
cp logs/rsl_rl/Unitree-G1-29dof-Unified/2025-12-03_21-30-45/config.yaml backup/unified_best_config.yaml
```

#### 2. 定期清理旧检查点

```powershell
# 只保留每 1000 iterations 的检查点
# 删除中间的检查点以节省空间

# PowerShell 脚本示例
$checkpoint_dir = "logs/rsl_rl/Unitree-G1-29dof-Unified/2025-12-03_21-30-45"
Get-ChildItem "$checkpoint_dir\model_*.pt" | Where-Object {
    $_.Name -match "model_(\d+)\.pt"
    $iter = [int]$Matches[1]
    $iter % 1000 -ne 0  # 不是 1000 的倍数
} | Remove-Item

# 保留 model_1000.pt, model_2000.pt, ... 删除 model_100.pt, model_200.pt, ...
```

#### 3. 跨机器迁移

```powershell
# 打包整个训练目录
tar -czf unified_training_20251203.tar.gz logs/rsl_rl/Unitree-G1-29dof-Unified/2025-12-03_21-30-45

# 在新机器上解压
tar -xzf unified_training_20251203.tar.gz

# 恢复训练
python scripts/rsl_rl/train.py --resume --load_run logs/rsl_rl/Unitree-G1-29dof-Unified/2025-12-03_21-30-45
```

---

## 🎓 最佳实践总结

### ✅ 推荐做法

1. **使用 `--resume --load_run`**
   - 最安全、最可靠的恢复方式
   - 保持训练连续性

2. **定期查看 TensorBoard**
   - 及时发现训练问题
   - 选择最佳检查点

3. **备份重要检查点**
   - 性能好的检查点单独保存
   - 防止意外覆盖

4. **记录训练配置**
   - 每次训练记录超参数
   - 便于复现和对比

5. **测试后再继续**
   - 恢复训练前先用 play 模式测试
   - 确认检查点质量

---

### ⚠️ 避免做法

1. ❌ **频繁修改配置后恢复**
   - 可能导致训练不稳定
   - 建议创建新实验

2. ❌ **盲目选择最新检查点**
   - 最新不一定最好
   - 先查看 TensorBoard

3. ❌ **删除中间检查点**
   - 可能需要回退
   - 至少保留几个关键点

4. ❌ **混用不同任务的检查点**
   - 会导致维度错误
   - 确保任务匹配

5. ❌ **忽略终端警告**
   - 警告可能预示问题
   - 及时检查日志

---

## 📋 快速参考卡片

```
┌─────────────────────────────────────────────────────────┐
│          训练恢复命令快速参考                            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│ 🔄 继续中断的训练（推荐）                                │
│ python scripts/rsl_rl/train.py `                         │
│     --task Unitree-G1-29dof-Unified `                    │
│     --resume `                                           │
│     --load_run logs/rsl_rl/[任务名]/[时间戳] `          │
│     --num_envs 4096 `                                    │
│     --headless                                           │
│                                                          │
│ ⏱️ 延长训练时间                                          │
│ ... (同上) `                                             │
│     --max_iterations 60000                               │
│                                                          │
│ 🔙 从特定检查点重新开始                                  │
│ python scripts/rsl_rl/train.py `                         │
│     --task Unitree-G1-29dof-Unified `                    │
│     --checkpoint logs/.../model_15000.pt `               │
│     --max_iterations 40000 `                             │
│     --num_envs 4096 `                                    │
│     --headless                                           │
│                                                          │
│ 🧪 测试检查点性能                                        │
│ python scripts/rsl_rl/play.py `                          │
│     --task Unitree-G1-29dof-Unified `                    │
│     --num_envs 16 `                                      │
│     --checkpoint logs/.../model_15000.pt                 │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 📚 相关文档

- **快速开始**: `unified_quick_start.md`
- **完整训练指南**: `unified_training_guide.md`
- **调参指南**: `unified_hyperparameter_tuning.md`
- **楼梯训练**: `stair_training_guide.md`

---

**记住**：当训练中断时，不要慌张！使用 `--resume --load_run` 就能轻松恢复 ✨
