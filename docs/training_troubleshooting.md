# 训练故障排除指南

**最后更新**: 2025-12-04

---

## 🚨 常见训练问题和解决方案

### 问题 1: 训练长时间无进展（奖励停滞）

**症状**:
- 训练数小时，Mean Reward 几乎不变
- Episode Length 始终很短（< 10 步）
- 机器人一直摔倒，无法学会行走

**可能原因 1: 速度命令配置错误** ⭐⭐⭐

**诊断方法**:
```python
# 检查配置文件中的速度命令范围
# 文件: source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/unified_env_cfg.py

# ❌ 错误配置（会导致训练无进展）
ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
    lin_vel_x=(0.0, 0.0),  # 目标速度始终为 0
    lin_vel_y=(0.0, 0.0),
    ang_vel_z=(0.0, 0.0),
)
```

**解决方案**:
```python
# ✅ 正确配置
ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
    lin_vel_x=(-0.5, 1.0),  # 允许前后移动
    lin_vel_y=(-0.3, 0.3),  # 允许侧向移动
    ang_vel_z=(-0.5, 0.5),  # 允许转向
)
```

**操作步骤**:
1. 停止当前训练（Ctrl+C）
2. 修改配置文件
3. 重新从头开始训练（不要使用 --resume）
4. 监控 `track_lin_vel_xy` 奖励是否开始增长

**预期效果**:
- Iteration 500: Mean Reward > 0.5
- Iteration 1000: Episode Length > 200
- Iteration 3000: Mean Reward > 20

---

**可能原因 2: 学习率过高或过低**

**诊断方法**:
```
查看终端输出：
- Policy Loss 剧烈震荡 → 学习率太高
- Policy Loss 几乎不变 → 学习率太低
```

**解决方案**:
```python
# 修改文件: agents/rsl_rl_ppo_cfg.py
algorithm = RslRlPpoAlgorithmCfg(
    learning_rate=5e-4,  # 从 1e-3 降低到 5e-4
    # 或
    learning_rate=3e-3,  # 从 1e-3 提高到 3e-3
)
```

---

**可能原因 3: 奖励函数权重不平衡**

**诊断方法**:
```
查看 TensorBoard:
- 某个奖励项支配了总奖励
- 正奖励总和 << 负奖励总和（总奖励一直是负数）
```

**解决方案**:
参考 `unified_hyperparameter_tuning.md` 第 2 章进行奖励权重调整。

---

### 问题 2: 训练崩溃（NaN 或 Inf）

**症状**:
- 训练突然停止
- 终端显示 NaN 或 Inf
- Mean Reward 突然暴跌

**可能原因**:
1. 学习率太高
2. 梯度爆炸
3. 奖励函数计算错误

**解决方案**:

**步骤 1: 降低学习率**
```python
learning_rate=5e-4  # 从 1e-3 降低
```

**步骤 2: 检查梯度裁剪**
```python
max_grad_norm=1.0  # 确保启用梯度裁剪
```

**步骤 3: 检查奖励函数**
```python
# 确保奖励不会产生极端值
# 避免除以零或对负数取 log
```

**步骤 4: 回退到早期检查点**
```powershell
# 回退到崩溃前的稳定检查点
python scripts/rsl_rl/train.py --checkpoint logs/.../model_5000.pt --num_envs 4096 --headless
```

---

### 问题 3: TensorBoard 显示"No dashboards"

**症状**:
- 打开 TensorBoard 显示"No dashboards are active"
- 找不到训练数据

**原因**:
- TensorBoard 路径不对
- 训练还没开始保存数据

**解决方案**:

**方法 1: 切换到正确目录**
```powershell
# 停止 TensorBoard (Ctrl+C)
cd unitree_rl_lab
tensorboard --logdir=logs/rsl_rl
```

**方法 2: 使用完整路径**
```powershell
tensorboard --logdir=e:\Aunitree\unitree_rl_lab\logs\rsl_rl
```

**方法 3: 确认训练已经开始**
```powershell
# 检查日志目录是否存在
ls unitree_rl_lab\logs\rsl_rl\Unitree-G1-29dof-Unified\
```

---

### 问题 4: 显存不足（CUDA out of memory）

**症状**:
```
RuntimeError: CUDA out of memory
```

**解决方案**:

**方法 1: 降低环境数量**
```powershell
# 从 4096 降低到 2048 或 1024
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Unified --num_envs 2048 --headless
```

**方法 2: 降低网络大小**
```python
# 修改 agents/rsl_rl_ppo_cfg.py
policy = RslRlPpoActorCriticCfg(
    actor_hidden_dims=[256, 128, 64],  # 从 [512, 256, 128] 降低
    critic_hidden_dims=[256, 128, 64],
)
```

**方法 3: 清理显存**
```powershell
# 停止所有训练进程
# 重启终端
nvidia-smi  # 检查显存使用情况
```

---

### 问题 5: 性能退化（训练后期奖励下降）

**症状**:
- 前期训练正常，奖励持续上升
- 到某个 iteration 后，奖励开始下降
- 策略变得不稳定

**可能原因**:
1. 学习率在后期仍然太高
2. 域随机化过强
3. 过拟合到某种行为

**解决方案**:

**方法 1: 使用学习率衰减**
```python
schedule="adaptive",  # 使用自适应学习率
desired_kl=0.01,      # KL 散度约束
```

**方法 2: 回退到最佳检查点并降低学习率**
```powershell
# 1. 找到最佳检查点（通过 TensorBoard 或 play 模式）
# 2. 修改配置降低学习率
# 3. 从该检查点继续训练

python scripts/rsl_rl/train.py --checkpoint logs/.../model_15000.pt --num_envs 4096 --headless
```

**方法 3: 调整域随机化强度**
```python
# 在 unified_env_cfg.py 中降低随机化范围
mass_distribution_params=(-0.5, 1.5),  # 从 (-1.0, 3.0) 降低
```

---

### 问题 6: 训练速度很慢

**症状**:
- FPS < 3000
- 每个 iteration 耗时 > 30 秒
- 预计训练时间 > 48 小时

**解决方案**:

**方法 1: 检查 GPU 利用率**
```powershell
nvidia-smi  # 查看 GPU 使用率
# 如果 GPU 利用率 < 80%，可能是 CPU 瓶颈
```

**方法 2: 调整并行度**
```python
# 修改 agents/rsl_rl_ppo_cfg.py
num_steps_per_env = 24,      # 增加到 32 或 48
num_mini_batches = 4,        # 降低到 2（减少计算）
```

**方法 3: 禁用调试功能**
```python
debug_vis=False,             # 关闭可视化
empirical_normalization=False,  # 关闭额外的归一化计算
```

---

## 📊 训练健康检查清单

### 每次启动训练前

- [ ] 速度命令范围是否合理（不是全零）
- [ ] 奖励函数权重是否平衡
- [ ] 环境数量是否适合 GPU 显存
- [ ] 实验名称是否正确（避免覆盖）
- [ ] TensorBoard 是否已启动

### 训练开始后 30 分钟（~500 iterations）

- [ ] Mean Reward 是否开始上升（> 0）
- [ ] Episode Length 是否增加（> 50）
- [ ] 无 NaN 或 Inf
- [ ] FPS 稳定（4000-6000）
- [ ] GPU 利用率 > 80%

### 训练 3 小时后（~3000 iterations）

- [ ] Mean Reward > 10
- [ ] Episode Length > 500
- [ ] 各奖励项趋势正常
- [ ] 无异常崩溃或性能退化

### 训练完成前（最后 1000 iterations）

- [ ] Mean Reward 稳定在高位（> 100）
- [ ] Episode Length 接近最大值（> 900）
- [ ] 测试策略表现良好

---

## 🔧 快速诊断流程图

```
训练有问题？
   ↓
查看 Mean Reward
   ↓
┌──────────────┴──────────────┐
│                              │
一直是负数？              开始正常，后来下降？
│                              │
检查速度命令配置          回退到最佳检查点
检查奖励函数权重          降低学习率
│                              │
修改配置，重新训练        继续训练
```

---

## 📞 获取帮助

如果遇到本文档未涵盖的问题：

1. **查看完整文档**:
   - `unified_training_guide.md` - 完整训练指南
   - `unified_hyperparameter_tuning.md` - 调参指南
   - `training_resume_guide.md` - 恢复训练指南

2. **检查终端输出**:
   - 保存完整的错误信息
   - 记录出现问题的 iteration

3. **检查 TensorBoard**:
   - 截图异常的曲线
   - 记录具体的数值

4. **记录实验配置**:
   - 超参数设置
   - 硬件配置
   - 操作系统版本

---

**记住**: 强化学习训练需要耐心，前期性能差是正常的。关键是观察**趋势**，而不是纠结于单个数值！
