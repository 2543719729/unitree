# Unitree G1 29DOF MuJoCo RL Training

基于 MuJoCo 的 G1 29DOF 人形机器人强化学习训练框架，仿照 IsaacLab `unified_env_cfg.py` 实现。

## 功能特性

- **4模式条件策略训练**：
  - 模式0: 平地盲走 (无 height_scan)
  - 模式1: 平地带传感器 (有 height_scan)
  - 模式2: 楼梯盲爬 (无 height_scan)
  - 模式3: 楼梯带传感器 (有 height_scan)

- **完整的观测空间**：
  - 基座角速度、重力投影
  - 关节位置/速度
  - 速度命令
  - 模式标志 (one-hot)
  - 条件 height_scan
  - 历史观测 (5帧)

- **丰富的奖励函数**：
  - 速度跟踪奖励
  - 向上进展奖励 (楼梯模式)
  - 姿态/步态奖励
  - 正则化惩罚

- **域随机化**：
  - 物理材质随机化
  - 初始状态随机化
  - 外部扰动

## 安装

```bash
cd g1_rl_train
pip install -r requirements.txt
```

## 项目结构

```
g1_rl_train/
├── configs/
│   ├── __init__.py
│   └── unified_env_cfg.py      # 环境配置 (仿照 IsaacLab)
├── envs/
│   ├── __init__.py
│   └── g1_29dof_env.py         # MuJoCo Gym 环境
├── utils/
│   ├── __init__.py
│   └── rewards.py              # 奖励函数模块
├── train_ppo.py                # PPO 训练脚本
├── requirements.txt
└── README.md
```

## 使用方法

### 训练

```bash
# 统一模式训练 (4模式)
python train_ppo.py train --mode unified --num_envs 16 --total_timesteps 10000000

# 平地盲走训练
python train_ppo.py train --mode flat_blind --num_envs 16

# 楼梯带传感器训练
python train_ppo.py train --mode stairs_sensor --num_envs 16

# 从检查点恢复训练
python train_ppo.py train --mode unified --resume ./logs/xxx/checkpoints/g1_29dof_xxx.zip
```

### 评估

```bash
python train_ppo.py eval --model_path ./logs/xxx/best_model/best_model.zip
```

### 测试环境

```bash
cd envs
python g1_29dof_env.py
```

## 配置说明

主要配置在 `configs/unified_env_cfg.py`:

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `sim_dt` | 仿真时间步 | 0.005s |
| `control_decimation` | 控制降采样 | 4 |
| `episode_length_s` | 回合时长 | 20.0s |
| `history_length` | 观测历史长度 | 5 |
| `action_scale` | 动作缩放 | 0.25 |
| `kp/kd` | PD控制器参数 | 50.0/3.5 |

## 与 IsaacLab 的对应关系

| IsaacLab | MuJoCo 实现 |
|----------|-------------|
| `UnifiedSceneCfg` | `G1_29DOF_Env._load_model()` |
| `UnifiedObservationsCfg` | `G1_29DOF_Env._get_observation()` |
| `UnifiedRewardsCfg` | `utils/rewards.py` |
| `UnifiedEventCfg` | `G1_29DOF_Env._apply_domain_randomization()` |
| `UnifiedActionsCfg` | `G1_29DOF_Env._apply_pd_control()` |

## 训练曲线

训练日志保存在 `./logs` 目录，可用 TensorBoard 查看：

```bash
tensorboard --logdir ./logs
```

## 注意事项

1. MuJoCo 版本需 >= 3.0.0
2. 单环境训练较慢，建议使用多环境并行
3. Height scan 功能需要实现 ray casting（当前为占位符）
4. 接触检测功能需要进一步完善

## License

MIT License
