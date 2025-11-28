# Unitree RL Lab 环境配置与使用指南

## 一、系统信息

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU (8GB) |
| Driver | 576.80 |
| Python | 3.11 |
| PyTorch | 2.7.0+cu126 (CUDA 12.6) |
| Isaac Sim | 5.1.0.0 |
| IsaacLab | 0.48.6 |

---

## 二、环境说明

### 2.1 Conda 环境位置
```
E:\isaacenv
```

> **为什么不用 `conda activate`？**
> 
> 由于 Windows PowerShell 的限制，`conda activate` 有时无法正确激活环境。
> 因此我们直接使用 Python 完整路径来确保使用正确的环境。

### 2.2 项目目录结构
```
E:\Aunitree\
├── IsaacLab/           # Isaac Lab 框架
├── unitree_rl_lab/     # Unitree RL 训练代码 (主工作目录)
├── unitree_ros/        # 机器人 URDF 模型文件
└── SETUP_GUIDE.md      # 本文档
```

---

## 三、命令详解

### 3.1 完整命令
```powershell
$env:GIT_PYTHON_GIT_EXECUTABLE='E:\isaacenv\Library\bin\git.exe'; E:\isaacenv\python.exe scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity
```

### 3.2 各部分解释

| 部分 | 作用 |
|------|------|
| `$env:GIT_PYTHON_GIT_EXECUTABLE='E:\isaacenv\Library\bin\git.exe'` | 设置 Git 路径环境变量。RSL-RL 库依赖 GitPython，需要找到 git.exe |
| `;` | PowerShell 命令分隔符，用于在一行执行多个命令 |
| `E:\isaacenv\python.exe` | 直接调用 conda 环境中的 Python 解释器（避免 conda activate 问题） |
| `scripts/rsl_rl/train.py` | 训练脚本路径 |
| `--headless` | 无头模式运行（不显示图形界面，适合纯训练） |
| `--task Unitree-G1-29dof-Velocity` | 指定训练任务名称 |

---

## 四、简化使用方法

### 4.1 方法一：创建启动脚本（推荐）

创建 `E:\Aunitree\train.ps1`：
```powershell
# Unitree RL Lab 训练启动脚本
$env:GIT_PYTHON_GIT_EXECUTABLE = 'E:\isaacenv\Library\bin\git.exe'
$env:OMNI_KIT_ACCEPT_EULA = 'yes'

Set-Location E:\Aunitree\unitree_rl_lab
& E:\isaacenv\python.exe scripts/rsl_rl/train.py @args
```

使用方式：
```powershell
# 训练 G1 机器人
.\train.ps1 --headless --task Unitree-G1-29dof-Velocity

# 训练 Go2 机器人
.\train.ps1 --headless --task Unitree-Go2-Velocity
```

### 4.2 方法二：设置永久环境变量

在 PowerShell 中运行一次（管理员权限）：
```powershell
[Environment]::SetEnvironmentVariable('GIT_PYTHON_GIT_EXECUTABLE', 'E:\isaacenv\Library\bin\git.exe', 'User')
[Environment]::SetEnvironmentVariable('OMNI_KIT_ACCEPT_EULA', 'yes', 'User')
```

之后只需：
```powershell
cd E:\Aunitree\unitree_rl_lab
E:\isaacenv\python.exe scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity
```

### 4.3 方法三：添加别名到 PowerShell Profile

编辑 `$PROFILE` 文件，添加：
```powershell
function utrain {
    $env:GIT_PYTHON_GIT_EXECUTABLE = 'E:\isaacenv\Library\bin\git.exe'
    $env:OMNI_KIT_ACCEPT_EULA = 'yes'
    Push-Location E:\Aunitree\unitree_rl_lab
    & E:\isaacenv\python.exe scripts/rsl_rl/train.py @args
    Pop-Location
}

function uplay {
    $env:GIT_PYTHON_GIT_EXECUTABLE = 'E:\isaacenv\Library\bin\git.exe'
    $env:OMNI_KIT_ACCEPT_EULA = 'yes'
    Push-Location E:\Aunitree\unitree_rl_lab
    & E:\isaacenv\python.exe scripts/rsl_rl/play.py @args
    Pop-Location
}
```

使用方式：
```powershell
utrain --headless --task Unitree-G1-29dof-Velocity
uplay --task Unitree-G1-29dof-Velocity
```

---

## 五、常用命令

### 5.1 训练命令
```powershell
cd E:\Aunitree\unitree_rl_lab
$env:GIT_PYTHON_GIT_EXECUTABLE='E:\isaacenv\Library\bin\git.exe'

# G1 人形机器人 (29自由度)
E:\isaacenv\python.exe scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity

# Go2 四足机器人
E:\isaacenv\python.exe scripts/rsl_rl/train.py --headless --task Unitree-Go2-Velocity

# H1 人形机器人
E:\isaacenv\python.exe scripts/rsl_rl/train.py --headless --task Unitree-H1-Velocity
```

### 5.2 推理/可视化
```powershell
# 加载训练好的模型进行可视化（去掉 --headless）
E:\isaacenv\python.exe scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity
```

### 5.3 常用参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--headless` | 无图形界面训练 | `--headless` |
| `--task` | 指定任务 | `--task Unitree-G1-29dof-Velocity` |
| `--num_envs` | 并行环境数量 | `--num_envs 2048` |
| `--max_iterations` | 最大迭代次数 | `--max_iterations 1000` |
| `--video` | 录制视频 | `--video` |
| `--video_length` | 视频长度(步) | `--video_length 200` |

### 5.4 查看帮助
```powershell
E:\isaacenv\python.exe scripts/rsl_rl/train.py --help
```

---

## 六、支持的任务

| 任务名称 | 机器人 | 说明 |
|----------|--------|------|
| `Unitree-Go2-Velocity` | Go2 | 四足机器人速度控制 |
| `Unitree-H1-Velocity` | H1 | 人形机器人速度控制 |
| `Unitree-G1-29dof-Velocity` | G1 | 29自由度人形机器人速度控制 |
| `Unitree-G1-29dof-Mimic-Dance-102` | G1 | 舞蹈模仿 |
| `Unitree-G1-29dof-Mimic-Gangnanm-Style` | G1 | 江南Style舞蹈模仿 |

---

## 七、训练输出说明

训练过程中会显示以下关键指标：

```
Learning iteration 10/50000           # 当前迭代/总迭代
Total timesteps: 1081344              # 总训练步数
Computation: 22223 steps/s            # 训练速度
Mean reward: -3.66                    # 平均奖励（越高越好）
Mean episode length: 42.60            # 平均回合长度
ETA: 16:06:36                         # 预计剩余时间
```

### 模型保存位置
```
E:\Aunitree\unitree_rl_lab\logs\rsl_rl\<task_name>\<timestamp>\
├── model_*.pt          # 模型检查点
├── config.yaml         # 训练配置
└── tensorboard/        # TensorBoard 日志
```

### 查看 TensorBoard
```powershell
E:\isaacenv\python.exe -m tensorboard.main --logdir E:\Aunitree\unitree_rl_lab\logs\rsl_rl
```

---

## 八、常见问题

### Q1: 报错 "No module named 'gymnasium'"
**原因**: 没有使用正确的 Python 环境
**解决**: 使用完整路径 `E:\isaacenv\python.exe` 而不是 `python`

### Q2: 报错 "Bad git executable"
**原因**: GitPython 找不到 git.exe
**解决**: 设置环境变量 `$env:GIT_PYTHON_GIT_EXECUTABLE='E:\isaacenv\Library\bin\git.exe'`

### Q3: 报错 "EULA not accepted"
**原因**: 未接受 NVIDIA 许可协议
**解决**: 设置环境变量 `$env:OMNI_KIT_ACCEPT_EULA='yes'`

### Q4: GPU 内存不足
**解决**: 减少并行环境数量
```powershell
E:\isaacenv\python.exe scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity --num_envs 1024
```

### Q5: 训练太慢
**建议**: RTX 4070 Laptop (8GB) 建议使用 2048-4096 个并行环境

---

## 九、快速开始（一键复制）

```powershell
# 复制以下内容到 PowerShell 即可开始训练
cd E:\Aunitree\unitree_rl_lab
$env:GIT_PYTHON_GIT_EXECUTABLE='E:\isaacenv\Library\bin\git.exe'
$env:OMNI_KIT_ACCEPT_EULA='yes'
E:\isaacenv\python.exe scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity
```

---

*文档更新时间: 2025-11-25*
