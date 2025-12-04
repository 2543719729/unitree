# Unitree RL Lab 部署指南

本文档详细介绍了 `deploy` 文件夹的内容、代码架构、使用方法以及需要人为修改的配置项。

---

## 目录

1. [概述](#概述)
2. [目录结构](#目录结构)
3. [核心架构](#核心架构)
4. [状态机系统 (FSM)](#状态机系统-fsm)
5. [IsaacLab 部署框架](#isaaclab-部署框架)
6. [手柄控制 DSL](#手柄控制-dsl)
7. [机器人配置](#机器人配置)
8. [策略文件格式](#策略文件格式)
9. [编译与运行](#编译与运行)
10. [需要人为修改的内容](#需要人为修改的内容)
11. [常见问题](#常见问题)

---

## 概述

`deploy` 文件夹包含了将 IsaacLab 中训练的强化学习策略部署到真实 Unitree 机器人上的完整 C++ 框架。该框架支持：

- **多种机器人类型**：Go2、G1 (23DOF/29DOF)、H1、H1_2、B2、Go2W
- **有限状态机 (FSM)**：管理机器人的不同控制模式
- **ONNX 推理**：加载训练好的 PyTorch 模型进行实时推理
- **手柄控制**：通过 DSL 定义复杂的按键组合进行状态切换
- **模块化设计**：观测 (Observation) 和动作 (Action) 管理器可扩展

---

## 目录结构

```
deploy/
├── include/                    # 公共头文件
│   ├── FSM/                    # 有限状态机相关
│   │   ├── BaseState.h         # 状态基类
│   │   ├── CtrlFSM.h           # FSM 控制器
│   │   ├── FSMState.h          # FSM 状态包装
│   │   ├── State_Passive.h     # 被动模式状态
│   │   ├── State_FixStand.h    # 固定站立状态
│   │   └── State_RLBase.h      # RL 控制基础状态
│   ├── isaaclab/               # IsaacLab 部署框架
│   │   ├── algorithms/         # 算法实现 (ONNX 推理)
│   │   ├── assets/             # 机器人数据结构
│   │   ├── envs/               # 环境管理
│   │   ├── manager/            # 观测/动作管理器
│   │   └── utils/              # 工具函数
│   ├── param.h                 # 参数加载和命令行解析
│   ├── unitree_articulation.h  # Unitree 机器人关节数据
│   ├── unitree_joystick_dsl.hpp # 手柄 DSL 解析器
│   └── LinearInterpolator.h    # 线性插值工具
├── robots/                     # 各机器人专用代码
│   ├── go2/                    # Go2 四足机器人
│   ├── g1_23dof/               # G1 人形机器人 (23自由度)
│   ├── g1_29dof/               # G1 人形机器人 (29自由度)
│   ├── h1/                     # H1 人形机器人
│   ├── h1_2/                   # H1_2 人形机器人
│   ├── b2/                     # B2 四足机器人
│   └── go2w/                   # Go2W 轮式机器人
└── thirdparty/                 # 第三方库
    └── onnxruntime-linux-x64-1.22.0/  # ONNX Runtime
```

---

## 核心架构

### 系统架构图

```
┌──────────────────────────────────────────────────────────────────┐
│                         主程序 (main.cpp)                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │   CtrlFSM   │───▶│  FSMState   │───▶│  State_RLBase       │  │
│  │  (状态机)   │    │  (状态包装) │    │  (RL控制状态)       │  │
│  └─────────────┘    └─────────────┘    └──────────┬──────────┘  │
│                                                    │             │
│                                                    ▼             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              ManagerBasedRLEnv (RL环境)                   │  │
│  ├───────────────────────────────────────────────────────────┤  │
│  │  ┌─────────────────┐  ┌──────────────┐  ┌──────────────┐ │  │
│  │  │ ObservationMgr  │  │  ActionMgr   │  │  OrtRunner   │ │  │
│  │  │   (观测管理)    │  │  (动作管理)  │  │  (推理引擎)  │ │  │
│  │  └────────┬────────┘  └──────┬───────┘  └──────┬───────┘ │  │
│  │           │                  │                 │          │  │
│  │           ▼                  ▼                 ▼          │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │              Articulation (机器人数据)              │ │  │
│  │  │  - joint_pos, joint_vel, root_ang_vel_b, etc.       │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Unitree SDK2 (DDS 通信)                        │
│              LowCmd (发送命令) / LowState (接收状态)              │
└──────────────────────────────────────────────────────────────────┘
```

### 控制循环

1. **FSM 线程** (1kHz): 状态切换检测、关节命令发布
2. **策略线程** (50Hz, 由 `step_dt` 决定): 观测计算、神经网络推理、动作处理

---

## 状态机系统 (FSM)

### 状态基类 (`BaseState`)

所有状态都继承自 `BaseState`，主要接口：

```cpp
class BaseState {
public:
    virtual void enter() {}      // 进入状态时调用
    virtual void pre_run() {}    // run 之前调用
    virtual void run() {}        // 主循环
    virtual void post_run() {}   // run 之后调用
    virtual void exit() {}       // 退出状态时调用

    // 状态转换条件列表：(条件函数, 目标状态ID)
    std::vector<std::pair<std::function<bool()>, int>> registered_checks;
};
```

### 预定义状态

| 状态名 | 说明 |
|--------|------|
| `State_Passive` | 被动模式，电机无力矩输出，仅设置阻尼 |
| `State_FixStand` | 固定站立，通过线性插值将关节移动到目标位置 |
| `State_RLBase` | RL 控制模式，加载策略并运行推理 |
| `State_Mimic` | 动作模仿模式 (G1_29dof 专用) |

### 状态注册宏

使用 `REGISTER_FSM` 宏自动注册状态：

```cpp
class State_MyCustom : public FSMState {
    // ...
};
REGISTER_FSM(State_MyCustom)
```

### FSM 配置 (config.yaml)

```yaml
FSM:
  _:  # 启用的状态列表
    Passive:
      id: 1
    FixStand:
      id: 2
    Velocity:
      id: 3
      type: RLBase  # 使用 State_RLBase 类
  
  Passive:
    transitions:  # 状态转换条件
      FixStand: LT + A.on_pressed  # 按住 LT 并按下 A 进入 FixStand
    kd: [3, 3, 3, ...]  # 阻尼参数
  
  FixStand:
    transitions:
      Passive: LT + B.on_pressed
      Velocity: start.on_pressed
    kp: [60, 80, 80, ...]  # 刚度参数
    kd: [5, 4, 4, ...]     # 阻尼参数
    ts: [0, 1, 2]          # 时间节点 (秒)
    qs: [                   # 对应时间的关节角度
      [],                   # 空表示使用当前位置
      [0.0, 1.36, -2.65, ...],  # 中间姿态
      [0., 0.8, -1.5, ...]      # 最终站立姿态
    ]
  
  Velocity:
    transitions:
      Passive: LT + B.on_pressed
    policy_dir: ../../../logs/rsl_rl/unitree_go2_velocity
```

---

## 手柄控制 DSL

`unitree_joystick_dsl.hpp` 实现了一个领域特定语言 (DSL)，用于定义复杂的手柄按键组合。

### 基本语法

| 语法 | 说明 | 示例 |
|------|------|------|
| `A` | 按键按住 | `A` |
| `A.on_pressed` | 按键刚按下 | `A.on_pressed` |
| `A.on_released` | 按键刚释放 | `A.on_released` |
| `A + B` | 同时按住 | `LT + A` |
| `A \| B` | 任一满足 | `X \| Y` |
| `!A` | 未按下 | `!LT + B` |
| `LT(2s)` | 长按 2 秒 | `LT(2s) + up` |
| `(A + B) \| C` | 分组 | `(LT + RT) \| start` |

### 可用按键

- **按钮**: `A`, `B`, `X`, `Y`, `LB`, `RB`, `back`, `start`, `LS`, `RS`, `F1`, `F2`
- **方向键**: `up`, `down`, `left`, `right`
- **扳机**: `LT`, `RT`
- **摇杆**: `lx`, `ly`, `rx`, `ry`

### 示例

```yaml
transitions:
  FixStand: LT + up.on_pressed        # LT 按住 + 上方向键刚按下
  Velocity: RB + X.on_pressed         # RB 按住 + X 刚按下
  Passive: LT + B.on_pressed          # LT 按住 + B 刚按下
  Mimic: LT(2s) + down.on_pressed     # LT 长按 2 秒 + 下方向键刚按下
```

---

## IsaacLab 部署框架

### ManagerBasedRLEnv

核心环境类，管理机器人状态、观测和动作：

```cpp
class ManagerBasedRLEnv {
public:
    float step_dt;  // 控制步长 (通常 0.02s = 50Hz)
    
    std::shared_ptr<Articulation> robot;  // 机器人数据
    std::unique_ptr<ObservationManager> observation_manager;
    std::unique_ptr<ActionManager> action_manager;
    std::unique_ptr<Algorithms> alg;  // ONNX 推理引擎
    
    void reset();
    void step();
};
```

### 观测管理器 (ObservationManager)

支持的观测项：

| 观测名 | 维度 | 说明 |
|--------|------|------|
| `base_ang_vel` | 3 | 机体角速度 (rad/s) |
| `projected_gravity` | 3 | 重力投影到机体坐标系 |
| `joint_pos` | N | 关节位置 (rad) |
| `joint_pos_rel` | N | 相对默认位置的关节偏差 |
| `joint_vel_rel` | N | 关节速度 (rad/s) |
| `last_action` | N | 上一步动作 |
| `velocity_commands` | 3 | 速度指令 (vx, vy, wz) |
| `gait_phase` | 2 | 步态相位 (sin, cos) |

#### 自定义观测

使用 `REGISTER_OBSERVATION` 宏添加新观测：

```cpp
REGISTER_OBSERVATION(my_custom_obs)
{
    // env: 环境指针
    // params: YAML 参数节点
    std::vector<float> obs;
    // ... 计算观测 ...
    return obs;
}
```

### 动作管理器 (ActionManager)

支持的动作类型：

| 动作类型 | 说明 |
|----------|------|
| `JointPositionAction` | 关节位置控制 |
| `JointVelocityAction` | 关节速度控制 |

动作处理流程：
1. 接收神经网络输出
2. 应用缩放 (`scale`)
3. 添加偏移 (`offset`)
4. 应用裁剪 (`clip`)

### ONNX 推理 (OrtRunner)

使用 ONNX Runtime 加载和运行 PyTorch 导出的模型：

```cpp
class OrtRunner : public Algorithms {
public:
    OrtRunner(std::string model_path);
    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs);
};
```

---

## 机器人配置

### 各机器人目录结构

每个机器人目录 (`robots/<robot_name>/`) 包含：

```
robots/go2/
├── CMakeLists.txt      # 编译配置
├── main.cpp            # 主程序入口
├── include/
│   └── Types.h         # 类型定义
├── src/
│   └── State_RLBase.cpp  # RL 状态实现
└── config/
    └── config.yaml     # FSM 和机器人配置
```

### Types.h

定义机器人特定的类型别名：

```cpp
// Go2 示例
#include "unitree/dds_wrapper/robots/go2/go2.h"

using LowCmd_t = unitree::robot::go2::publisher::LowCmd;
using LowState_t = unitree::robot::go2::subscription::LowState;
```

### main.cpp

```cpp
int main(int argc, char** argv)
{
    // 1. 解析命令行参数
    auto vm = param::helper(argc, argv);

    // 2. 初始化 DDS 通信
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    // 3. 初始化 FSM 状态
    init_fsm_state();

    // 4. 启动 FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    // 5. 主循环
    while (true) { sleep(1); }
}
```

---

## 策略文件格式

### 目录结构

```
policy_dir/
├── exported/
│   └── policy.onnx     # 导出的 ONNX 模型
└── params/
    └── deploy.yaml     # 部署配置
```

### deploy.yaml 格式

```yaml
# 关节 ID 映射 (训练顺序 -> SDK 顺序)
joint_ids_map: [0, 6, 12, 1, 7, 13, ...]

# 控制步长 (秒)
step_dt: 0.02

# PD 控制参数
stiffness: [100.0, 100.0, ...]  # Kp
damping: [2.0, 2.0, ...]        # Kd

# 默认关节位置
default_joint_pos: [-0.1, -0.1, 0.0, ...]

# 速度指令范围
commands:
  base_velocity:
    ranges:
      lin_vel_x: [-0.5, 1.0]   # 前后速度范围 (m/s)
      lin_vel_y: [-0.3, 0.3]   # 左右速度范围 (m/s)
      ang_vel_z: [-0.2, 0.2]   # 转向速度范围 (rad/s)

# 动作配置
actions:
  JointPositionAction:
    scale: [0.25, 0.25, ...]   # 动作缩放
    offset: [-0.1, -0.1, ...]  # 默认位置偏移
    clip: null                  # 裁剪范围 (可选)

# 观测配置
observations:
  base_ang_vel:
    params: {}
    scale: [0.2, 0.2, 0.2]     # 观测缩放
    clip: null
    history_length: 5           # 历史帧数

  projected_gravity:
    params: {}
    scale: [1.0, 1.0, 1.0]
    history_length: 5

  velocity_commands:
    params: {command_name: base_velocity}
    scale: [1.0, 1.0, 1.0]
    history_length: 5

  joint_pos_rel:
    params: {}
    scale: [1.0, 1.0, ...]
    history_length: 5

  joint_vel_rel:
    params: {}
    scale: [0.05, 0.05, ...]
    history_length: 5

  last_action:
    params: {}
    scale: [1.0, 1.0, ...]
    history_length: 5
```

---

## 编译与运行

### 依赖项

- **Boost** (program_options)
- **yaml-cpp**
- **Eigen3**
- **ONNX Runtime** (位于 `thirdparty/`)
- **Unitree SDK2**
- **spdlog** 和 **fmt**

### 编译

```bash
cd deploy/robots/go2
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 运行

```bash
# 基本运行
./go2_ctrl

# 指定网络接口
./go2_ctrl -n eth0

# 启用日志记录
./go2_ctrl --log

# 查看帮助
./go2_ctrl -h
```

### 命令行参数

| 参数 | 说明 |
|------|------|
| `-h, --help` | 显示帮助信息 |
| `-v, --version` | 显示版本号 |
| `-n, --network` | 指定 DDS 网络接口 |
| `--log` | 启用日志记录到文件 |

---

## 需要人为修改的内容

### 1. config.yaml - FSM 配置

**文件位置**: `robots/<robot>/config/config.yaml`

需要修改的内容：

```yaml
FSM:
  # ★ 修改策略路径
  Velocity:
    policy_dir: ../../../logs/rsl_rl/unitree_go2_velocity  # 指向你的策略目录

  # ★ 修改 FixStand 目标姿态
  FixStand:
    qs: [
      [],  # 初始位置 (自动填充)
      [...],  # 中间姿态
      [...],  # 最终站立姿态
    ]

  # ★ 调整 PD 参数
  FixStand:
    kp: [60, 80, 80, ...]  # 刚度
    kd: [5, 4, 4, ...]     # 阻尼
```

### 2. deploy.yaml - 策略部署配置

**文件位置**: `<policy_dir>/params/deploy.yaml`

通常由训练脚本自动生成，但可能需要调整：

```yaml
# ★ 关节 ID 映射 (如果训练和 SDK 关节顺序不同)
joint_ids_map: [0, 6, 12, 1, 7, 13, ...]

# ★ PD 控制参数 (根据实机调试)
stiffness: [100.0, 100.0, ...]
damping: [2.0, 2.0, ...]

# ★ 速度指令范围 (根据策略能力调整)
commands:
  base_velocity:
    ranges:
      lin_vel_x: [-0.5, 1.0]
      lin_vel_y: [-0.3, 0.3]
      ang_vel_z: [-0.2, 0.2]
```

### 3. State_RLBase.cpp - RL 状态实现

**文件位置**: `robots/<robot>/src/State_RLBase.cpp`

可能需要修改的内容：

```cpp
// ★ 添加自定义终止条件
this->registered_checks.emplace_back(
    std::make_pair(
        [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
        FSMStringMap.right.at("Passive")
    )
);

// ★ 修改关节命令发送方式
void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
```

### 4. 添加新的观测项

**文件位置**: `include/isaaclab/envs/mdp/observations/observations.h`

```cpp
REGISTER_OBSERVATION(my_custom_obs)
{
    auto & asset = env->robot;
    std::vector<float> data;
    
    // 计算自定义观测
    // ...
    
    return data;
}
```

### 5. 添加新的状态

**步骤**:
1. 创建新的头文件 `include/FSM/State_MyCustom.h`
2. 继承 `FSMState` 并实现所需方法
3. 使用 `REGISTER_FSM(State_MyCustom)` 注册
4. 在 `config.yaml` 中添加配置

---

## 常见问题

### Q1: 策略无法加载

**可能原因**:
- 策略路径不正确
- 缺少 `exported/policy.onnx` 或 `params/deploy.yaml`
- ONNX 模型输入/输出名称不匹配

**解决方案**:
- 检查 `policy_dir` 路径
- 确保训练时正确导出了部署文件
- 检查 `deploy.yaml` 中的观测配置是否与模型输入匹配

### Q2: 通信超时

**可能原因**:
- 网络接口配置错误
- 其他进程占用了控制通道

**解决方案**:
- 使用 `-n` 参数指定正确的网络接口
- 关闭其他可能使用 `LowCmd` 通道的程序

### Q3: 机器人抖动

**可能原因**:
- PD 参数不合适
- 观测噪声过大
- 控制频率与训练不匹配

**解决方案**:
- 调整 `stiffness` 和 `damping` 参数
- 检查 `step_dt` 是否与训练一致
- 检查观测 `scale` 是否正确

### Q4: 姿态检测触发过于敏感

**解决方案**:
修改 `State_RLBase.cpp` 中的姿态阈值：

```cpp
[&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.2); }  // 增大阈值
```

---

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| 1.0.0.1 | 2025 | 初始版本 |

---

## 联系方式

如有问题，请联系 Unitree Robotics 技术支持。

---

*本文档由 Unitree RL Lab 团队维护*
