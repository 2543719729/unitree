# Unitree RL Lab 训练配置指南

本文档详细介绍了 Unitree RL Lab 中强化学习训练的各项可配置参数，帮助你自定义训练场景和策略。

---

## 目录

1. [配置文件结构](#一配置文件结构)
2. [PPO算法配置](#二ppo算法配置)
3. [奖励函数配置](#三奖励函数配置)
4. [观测配置](#四观测配置)
5. [动作配置](#五动作配置)
6. [场景与地形配置](#六场景与地形配置)
7. [命令配置](#七命令配置)
8. [终止条件配置](#八终止条件配置)
9. [事件与随机化配置](#九事件与随机化配置)
10. [课程学习配置](#十课程学习配置)
11. [创建自定义训练任务](#十一创建自定义训练任务)

---

## 一、配置文件结构

训练配置主要分布在以下目录结构中：

```
unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/
├── locomotion/                    # 移动控制任务
│   ├── agents/                    # PPO算法配置
│   │   └── rsl_rl_ppo_cfg.py
│   ├── mdp/                       # MDP组件（奖励、观测、动作等）
│   │   ├── rewards.py
│   │   ├── observations.py
│   │   └── commands/
│   └── robots/                    # 各机器人的环境配置
│       ├── go2/velocity_env_cfg.py
│       ├── g1/29dof/velocity_env_cfg.py
│       └── h1/velocity_env_cfg.py
└── mimic/                         # 动作模仿任务
    ├── agents/
    ├── mdp/
    └── robots/g1_29dof/
```

### 核心配置类

每个环境配置文件通常包含以下配置类：

| 配置类 | 作用 |
|--------|------|
| `RobotSceneCfg` | 场景配置（地形、机器人、传感器、灯光） |
| `CommandsCfg` | 命令配置（速度指令范围等） |
| `ActionsCfg` | 动作配置（关节控制方式） |
| `ObservationsCfg` | 观测配置（策略输入） |
| `RewardsCfg` | 奖励配置（各项奖励权重） |
| `TerminationsCfg` | 终止条件配置 |
| `EventCfg` | 事件配置（随机化、扰动） |
| `CurriculumCfg` | 课程学习配置 |
| `RobotEnvCfg` | 总环境配置（整合以上所有配置） |

---

## 二、PPO算法配置

**配置文件位置**: `tasks/locomotion/agents/rsl_rl_ppo_cfg.py`

### 2.1 基础运行配置

```python
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24        # 每个环境每次迭代收集的步数
    max_iterations = 50000        # 最大训练迭代次数
    save_interval = 100           # 模型保存间隔（每N次迭代保存一次）
    experiment_name = ""          # 实验名称（默认与任务名相同）
    empirical_normalization = False  # 是否使用经验归一化
```

### 2.2 神经网络结构配置

```python
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,                    # 初始动作噪声标准差
        actor_hidden_dims=[512, 256, 128],     # Actor网络隐藏层维度
        critic_hidden_dims=[512, 256, 128],    # Critic网络隐藏层维度
        activation="elu",                       # 激活函数 ("elu", "relu", "tanh")
    )
```

**可调整参数说明**：
- `actor_hidden_dims`: 增大网络可以提升表达能力，但会增加训练时间
- `init_noise_std`: 较大的初始噪声有助于探索，但可能影响收敛速度

### 2.3 PPO算法超参数

```python
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,          # 价值函数损失系数
        use_clipped_value_loss=True,  # 是否使用裁剪的价值损失
        clip_param=0.2,               # PPO裁剪参数
        entropy_coef=0.01,            # 熵正则化系数（鼓励探索）
        num_learning_epochs=5,        # 每次更新的训练轮数
        num_mini_batches=4,           # mini-batch数量
        learning_rate=1.0e-3,         # 学习率
        schedule="adaptive",          # 学习率调度 ("fixed", "adaptive")
        gamma=0.99,                   # 折扣因子
        lam=0.95,                     # GAE lambda参数
        desired_kl=0.01,              # 目标KL散度（用于adaptive schedule）
        max_grad_norm=1.0,            # 梯度裁剪阈值
    )
```

**关键超参数调整建议**：

| 参数 | 作用 | 调整建议 |
|------|------|----------|
| `learning_rate` | 控制学习速度 | 过大导致不稳定，过小收敛慢 |
| `gamma` | 折扣因子 | 接近1更重视长期回报 |
| `entropy_coef` | 探索程度 | 增大促进探索，减小促进利用 |
| `clip_param` | PPO裁剪范围 | 通常0.1-0.3，控制策略更新幅度 |

---

## 三、奖励函数配置

**配置文件位置**: `tasks/locomotion/robots/go2/velocity_env_cfg.py` 中的 `RewardsCfg`

奖励函数是训练中最重要的配置之一，直接决定了机器人学到的行为。

### 3.1 奖励项定义格式

```python
from isaaclab.managers import RewardTermCfg as RewTerm

@configclass
class RewardsCfg:
    reward_name = RewTerm(
        func=mdp.reward_function,  # 奖励函数
        weight=1.0,                 # 权重（正数为奖励，负数为惩罚）
        params={...}                # 函数参数
    )
```

### 3.2 速度跟踪奖励

```python
    # 线速度跟踪（xy平面）
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.5}
    )
    
    # 角速度跟踪（z轴旋转）
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": 0.5}
    )
```

### 3.3 基础惩罚项

```python
    # 垂直方向线速度惩罚（防止跳跃）
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    
    # 横滚和俯仰角速度惩罚（保持稳定）
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    
    # 关节速度惩罚
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    
    # 关节加速度惩罚（平滑运动）
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    
    # 关节力矩惩罚（节能）
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2e-4)
    
    # 动作变化率惩罚（平滑控制）
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    
    # 关节限位惩罚
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    
    # 能耗惩罚
    energy = RewTerm(func=mdp.energy, weight=-2e-5)
```

### 3.4 姿态奖励

```python
    # 保持水平姿态
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
    
    # 关节位置偏离默认位置的惩罚
    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,      # 静止时惩罚倍数
            "velocity_threshold": 0.3,      # 速度阈值
        },
    )
```

### 3.5 步态相关奖励

```python
    # 脚部滞空时间奖励（鼓励抬脚）
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    
    # 滞空时间方差惩罚（四足步态对称性）
    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    
    # 脚部滑动惩罚
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
```

### 3.6 接触惩罚

```python
    # 非期望接触惩罚（防止身体碰撞地面）
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["Head_.*", ".*_hip", ".*_thigh", ".*_calf"]
            ),
        },
    )
```

### 3.7 可用的奖励函数一览

**文件位置**: `tasks/locomotion/mdp/rewards.py`

| 函数名 | 作用 |
|--------|------|
| `track_lin_vel_xy_exp` | 线速度跟踪（指数形式） |
| `track_ang_vel_z_exp` | 角速度跟踪（指数形式） |
| `lin_vel_z_l2` | 垂直速度L2惩罚 |
| `ang_vel_xy_l2` | 横滚俯仰角速度L2惩罚 |
| `joint_vel_l2` | 关节速度L2惩罚 |
| `joint_acc_l2` | 关节加速度L2惩罚 |
| `joint_torques_l2` | 关节力矩L2惩罚 |
| `joint_pos_limits` | 关节限位惩罚 |
| `action_rate_l2` | 动作变化率L2惩罚 |
| `flat_orientation_l2` | 姿态水平惩罚 |
| `energy` | 能耗惩罚 |
| `feet_air_time` | 脚部滞空时间奖励 |
| `feet_slide` | 脚部滑动惩罚 |
| `feet_stumble` | 脚部绊倒惩罚 |
| `feet_gait` | 步态周期奖励 |
| `foot_clearance_reward` | 抬脚高度奖励 |
| `undesired_contacts` | 非期望接触惩罚 |
| `joint_mirror` | 关节对称性惩罚 |
| `orientation_l2` | 重力方向对齐奖励 |
| `upward` | 保持直立奖励 |

---

## 四、观测配置

**配置文件位置**: `velocity_env_cfg.py` 中的 `ObservationsCfg`

观测决定了策略网络能够感知到哪些信息。

### 4.1 观测项定义格式

```python
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        obs_name = ObsTerm(
            func=mdp.observation_function,  # 观测函数
            scale=1.0,                       # 缩放系数
            clip=(-100, 100),                # 裁剪范围
            noise=Unoise(n_min=-0.1, n_max=0.1),  # 添加噪声（可选）
            params={...}                     # 函数参数
        )
```

### 4.2 策略观测（Policy Observations）

```python
    @configclass
    class PolicyCfg(ObsGroup):
        # 基座角速度
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        
        # 投影重力向量（用于感知倾斜）
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        
        # 速度命令
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            clip=(-100, 100),
            params={"command_name": "base_velocity"}
        )
        
        # 相对关节位置（相对于默认位置）
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        
        # 相对关节速度
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            clip=(-100, 100),
            noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        
        # 上一时刻的动作
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self):
            self.enable_corruption = True   # 是否启用噪声
            self.concatenate_terms = True   # 是否拼接所有观测
```

### 4.3 特权观测（Critic Observations）

特权观测仅用于训练Critic网络，可以包含策略无法直接获取的信息：

```python
    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))  # 线速度（策略无法直接获取）
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))  # 关节力矩
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        # 可选：高度扫描传感器
        # height_scanner = ObsTerm(func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner")})
```

### 4.4 常用观测函数

| 函数名 | 输出维度 | 说明 |
|--------|----------|------|
| `base_lin_vel` | 3 | 基座线速度 (x, y, z) |
| `base_ang_vel` | 3 | 基座角速度 (roll, pitch, yaw) |
| `projected_gravity` | 3 | 投影到机器人坐标系的重力向量 |
| `joint_pos_rel` | N | 相对关节位置 |
| `joint_vel_rel` | N | 相对关节速度 |
| `joint_effort` | N | 关节力矩 |
| `last_action` | N | 上一时刻动作 |
| `generated_commands` | 3 | 速度命令 (vx, vy, wz) |
| `height_scan` | M | 高度扫描点云 |

---

## 五、动作配置

**配置文件位置**: `velocity_env_cfg.py` 中的 `ActionsCfg`

### 5.1 关节位置控制

```python
@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],           # 控制哪些关节（正则表达式匹配）
        scale=0.25,                   # 动作缩放系数
        use_default_offset=True,      # 是否使用默认关节位置作为偏移
        clip={".*": (-100.0, 100.0)}  # 动作裁剪范围
    )
```

### 5.2 动作缩放说明

- `scale`: 网络输出乘以此系数后作为关节位置增量
- 较小的scale使动作更平滑，但响应较慢
- 较大的scale响应快，但可能导致抖动

---

## 六、场景与地形配置

**配置文件位置**: `velocity_env_cfg.py` 中的 `RobotSceneCfg`

### 6.1 地形生成器配置

```python
import isaaclab.terrains as terrain_gen

TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),              # 每块地形大小
    border_width=20.0,            # 边界宽度
    num_rows=10,                  # 行数
    num_cols=20,                  # 列数
    horizontal_scale=0.1,         # 水平分辨率
    vertical_scale=0.005,         # 垂直分辨率
    slope_threshold=0.75,         # 坡度阈值
    difficulty_range=(0.0, 1.0),  # 难度范围
    use_cache=False,              # 是否缓存地形
    sub_terrains={...}            # 子地形配置
)
```

### 6.2 可用的地形类型

```python
sub_terrains={
    # 平坦地面
    "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
    
    # 随机粗糙地形
    "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        proportion=0.1,
        noise_range=(0.01, 0.06),
        noise_step=0.01,
        border_width=0.25
    ),
    
    # 金字塔斜坡
    "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        proportion=0.1,
        slope_range=(0.0, 0.4),
        platform_width=2.0,
        border_width=0.25
    ),
    
    # 倒金字塔斜坡
    "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        proportion=0.1,
        slope_range=(0.0, 0.4),
        platform_width=2.0,
        border_width=0.25
    ),
    
    # 随机方块
    "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        proportion=0.2,
        grid_width=0.45,
        grid_height_range=(0.05, 0.2),
        platform_width=2.0
    ),
    
    # 金字塔楼梯
    "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        proportion=0.2,
        step_height_range=(0.05, 0.23),
        step_width=0.3,
        platform_width=3.0,
        border_width=1.0,
        holes=False
    ),
    
    # 倒金字塔楼梯
    "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        proportion=0.2,
        step_height_range=(0.05, 0.23),
        step_width=0.3,
        platform_width=3.0,
        border_width=1.0,
        holes=False
    ),
}
```

### 6.3 场景配置

```python
@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    # 地形配置
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane"（平面）或 "generator"（生成器）
        terrain_generator=TERRAIN_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    
    # 机器人配置
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # 高度扫描传感器
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    
    # 接触力传感器
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True
    )
```

### 6.4 物理材质参数

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `static_friction` | 静摩擦系数 | 0.8-1.2 |
| `dynamic_friction` | 动摩擦系数 | 0.8-1.2 |
| `restitution` | 恢复系数（弹性） | 0.0-0.2 |

---

## 七、命令配置

**配置文件位置**: `velocity_env_cfg.py` 中的 `CommandsCfg`

命令配置定义了训练时给机器人的目标指令。

### 7.1 速度命令配置

```python
@configclass
class CommandsCfg:
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),  # 命令重采样时间间隔
        rel_standing_envs=0.1,                # 站立环境比例（10%的环境速度为0）
        debug_vis=True,                       # 是否可视化命令
        
        # 初始速度范围（训练开始时）
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1),   # 前后速度 (m/s)
            lin_vel_y=(-0.1, 0.1),   # 左右速度 (m/s)
            ang_vel_z=(-1, 1)        # 旋转速度 (rad/s)
        ),
        
        # 最大速度范围（课程学习后）
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.4, 0.4),
            ang_vel_z=(-1.0, 1.0)
        ),
    )
```

### 7.2 动作模仿任务的命令配置

```python
@configclass
class CommandsCfg:
    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        motion_file="path/to/motion.npz",   # 动作捕捉文件
        anchor_body_name="torso_link",      # 锚点身体
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        
        # 位姿随机化范围
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        
        # 速度随机化范围
        velocity_range={
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.2, 0.2),
            "roll": (-0.52, 0.52),
            "pitch": (-0.52, 0.52),
            "yaw": (-0.78, 0.78),
        },
        
        joint_position_range=(-0.1, 0.1),  # 关节位置随机化
        body_names=[...],                   # 追踪的身体部位列表
    )
```

---

## 八、终止条件配置

**配置文件位置**: `velocity_env_cfg.py` 中的 `TerminationsCfg`

终止条件决定了何时结束一个episode。

```python
from isaaclab.managers import TerminationTermCfg as DoneTerm

@configclass
class TerminationsCfg:
    # 时间超限
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # 基座接触地面
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0
        },
    )
    
    # 姿态过度倾斜
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.8}  # 弧度，约46度
    )
```

### 可用的终止函数

| 函数名 | 说明 |
|--------|------|
| `time_out` | 达到最大episode时间 |
| `illegal_contact` | 非法接触（如身体碰地） |
| `bad_orientation` | 姿态倾斜超过阈值 |
| `bad_anchor_pos_z_only` | 锚点高度偏差过大（用于模仿任务） |
| `bad_anchor_ori` | 锚点朝向偏差过大 |
| `bad_motion_body_pos_z_only` | 身体部位高度偏差过大 |

---

## 九、事件与随机化配置

**配置文件位置**: `velocity_env_cfg.py` 中的 `EventCfg`

事件配置用于域随机化，提高策略的鲁棒性。

### 9.1 启动时随机化（startup）

```python
from isaaclab.managers import EventTermCfg as EventTerm

@configclass
class EventCfg:
    # 随机化物理材质（摩擦力）
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 64,
        },
    )
    
    # 随机化基座质量
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),  # 增减质量范围 (kg)
            "operation": "add",
        },
    )
```

### 9.2 重置时随机化（reset）

```python
    # 外部力/力矩扰动
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )
    
    # 随机初始位姿
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )
    
    # 随机初始关节状态
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )
```

### 9.3 周期性事件（interval）

```python
    # 周期性推力扰动
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),  # 每5-10秒触发一次
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )
```

---

## 十、课程学习配置

**配置文件位置**: `velocity_env_cfg.py` 中的 `CurriculumCfg`

课程学习可以逐步增加训练难度。

```python
from isaaclab.managers import CurriculumTermCfg as CurrTerm

@configclass
class CurriculumCfg:
    # 地形难度课程
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    
    # 速度命令课程
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)
```

### 课程学习说明

- `terrain_levels`: 根据机器人表现自动调整地形难度
- `lin_vel_cmd_levels`: 逐步增加速度命令范围

要禁用课程学习，设置 `curriculum = None`。

---

## 十一、创建自定义训练任务

### 11.1 创建新任务的步骤

1. **复制现有配置文件**
   ```
   unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py
   ```
   复制到新目录，如 `my_robot/my_task_env_cfg.py`

2. **修改环境配置**
   - 调整 `RewardsCfg` 中的奖励权重
   - 修改 `ObservationsCfg` 添加/删除观测
   - 配置 `TerminationsCfg` 终止条件
   - 调整 `EventCfg` 域随机化参数

3. **注册新任务**
   在 `__init__.py` 中添加：
   ```python
   import gymnasium as gym
   
   gym.register(
       id="Unitree-MyRobot-MyTask-v0",
       entry_point="isaaclab.envs:ManagerBasedRLEnv",
       kwargs={
           "env_cfg_entry_point": "path.to.my_task_env_cfg:MyRobotEnvCfg",
           "rsl_rl_cfg_entry_point": "path.to.rsl_rl_ppo_cfg:MyPPORunnerCfg",
       },
   )
   ```

4. **运行训练**
   ```powershell
   E:\isaacenv\python.exe scripts/rsl_rl/train.py --headless --task Unitree-MyRobot-MyTask-v0
   ```

### 11.2 常见自定义场景示例

#### 示例1：增加速度跟踪权重
```python
@configclass
class RewardsCfg:
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=3.0,  # 原来是1.5，增加到3.0
        params={"command_name": "base_velocity", "std": 0.25}
    )
```

#### 示例2：添加步态周期奖励
```python
    feet_gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "period": 0.5,           # 步态周期0.5秒
            "offset": [0.0, 0.5],    # 两脚相位差
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot"]),
            "threshold": 0.5,
            "command_name": "base_velocity"
        },
    )
```

#### 示例3：仅使用平坦地形
```python
@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",  # 改为平面
        # terrain_generator=None,  # 不使用地形生成器
        ...
    )
```

#### 示例4：增加域随机化强度
```python
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-2.0, 5.0),  # 更大的质量变化范围
            "operation": "add",
        },
    )
    
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),  # 更频繁的推力
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},  # 更强的推力
    )
```

### 11.3 调试技巧

1. **减少环境数量以便调试**
   ```powershell
   --num_envs 64
   ```

2. **开启可视化**
   去掉 `--headless` 参数可以看到仿真画面

3. **查看奖励分解**
   训练日志会显示各项奖励的分解值

4. **从检查点恢复训练**
   ```powershell
   --resume --load_run <run_name>
   ```

---

## 十二、环境总配置示例

以下是一个完整的环境配置类示例：

```python
@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # 控制频率设置
        self.decimation = 4              # 动作重复次数
        self.episode_length_s = 20.0     # episode最大时长（秒）
        
        # 仿真设置
        self.sim.dt = 0.005              # 仿真时间步长
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        
        # 传感器更新周期
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
```

---

## 附录：常用命令行参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--task` | 任务名称 | `Unitree-Go2-Velocity` |
| `--num_envs` | 并行环境数量 | `4096` |
| `--max_iterations` | 最大训练迭代数 | `10000` |
| `--headless` | 无界面模式 | - |
| `--video` | 录制训练视频 | - |
| `--resume` | 从检查点恢复 | - |
| `--load_run` | 加载指定运行 | `2024-01-01_12-00-00` |
| `--seed` | 随机种子 | `42` |

---

## 参考文件路径

| 配置类型 | 文件路径 |
|----------|----------|
| PPO算法 | `tasks/locomotion/agents/rsl_rl_ppo_cfg.py` |
| Go2环境 | `tasks/locomotion/robots/go2/velocity_env_cfg.py` |
| G1-29dof环境 | `tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py` |
| H1环境 | `tasks/locomotion/robots/h1/velocity_env_cfg.py` |
| Mimic任务 | `tasks/mimic/robots/g1_29dof/gangnanm_style/tracking_env_cfg.py` |
| 奖励函数 | `tasks/locomotion/mdp/rewards.py` |
| 训练脚本 | `scripts/rsl_rl/train.py` |
| 推理脚本 | `scripts/rsl_rl/play.py` |

