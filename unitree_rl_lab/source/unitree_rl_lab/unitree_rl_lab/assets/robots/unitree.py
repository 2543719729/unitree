# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree 机器人配置文件

本模块定义了所有 Unitree 机器人的物理模型配置，包括：
- Go2/Go2W: 四足机器人
- B2: 重型四足机器人
- H1: 人形机器人（20自由度）
- G1-23DOF: 简化版人形机器人（23自由度）
- G1-29DOF: 完整版人形机器人（29自由度）
- G1-29DOF-MIMIC: 模仿学习专用配置

主要功能：
1. 定义机器人的初始状态（位置、关节角度）
2. 配置执行器参数（刚度、阻尼、扭矩限制）
3. 设置物理属性（自碰撞、求解器参数）
4. 提供 SDK 关节名称映射（用于实体机器人部署）

参考: https://github.com/unitreerobotics/unitree_ros
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots import unitree_actuators

# ============================================================================
#                           路径配置
# ============================================================================

UNITREE_MODEL_DIR = "E:/Aunitree/unitree_model"  # USD 模型文件目录（Omniverse 原生格式）
UNITREE_ROS_DIR = "E:/Aunitree/unitree_ros"      # URDF 模型文件目录（ROS 标准格式）

# 注意：请根据实际安装路径修改上述目录


# ============================================================================
#                           基础配置类
# ============================================================================

@configclass
class UnitreeArticulationCfg(ArticulationCfg):
    """Unitree 关节机器人配置基类
    
    扩展了 Isaac Lab 的 ArticulationCfg，添加 Unitree 特有的配置项：
    - joint_sdk_names: SDK 关节名称映射（用于实体机器人通信）
    - soft_joint_pos_limit_factor: 软关节限制因子
    """

    joint_sdk_names: list[str] = None
    """SDK 关节名称列表
    
    用于将仿真中的关节映射到实体机器人的 SDK 接口
    顺序必须与 Unitree SDK 的关节索引一致
    示例: ["left_hip_pitch_joint", "left_hip_roll_joint", ...]
    """

    soft_joint_pos_limit_factor = 0.9
    """软关节位置限制因子
    
    实际限制 = 硬限制 * soft_joint_pos_limit_factor
    默认 0.9 表示使用硬限制的 90%，留 10% 安全余量
    这样可以避免机器人碰到机械限位
    """


@configclass
class UnitreeUsdFileCfg(sim_utils.UsdFileCfg):
    """Unitree USD 模型加载配置
    
    USD (Universal Scene Description) 是 Omniverse 的原生格式
    优点：
    - 渲染性能好
    - 物理仿真精确
    - 支持 GPU 加速
    
    适用于：高保真度仿真训练
    """
    
    activate_contact_sensors: bool = True
    """激活接触传感器
    
    用于检测机器人与环境的接触力，是计算接触奖励的基础
    """
    
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,              # 启用重力
        retain_accelerations=False,         # 不保留加速度历史（节省内存）
        linear_damping=0.0,                 # 线性阻尼（0表示无空气阻力）
        angular_damping=0.0,                # 角阻尼
        max_linear_velocity=1000.0,         # 最大线速度 (m/s)，防止数值爆炸
        max_angular_velocity=1000.0,        # 最大角速度 (rad/s)
        max_depenetration_velocity=1.0,     # 最大去穿透速度，防止碰撞抖动
    )
    """刚体物理属性配置"""
    
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,          # 启用自碰撞检测（如腿与腿）
        solver_position_iteration_count=8,     # 位置求解器迭代次数（影响精度）
        solver_velocity_iteration_count=4      # 速度求解器迭代次数
    )
    """关节机器人根属性配置
    
    更多迭代次数 = 更高精度 + 更多计算时间
    当前设置在精度和性能间取得平衡
    """


@configclass
class UnitreeUrdfFileCfg(sim_utils.UrdfFileCfg):
    """Unitree URDF 模型加载配置
    
    URDF (Unified Robot Description Format) 是 ROS 的标准格式
    优点：
    - 易于编辑和修改
    - 与 ROS 生态兼容
    - 文本格式，便于版本控制
    
    适用于：快速原型开发、ROS 集成
    """
    
    fix_base: bool = False
    """是否固定基座
    
    False: 机器人可以自由移动（正常模式）
    True: 机器人基座固定在空中（调试模式）
    """
    
    activate_contact_sensors: bool = True
    """激活接触传感器"""
    
    replace_cylinders_with_capsules = True
    """用胶囊体替换圆柱体
    
    胶囊体碰撞检测更快更稳定
    适合机器人腿部等细长结构
    """
    
    joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=0,  # 刚度设为0，表示不使用 URDF 内置的驱动
            damping=0     # 阻尼设为0，我们将使用自定义的执行器模型
        )
    )
    """关节驱动配置
    
    设为0表示禁用 URDF 的内置驱动，改用 Isaac Lab 的执行器模型
    这样可以使用更精确的 Unitree 执行器物理模型
    """
    
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,          # 启用自碰撞
        solver_position_iteration_count=8,     # 位置求解器迭代8次
        solver_velocity_iteration_count=4,     # 速度求解器迭代4次
    )
    """关节机器人属性（同USD配置）"""
    
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,              # 启用重力
        retain_accelerations=False,         # 不保留加速度
        linear_damping=0.0,                 # 无线性阻尼
        angular_damping=0.0,                # 无角阻尼
        max_linear_velocity=1000.0,         # 最大线速度限制
        max_angular_velocity=1000.0,        # 最大角速度限制
        max_depenetration_velocity=1.0,     # 去穿透速度限制
    )
    """刚体物理属性（同USD配置）"""

    def replace_asset(self, meshes_dir, urdf_path):
        """替换资产文件，使用临时副本避免修改原始文件
        
        使用场景：
        当需要修改碰撞体时，将修改后的 URDF 文件单独放置，
        mesh 目录仍使用 unitree_ros 提供的原始 mesh。
        
        工作原理：
        1. 在 /tmp 目录创建临时的 robot_description 结构
        2. 使用符号链接链接到实际的 mesh 目录
        3. 使用符号链接链接到修改后的 URDF 文件
        
        注意：
        URDF 内部的 mesh 引用路径应该与 URDF 文件在同一目录层级
        
        Args:
            meshes_dir: mesh 文件目录（通常来自 unitree_ros）
            urdf_path: 修改后的 URDF 文件路径
        """
        tmp_meshes_dir = "/tmp/IsaacLab/unitree_rl_lab/meshes"
        if os.path.exists(tmp_meshes_dir):
            os.remove(tmp_meshes_dir)
        os.makedirs("/tmp/IsaacLab/unitree_rl_lab", exist_ok=True)
        os.symlink(meshes_dir, tmp_meshes_dir)

        self.asset_path = "/tmp/IsaacLab/unitree_rl_lab/robot.urdf"
        if os.path.exists(self.asset_path):
            os.remove(self.asset_path)
        os.symlink(urdf_path, self.asset_path)


# ============================================================================
#                           Unitree 机器人配置
# ============================================================================
# 以下定义了所有 Unitree 机器人型号的详细配置
# 每个配置包括：
# 1. 模型加载方式（USD 或 URDF）
# 2. 初始状态（位置、关节角度）
# 3. 执行器配置（扭矩、速度限制、PD参数）
# 4. SDK 关节名称映射
# ============================================================================

# ========== Go2 四足机器人 ==========
UNITREE_GO2_CFG = UnitreeArticulationCfg(
    # 模型加载方式：使用 USD 格式（更好的渲染性能）
    # 也可以使用 URDF 格式（注释掉的部分）
    # spawn=UnitreeUrdfFileCfg(
    #     asset_path=f"{UNITREE_ROS_DIR}/robots/go2_description/urdf/go2_description.urdf",
    # ),
    spawn=UnitreeUsdFileCfg(
        usd_path=f"{UNITREE_MODEL_DIR}/Go2/usd/go2.usd",
    ),
    
    # 初始状态配置
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),  # 初始位置: (前x, 前y, 高度0.4m)
        joint_pos={
            # 髋关节：左右分开
            ".*R_hip_joint": -0.1,  # 右侧髋关节: -0.1 rad
            ".*L_hip_joint": 0.1,   # 左侧髋关节: 0.1 rad
            # 大腿关节：前腿和后腿角度不同
            "F[L,R]_thigh_joint": 0.8,   # 前腿: 0.8 rad
            "R[L,R]_thigh_joint": 1.0,   # 后腿: 1.0 rad (更大弯曲)
            # 小腿关节：所有腿统一
            ".*_calf_joint": -1.5,       # 小腿: -1.5 rad (向后弯曲)
        },
        joint_vel={".*": 0.0},  # 所有关节初始速度为0
    ),
    
    # 执行器配置：所有关节使用相同的 Go2HV 执行器
    actuators={
        "GO2HV": unitree_actuators.UnitreeActuatorCfg_Go2HV(
            joint_names_expr=[".*"],  # 匹配所有关节
            stiffness=25.0,            # PD 控制器刚度: 25 N·m/rad
            damping=0.5,               # PD 控制器阻尼: 0.5 N·m·s/rad
            friction=0.01,             # 关节摩擦: 0.01 N·m
        ),
    },
    
    # SDK 关节名称映射（用于实体机器人通信）
    # 顺序：FR(前右) -> FL(前左) -> RR(后右) -> RL(后左)
    # 每条腿 3 个关节: hip(髋) -> thigh(大腿) -> calf(小腿)
    # 总计 12 DOF
    # fmt: off
    joint_sdk_names=[
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",  # 前右腿
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",  # 前左腿
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",  # 后右腿
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"   # 后左腿
    ],
    # fmt: on
)

UNITREE_GO2W_CFG = UnitreeArticulationCfg(
    # spawn=UnitreeUrdfFileCfg(
    #     asset_path=f"{UNITREE_ROS_DIR}/robots/go2w_description/urdf/go2w_description.urdf",
    # ),
    spawn=UnitreeUsdFileCfg(
        usd_path=f"{UNITREE_MODEL_DIR}/Go2W/usd/go2w.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            "F.*_thigh_joint": 0.8,
            "R.*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "GO2HV": IdealPDActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=23.5,
            velocity_limit=30.0,
            stiffness={
                ".*_hip_.*": 25.0,
                ".*_thigh_.*": 25.0,
                ".*_calf_.*": 25.0,
                ".*_foot_.*": 0,
            },
            damping=0.5,
            friction=0.01,
        ),
    },
    # fmt: off
    joint_sdk_names=[
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint"
    ],
    # fmt: on
)

UNITREE_B2_CFG = UnitreeArticulationCfg(
    # spawn=UnitreeUrdfFileCfg(
    #     asset_path=f"{UNITREE_ROS_DIR}/robots/b2_description/urdf/b2_description.urdf",
    # ),
    spawn=UnitreeUsdFileCfg(
        usd_path=f"{UNITREE_MODEL_DIR}/B2/usd/b2.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.58),
        joint_pos={
            ".*R_hip_joint": -0.1,
            ".*L_hip_joint": 0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "M107-24-2": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_.*", ".*_thigh_.*"],
            effort_limit=200,
            velocity_limit=23,
            stiffness=160.0,
            damping=5.0,
            friction=0.01,
        ),
        "2": IdealPDActuatorCfg(
            joint_names_expr=[".*_calf_.*"],
            effort_limit=320,
            velocity_limit=14,
            stiffness=160.0,
            damping=5.0,
            friction=0.01,
        ),
    },
    joint_sdk_names=UNITREE_GO2_CFG.joint_sdk_names.copy(),
)

UNITREE_H1_CFG = UnitreeArticulationCfg(
    # spawn=UnitreeUrdfFileCfg(
    #     asset_path=f"{UNITREE_ROS_DIR}/robots/h1_description/urdf/h1.urdf",
    # ),
    spawn=UnitreeUsdFileCfg(
        usd_path=f"{UNITREE_MODEL_DIR}/H1/h1/usd/h1.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.1),
        joint_pos={
            ".*_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.20,
            ".*_elbow_joint": 0.32,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "GO2HV-1": IdealPDActuatorCfg(
            joint_names_expr=[".*ankle.*", ".*_shoulder_pitch_.*", ".*_shoulder_roll_.*"],
            effort_limit=40,
            velocity_limit=9,
            stiffness={
                ".*ankle.*": 40.0,
                ".*_shoulder_.*": 100.0,
            },
            damping=2.0,
            armature=0.01,
        ),
        "GO2HV-2": IdealPDActuatorCfg(
            joint_names_expr=[".*_shoulder_yaw_.*", ".*_elbow_.*"],
            effort_limit=18,
            velocity_limit=20,
            stiffness=50,
            damping=2.0,
            armature=0.01,
        ),
        "M107-24-1": IdealPDActuatorCfg(
            joint_names_expr=[".*_knee_.*"],
            effort_limit=300.0,
            velocity_limit=14.0,
            stiffness=200.0,
            damping=4.0,
            armature=0.01,
        ),
        "M107-24-2": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_.*", "torso_joint"],
            effort_limit=200,
            velocity_limit=23.0,
            stiffness={
                ".*_hip_.*": 150.0,
                "torso_joint": 300.0,
            },
            damping={
                ".*_hip_.*": 2.0,
                "torso_joint": 6.0,
            },
            armature=0.01,
        ),
    },
    joint_sdk_names=[
        "right_hip_roll_joint",
        "right_hip_pitch_joint",
        "right_knee_joint",
        "left_hip_roll_joint",
        "left_hip_pitch_joint",
        "left_knee_joint",
        "torso_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "",
        "left_ankle_joint",
        "right_ankle_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
    ],
)

UNITREE_G1_23DOF_CFG = UnitreeArticulationCfg(
    # spawn=UnitreeUrdfFileCfg(
    #     asset_path=f"{UNITREE_ROS_DIR}/robots/g1_description/g1_23dof_rev_1_0.urdf",
    # ),
    spawn=UnitreeUsdFileCfg(
        usd_path=f"{UNITREE_MODEL_DIR}/G1/23dof/usd/g1_23dof_rev_1_0/g1_23dof_rev_1_0.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            ".*_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.25,
            "right_shoulder_roll_joint": -0.25,
            ".*_elbow_joint": 0.97,
            "left_wrist_roll_joint": 0.15,
            "right_wrist_roll_joint": -0.15,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "N7520-14.3": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_.*", ".*_hip_yaw_.*", "waist_yaw_joint"],  # 5
            effort_limit_sim=88,
            velocity_limit_sim=32.0,
            stiffness={
                ".*_hip_.*": 100.0,
                "waist_yaw_joint": 200.0,
            },
            damping={
                ".*_hip_.*": 2.0,
                "waist_yaw_joint": 5.0,
            },
            armature=0.01,
        ),
        "N7520-22.5": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll_.*", ".*_knee_.*"],  # 4
            effort_limit_sim=139,
            velocity_limit_sim=20.0,
            stiffness={
                ".*_hip_roll_.*": 100.0,
                ".*_knee_.*": 150.0,
            },
            damping={
                ".*_hip_roll_.*": 2.0,
                ".*_knee_.*": 4.0,
            },
            armature=0.01,
        ),
        "N5020-16": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_roll_.*"],  # 10
            effort_limit_sim=25,
            velocity_limit_sim=37,
            stiffness=40.0,
            damping=1.0,
            armature=0.01,
        ),
        "N5020-16-parallel": ImplicitActuatorCfg(
            joint_names_expr=[".*ankle.*"],  # 4
            effort_limit_sim=35,
            velocity_limit_sim=30,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
        ),
    },
    joint_sdk_names=[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "",
        "",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "",
        "",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
    ],
)

UNITREE_G1_29DOF_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUrdfFileCfg(
        asset_path=f"{UNITREE_ROS_DIR}/robots/g1_description/g1_29dof_rev_1_0.urdf",
    ),
    # spawn=UnitreeUsdFileCfg(
    #     usd_path=f"{UNITREE_MODEL_DIR}/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd",
    # ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            "left_hip_pitch_joint": -0.1,
            "right_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.25,
            "right_shoulder_roll_joint": -0.25,
            ".*_elbow_joint": 0.97,
            "left_wrist_roll_joint": 0.15,
            "right_wrist_roll_joint": -0.15,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "N7520-14.3": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_.*", ".*_hip_yaw_.*", "waist_yaw_joint"],
            effort_limit_sim=88,
            velocity_limit_sim=32.0,
            stiffness={
                ".*_hip_.*": 100.0,
                "waist_yaw_joint": 200.0,
            },
            damping={
                ".*_hip_.*": 2.0,
                "waist_yaw_joint": 5.0,
            },
            armature=0.01,
        ),
        "N7520-22.5": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll_.*", ".*_knee_.*"],
            effort_limit_sim=139,
            velocity_limit_sim=20.0,
            stiffness={
                ".*_hip_roll_.*": 100.0,
                ".*_knee_.*": 150.0,
            },
            damping={
                ".*_hip_roll_.*": 2.0,
                ".*_knee_.*": 4.0,
            },
            armature=0.01,
        ),
        "N5020-16": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_.*",
                ".*_elbow_.*",
                ".*_wrist_roll.*",
                ".*_ankle_.*",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit_sim=25,
            velocity_limit_sim=37,
            stiffness=40.0,
            damping={
                ".*_shoulder_.*": 1.0,
                ".*_elbow_.*": 1.0,
                ".*_wrist_roll.*": 1.0,
                ".*_ankle_.*": 2.0,
                "waist_.*_joint": 5.0,
            },
            armature=0.01,
        ),
        "W4010-25": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_pitch.*", ".*_wrist_yaw.*"],
            effort_limit_sim=5,
            velocity_limit_sim=22,
            stiffness=40.0,
            damping=1.0,
            armature=0.01,
        ),
    },
    joint_sdk_names=[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
)


ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2  # 14.25062309787429
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2  # 40.17923847137318
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2  # 99.09842777666113
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2  # 16.77832748089279

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ  # 0.907222843292423
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ  # 2.5578897650279457
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ  # 6.3088018534966395
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ  # 1.06814150219

UNITREE_G1_29DOF_MIMIC_CFG = UnitreeArticulationCfg(
    # spawn=UnitreeUrdfFileCfg(
    #     asset_path=f"{UNITREE_ROS_DIR}/robots/g1_description/g1_29dof_rev_1_0.urdf",
    # ),
    spawn=UnitreeUsdFileCfg(
        usd_path=f"{UNITREE_MODEL_DIR}/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.76),
        joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": 0.6,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_7520_14,
                ".*_hip_roll_joint": STIFFNESS_7520_22,
                ".*_hip_yaw_joint": STIFFNESS_7520_14,
                ".*_knee_joint": STIFFNESS_7520_22,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_7520_14,
                ".*_hip_roll_joint": DAMPING_7520_22,
                ".*_hip_yaw_joint": DAMPING_7520_14,
                ".*_knee_joint": DAMPING_7520_22,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_7520_14,
                ".*_hip_roll_joint": ARMATURE_7520_22,
                ".*_hip_yaw_joint": ARMATURE_7520_14,
                ".*_knee_joint": ARMATURE_7520_22,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "waist": ImplicitActuatorCfg(
            effort_limit_sim=50,
            velocity_limit_sim=37.0,
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            effort_limit_sim=88,
            velocity_limit_sim=32.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_5020,
                ".*_shoulder_roll_joint": STIFFNESS_5020,
                ".*_shoulder_yaw_joint": STIFFNESS_5020,
                ".*_elbow_joint": STIFFNESS_5020,
                ".*_wrist_roll_joint": STIFFNESS_5020,
                ".*_wrist_pitch_joint": STIFFNESS_4010,
                ".*_wrist_yaw_joint": STIFFNESS_4010,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_5020,
                ".*_shoulder_roll_joint": DAMPING_5020,
                ".*_shoulder_yaw_joint": DAMPING_5020,
                ".*_elbow_joint": DAMPING_5020,
                ".*_wrist_roll_joint": DAMPING_5020,
                ".*_wrist_pitch_joint": DAMPING_4010,
                ".*_wrist_yaw_joint": DAMPING_4010,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_5020,
                ".*_shoulder_roll_joint": ARMATURE_5020,
                ".*_shoulder_yaw_joint": ARMATURE_5020,
                ".*_elbow_joint": ARMATURE_5020,
                ".*_wrist_roll_joint": ARMATURE_5020,
                ".*_wrist_pitch_joint": ARMATURE_4010,
                ".*_wrist_yaw_joint": ARMATURE_4010,
            },
        ),
    },
    joint_sdk_names=[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
)

UNITREE_G1_29DOF_MIMIC_ACTION_SCALE = {}
for a in UNITREE_G1_29DOF_MIMIC_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            UNITREE_G1_29DOF_MIMIC_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
