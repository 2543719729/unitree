"""
===============================================================================
Unitree G1 29DOF 人形机器人上楼梯强化学习环境配置文件

本文件定义了用于训练 Unitree G1 人形机器人进行上楼梯任务的完整环境配置。
基于 velocity_env_cfg.py 修改，主要变化包括:
    1. 楼梯地形配置 (STAIR_TERRAIN_CFG): 使用 MeshPyramidStairsTerrainCfg
    2. 新增高度扫描观测 (height_scan): 用于感知楼梯地形
    3. 楼梯专用奖励函数: 向上进展奖励、相对高度惩罚等
    4. 调整命令空间: 限制横向和旋转速度，专注于前进上楼
    5. 楼梯专用课程学习: 从低阶梯到高阶梯
===============================================================================
"""

import math

# ======================== Isaac Lab 核心模块导入 ========================
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# ======================== 自定义模块导入 ========================
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp

# ============================================================================
#                           楼梯地形生成器配置
# ============================================================================
# 使用 MeshInvertedPyramidStairsTerrainCfg（倒金字塔）
# 机器人从边缘（底部）出生，向中心（顶部）攀爬
# 这样符合"上楼梯"的逻辑：从低处向高处爬
STAIR_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),              # 每个地形块尺寸
    border_width=20.0,            # 边界宽度
    num_rows=10,                  # 行数（难度等级）
    num_cols=20,                  # 列数（每个难度的变体）
    horizontal_scale=0.1,         # 水平分辨率
    vertical_scale=0.005,         # 垂直分辨率
    slope_threshold=0.75,         # 斜坡阈值
    difficulty_range=(0.0, 1.0),  # 难度范围
    use_cache=False,              # 不使用缓存
    sub_terrains={
        # 基础平地（15%）- 用于学习基础平衡
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.15),

        # ========== 上楼梯（使用 InvertedPyramid，从边缘向中心爬升）==========
        # 简单上楼梯（30%）- 低阶高，宽踏面
        "stairs_up_easy": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.30,
            step_height_range=(0.08, 0.12),  # 8-12cm 阶高
            step_width=0.35,                  # 35cm 踏面宽度
            platform_width=2.0,               # 2m 顶部平台
            border_width=1.0,                 # 1m 边界
            holes=False,
        ),

        # 中等上楼梯（25%）- 标准室内楼梯
        "stairs_up_medium": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.12, 0.18),  # 12-18cm 阶高
            step_width=0.30,                  # 30cm 踏面宽度
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),

        # 困难上楼梯（15%）- 较高阶梯
        "stairs_up_hard": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.16, 0.20),  # 16-20cm 阶高
            step_width=0.28,                  # 28cm 踏面宽度
            platform_width=1.5,
            border_width=1.0,
            holes=False,
        ),

        # ========== 下楼梯（使用正金字塔，从中心向边缘下降）==========
        # 下楼梯（15%）- 用于学习下楼能力
        "stairs_down": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.10, 0.15),  # 10-15cm 阶高
            step_width=0.32,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


# ============================================================================
#                           楼梯场景配置类（带高度扫描）
# ============================================================================
@configclass
class StairSceneCfg(InteractiveSceneCfg):
    """
    楼梯场景配置类（带 height_scanner）

    用于阶段4：带传感器爬楼梯
    """

    # ======================== 地形配置 ========================
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=STAIR_TERRAIN_CFG,
        max_init_terrain_level=STAIR_TERRAIN_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # ======================== 机器人配置 ========================
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # ======================== 传感器配置 ========================
    # 高度扫描器：用于感知前方楼梯地形
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.2, 0.0, 0.8)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # 接触力传感器
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # ======================== 灯光配置 ========================
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# ============================================================================
#                           盲爬楼梯场景配置类（无高度扫描）
# ============================================================================
@configclass
class StairBlindSceneCfg(InteractiveSceneCfg):
    """
    盲爬楼梯场景配置类（无 height_scanner）

    用于阶段3：盲爬楼梯
    只依赖本体感知（关节角度、速度、IMU），不使用地形感知
    """

    # ======================== 地形配置 ========================
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=STAIR_TERRAIN_CFG,
        max_init_terrain_level=STAIR_TERRAIN_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # ======================== 机器人配置 ========================
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # ======================== 传感器配置 ========================
    # 盲爬模式：不使用 height_scanner，只用接触力传感器
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # ======================== 灯光配置 ========================
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# ============================================================================
#                           事件配置类（域随机化）
# ============================================================================
@configclass
class StairEventCfg:
    """
    楼梯任务事件配置类

    相比平地任务，降低了推力干扰强度，避免机器人在楼梯上被推倒
    """

    # ======================== 启动时事件 ========================
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.0),   # 楼梯摩擦变化更大
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # ======================== 重置时事件 ========================
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),  # 减小偏航角范围，让机器人更多面向楼梯
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-0.5, 0.5),  # 减小初始关节速度
        },
    )

    # ======================== 间隔事件 ========================
    # 楼梯上减小推力干扰
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),  # 降低推力频率
        params={
            "velocity_range": {
                "x": (-0.3, 0.3),  # 减小推力强度
                "y": (-0.3, 0.3),
            }
        },
    )


# ============================================================================
#                           命令配置类
# ============================================================================
@configclass
class StairCommandsCfg:
    """
    楼梯任务命令配置类

    专注于前进方向，限制横向和旋转速度
    """

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.05,  # 5% 站立环境
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        # 初始速度范围：小范围，便于学习
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.1, 0.3),    # 前进为主
            lin_vel_y=(-0.05, 0.05), # 限制横向
            ang_vel_z=(-0.1, 0.1),   # 限制转向
        ),
        # 最终速度范围：楼梯上不需要太快
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.2, 0.6),    # 前进速度
            lin_vel_y=(-0.1, 0.1),   # 小幅横向
            ang_vel_z=(-0.15, 0.15), # 小幅转向
        ),
    )


# ============================================================================
#                           动作配置类
# ============================================================================
@configclass
class StairActionsCfg:
    """楼梯任务动作配置类"""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )


# ============================================================================
#                     盲爬楼梯观测配置类（无 height_scan）
# ============================================================================
@configclass
class StairBlindObservationsCfg:
    """
    盲爬楼梯观测配置类（阶段3）

    只使用本体感知信息，不使用 height_scan
    参考 Cassie 盲爬楼梯论文：只靠触觉反馈自适应调整
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """策略网络观测组 - 纯本体感知"""

        # 基座角速度
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )

        # 重力投影（关键：感知身体倾斜）
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        # 速度命令
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        # 关节位置偏差
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        # 关节速度
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )

        # 上一步动作
        last_action = ObsTerm(func=mdp.last_action)

        # 盲爬模式：不使用 height_scan

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """评论家网络观测组（特权信息，可包含真实速度）"""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        # 盲爬模式：评论家也不使用 height_scan

        def __post_init__(self):
            self.history_length = 5

    critic: CriticCfg = CriticCfg()


# ============================================================================
#                     带传感器楼梯观测配置类（含 height_scan）
# ============================================================================
@configclass
class StairObservationsCfg:
    """
    带传感器楼梯观测配置类（阶段4）

    包含 height_scan 观测，用于感知前方楼梯地形
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """策略网络观测组 - 包含高度扫描"""

        # 基座角速度
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )

        # 重力投影
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        # 速度命令
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        # 关节位置偏差
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        # 关节速度
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )

        # 上一步动作
        last_action = ObsTerm(func=mdp.last_action)

        # 高度扫描 - 楼梯任务关键观测
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),  # 模拟 LiDAR 噪声
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """评论家网络观测组（特权信息）"""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        # 评论家也使用高度扫描（无噪声）
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.history_length = 5

    critic: CriticCfg = CriticCfg()


# ============================================================================
#                     盲爬楼梯奖励配置类（不依赖 height_scanner）
# ============================================================================
@configclass
class StairBlindRewardsCfg:
    """
    盲爬楼梯奖励配置类（阶段3）

    不依赖 height_scanner，使用固定目标高度
    参考 Cassie 论文：奖励函数保持与平地行走相同
    """

    # ====================== 任务奖励 ======================
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # 向上进展奖励 - 楼梯任务核心奖励
    upward_progress = RewTerm(
        func=mdp.upward_progress,
        weight=1.5,
    )

    # ====================== 基座运动正则化 ======================
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    # ====================== 关节运动正则化 ======================
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    # ====================== 关节偏差惩罚 ======================
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*"],
            )
        },
    )

    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.8,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"])},
    )

    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"]
            )
        },
    )

    # ====================== 姿态奖励 ======================
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0)

    # 盲爬模式：使用固定目标高度（不依赖 height_scanner）
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-3.0,  # 降低权重，允许高度变化
        params={"target_height": 0.78},
    )

    # ====================== 步态奖励 ======================
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.2,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.15,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )

    # ====================== 安全惩罚 ======================
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.8,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )


# ============================================================================
#                     带传感器楼梯奖励配置类（依赖 height_scanner）
# ============================================================================
@configclass
class StairRewardsCfg:
    """
    带传感器楼梯奖励配置类（阶段4）

    使用 base_height_adaptive（依赖 height_scanner）来适应地形高度
    """

    # ====================== 任务奖励 ======================
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # 向上进展奖励 - 楼梯任务核心奖励
    upward_progress = RewTerm(
        func=mdp.upward_progress,
        weight=1.5,
    )

    # ====================== 基座运动正则化 ======================
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    # ====================== 关节运动正则化 ======================
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    # ====================== 关节偏差惩罚 ======================
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*"],
            )
        },
    )

    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.8,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"])},
    )

    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"]
            )
        },
    )

    # ====================== 姿态奖励 ======================
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0)

    # 相对高度奖励 - 使用 RayCaster 修正版本
    base_height_relative = RewTerm(
        func=mdp.base_height_adaptive,
        weight=-5.0,
        params={
            "target_height": 0.78,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )

    # ====================== 步态奖励 ======================
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.2,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.15,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )

    # ====================== 安全惩罚 ======================
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.8,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )


# ============================================================================
#                           终止条件配置类
# ============================================================================
@configclass
class StairTerminationsCfg:
    """楼梯任务终止条件配置"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 高度过低终止 - 楼梯上调整阈值
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.25},  # 略微提高，楼梯上更容易跌倒
    )

    # 姿态异常终止
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.7},  # 略微减小，更早终止不稳定状态
    )


# ============================================================================
#                           课程学习配置类
# ============================================================================
@configclass
class StairCurriculumCfg:
    """楼梯任务课程学习配置"""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)


# ============================================================================
#                  阶段3：盲爬楼梯环境配置类（无 height_scan）
# ============================================================================
@configclass
class StairBlindEnvCfg(ManagerBasedRLEnvCfg):
    """
    盲爬楼梯环境配置类（阶段3）

    只使用本体感知，不使用 height_scanner
    参考 Cassie 盲爬楼梯论文的方法
    """

    # ======================== 场景配置 ========================
    scene: StairBlindSceneCfg = StairBlindSceneCfg(num_envs=4096, env_spacing=2.5)

    # ======================== MDP 配置 ========================
    observations: StairBlindObservationsCfg = StairBlindObservationsCfg()
    actions: StairActionsCfg = StairActionsCfg()
    commands: StairCommandsCfg = StairCommandsCfg()

    # ======================== 核心配置 ========================
    rewards: StairBlindRewardsCfg = StairBlindRewardsCfg()
    terminations: StairTerminationsCfg = StairTerminationsCfg()
    events: StairEventCfg = StairEventCfg()
    curriculum: StairCurriculumCfg = StairCurriculumCfg()

    def __post_init__(self):
        """后初始化方法"""
        self.decimation = 4
        self.episode_length_s = 20.0

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        self.scene.contact_forces.update_period = self.sim.dt
        # 盲爬模式没有 height_scanner

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class StairBlindPlayEnvCfg(StairBlindEnvCfg):
    """盲爬楼梯演示环境配置"""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 8

        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


# ============================================================================
#                  阶段4：带传感器楼梯环境配置类（有 height_scan）
# ============================================================================
@configclass
class StairClimbEnvCfg(ManagerBasedRLEnvCfg):
    """
    带传感器楼梯攀爬环境配置类（阶段4）

    使用 height_scanner 感知前方楼梯地形
    """

    # ======================== 场景配置 ========================
    scene: StairSceneCfg = StairSceneCfg(num_envs=4096, env_spacing=2.5)

    # ======================== MDP 配置 ========================
    observations: StairObservationsCfg = StairObservationsCfg()
    actions: StairActionsCfg = StairActionsCfg()
    commands: StairCommandsCfg = StairCommandsCfg()

    # ======================== 核心配置 ========================
    rewards: StairRewardsCfg = StairRewardsCfg()
    terminations: StairTerminationsCfg = StairTerminationsCfg()
    events: StairEventCfg = StairEventCfg()
    curriculum: StairCurriculumCfg = StairCurriculumCfg()

    def __post_init__(self):
        """后初始化方法"""
        self.decimation = 4
        self.episode_length_s = 20.0

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class StairClimbPlayEnvCfg(StairClimbEnvCfg):
    """带传感器楼梯攀爬演示环境配置"""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 8

        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
