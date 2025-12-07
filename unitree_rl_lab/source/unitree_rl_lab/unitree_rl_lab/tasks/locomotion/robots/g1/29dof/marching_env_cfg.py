"""
===============================================================================
Unitree G1 29DOF 人形机器人原地踏步强化学习环境配置文件

本文件定义了用于训练 Unitree G1 人形机器人进行原地踏步任务的完整环境配置。
训练目标：
    - 机器人保持原地不动（无前后左右移动）
    - 机器人不进行自转（无yaw旋转）
    - 产生明显的步态动作（抬腿踏步）
    - 保持身体稳定和良好姿态

主要配置模块:
    1. 场景配置 (MarchingSceneCfg): 平地地形、机器人、传感器
    2. 事件配置 (MarchingEventCfg): 域随机化、重置逻辑
    3. 命令配置 (MarchingCommandsCfg): 零速度指令
    4. 动作配置 (MarchingActionsCfg): 关节位置控制
    5. 观测配置 (MarchingObservationsCfg): 策略和评论家观测
    6. 奖励配置 (MarchingRewardsCfg): 原地踏步专用奖励
    7. 终止条件配置 (MarchingTerminationsCfg): Episode终止条件
    8. 环境配置 (MarchingEnvCfg): 整合所有配置的主类
===============================================================================
"""

import math

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
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp


# ============================================================================
#                           地形生成器配置
# ============================================================================
# 原地踏步任务：从平地到不平地的难度递增
MARCHING_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=8,  # 8个难度等级
    num_cols=15,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),  # 从简单到困难
    use_cache=False,
    sub_terrains={
        # 平地（30%）- 最简单
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.30),

        # 轻微粗糙地面（25%）- 小幅度起伏
        "rough_flat_easy": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25,
            noise_range=(0.0, 0.02),  # 最大2cm起伏
            noise_step=0.005,
            border_width=0.25,
        ),

        # 中等粗糙地面（25%）- 中等起伏
        "rough_flat_medium": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25,
            noise_range=(0.0, 0.04),  # 最大4cm起伏
            noise_step=0.005,
            border_width=0.25,
        ),

        # 困难粗糙地面（20%）- 较大起伏
        "rough_flat_hard": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.20,
            noise_range=(0.0, 0.06),  # 最大6cm起伏
            noise_step=0.005,
            border_width=0.25,
        ),
    },
)


# ============================================================================
#                           场景配置类
# ============================================================================
@configclass
class MarchingSceneCfg(InteractiveSceneCfg):
    """
    原地踏步场景配���

    使用从平地到不平地的渐进式地形
    """

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=MARCHING_TERRAIN_CFG,
        max_init_terrain_level=MARCHING_TERRAIN_CFG.num_rows - 1,
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

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# ============================================================================
#                           事件配置类
# ============================================================================
@configclass
class MarchingEventCfg:
    """
    原地踏步事件配置
    """

    # 物理材质随机化
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.2),
            "dynamic_friction_range": (0.5, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # 基座质量随机化
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-0.5, 2.0),
            "operation": "add",
        },
    )

    # 基座状态重置：位置随机，朝向随机
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "yaw": (-3.14, 3.14),
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

    # 关节状态重置
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.8, 1.2),
            "velocity_range": (0.0, 0.0),
        },
    )

    # 推力干扰（降低强度）
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
            }
        },
    )


# ============================================================================
#                           命令配置类
# ============================================================================
@configclass
class MarchingCommandsCfg:
    """
    原地踏步命令配置

    速度命令接近零，鼓励机器人保持原地不动
    """

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=1.0,  # 100%站立环境
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=False,
        # 速度命令范围接近0
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(-3.14, 3.14),
        ),
        # 限制范围也是0
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(-3.14, 3.14),
        ),
    )


# ============================================================================
#                           动作配置类
# ============================================================================
@configclass
class MarchingActionsCfg:
    """原地踏步动作配置"""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.30,
        use_default_offset=True,
    )


# ============================================================================
#                           观测配置类
# ============================================================================
@configclass
class MarchingObservationsCfg:
    """
    原地踏步观测配置

    简化观测，专注于本体感知和关节状态
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """策略网络观测组"""

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            scale=1.0,
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )

        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        # 命令观测（全为0）
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )

        last_action = ObsTerm(func=mdp.last_action)

        gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.9})

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """评论家网络观测组"""

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

        def __post_init__(self):
            self.history_length = 5

    critic: CriticCfg = CriticCfg()


# ============================================================================
#                           奖励配置类
# ============================================================================
@configclass
class MarchingRewardsCfg:
    """
    原地踏步专用奖励配置

    重点：
        1. 保持位置不变（无xy移动）
        2. 保持朝向不变（无yaw旋转）
        3. 鼓励步态动作（抬腿踏步）
        4. 保持身体稳定
    """

    # ========== 核心任务奖励 ==========
    # 强惩罚xy平面移动
    stay_in_place = RewTerm(
        func=mdp.base_lin_vel_xy_l2,
        weight=-3.0,  # 强惩罚
        params={},
    )

    # 惩罚yaw旋转
    no_yaw_rotation = RewTerm(
        func=mdp.ang_vel_z_l2,
        weight=-2.0,
        params={},
    )

    # 鼓励抬脚动作
    foot_clearance = RewTerm(
        func=mdp.feet_air_time,
        weight=1.8,  # 高权重，鼓励明显踏步
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "command_name": "base_velocity",
            "threshold": 0.20,  # 适度降低阈值，鼓励更高抬脚但减少跳跃
        },
    )

    # 鼓励左右交替节奏
    foot_symmetry = RewTerm(
        func=mdp.feet_gait,
        weight=0.8,
        params={
            "period": 0.9,
            "offset": [0.0, 0.5],  # 左右腿交替
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "threshold": 0.6,
        },
    )

    # 惩罚左右脚腾空/着地时间差异过大
    air_time_balance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link")},
    )

    # 鼓励关节运动（避免僵硬）
    joint_motion = RewTerm(
        func=mdp.joint_vel_magnitude,
        weight=0.15,
        params={},
    )

    # 能耗惩罚，约束过度用力
    energy_cost = RewTerm(
        func=mdp.energy,
        weight=-2.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # ========== 存活奖励 ==========
    alive = RewTerm(func=mdp.is_alive, weight=0.5)

    # ========== 姿态奖励 ==========
    # 保持躯干水平
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0)

    # 保持合适高度
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "target_height": 0.72,
        },
    )

    # ========== 正则化惩罚 ==========
    # 惩罚垂直速度（减少弹跳）
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)

    # 惩罚roll/pitch角速度
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)

    # 惩罚动作变化率
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # 惩罚关节加速度
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # ========== 关节偏差惩罚 ==========
    # 手臂保持默认姿态
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )

    # 躯干保持直立
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["waist.*"],
            )
        },
    )

    # ========== 安全惩罚 ==========
    # 惩罚非期望碰撞
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*hip.*_link", ".*knee_link", "torso_link"]),
            "threshold": 1.0,
        },
    )

    # 惩罚脚部撞击垂直面（防止乱踢导致漂移）
    stumble_penalty = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link")},
    )


# ============================================================================
#                           终止条件配置类
# ============================================================================
@configclass
class MarchingTerminationsCfg:
    """原地踏步终止条件配置"""

    # 超时终止
    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )

    # 高度过低终止
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.25},
    )

    # 姿态异常终止
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.7},
    )


# ============================================================================
#                           课程学习配置类
# ============================================================================
@configclass
class MarchingCurriculumCfg:
    """
    原地踏步课程学习配置

    根据性能逐步增加地形难度
    """

    # 地形难度课程：根据机器人性能自动调整地形难度等级
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


# ============================================================================
#                           主环境配置类
# ============================================================================
@configclass
class MarchingEnvCfg(ManagerBasedRLEnvCfg):
    """
    原地踏步环境主配置类

    训练机器人原地踏步：
        - 保持位置不变
        - 保持朝向不变
        - 产生明显的步态动作
    """

    scene: MarchingSceneCfg = MarchingSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: MarchingObservationsCfg = MarchingObservationsCfg()
    actions: MarchingActionsCfg = MarchingActionsCfg()
    commands: MarchingCommandsCfg = MarchingCommandsCfg()
    rewards: MarchingRewardsCfg = MarchingRewardsCfg()
    terminations: MarchingTerminationsCfg = MarchingTerminationsCfg()
    events: MarchingEventCfg = MarchingEventCfg()
    curriculum: MarchingCurriculumCfg = MarchingCurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        self.scene.contact_forces.update_period = self.sim.dt

        # 启用课程学习
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class MarchingPlayEnvCfg(MarchingEnvCfg):
    """原地踏步演示环境配置"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 10
