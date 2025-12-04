"""
===============================================================================
Unitree G1 29DOF 统一条件策略环境配置文件

本文件实现4模式条件策略训练：
    - 模式0: 平地盲走 (无 height_scan)
    - 模式1: 平地带传感器 (有 height_scan)
    - 模式2: 楼梯盲爬 (无 height_scan)
    - 模式3: 楼梯带传感器 (有 height_scan)

通过 mode_flag 观测让网络学会根据模式调整行为。
训练时随机切换模式和地形，部署时可智能选择模式。
===============================================================================
"""

import math
import torch

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

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp


# ============================================================================
#                           统一地形生成器配置
# ============================================================================
# 混合平地和楼梯地形，用于4模式训练
UNIFIED_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        # ========== 平地区域 (40%) - 模式0/1 ==========
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.25),
        "flat_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15,
            noise_range=(0.0, 0.03),
            noise_step=0.005,
            border_width=0.25,
        ),

        # ========== 楼梯区域 (60%) - 模式2/3 ==========
        # 简单上楼梯
        "stairs_up_easy": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.08, 0.12),
            step_width=0.35,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        # 中等上楼梯
        "stairs_up_medium": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.12, 0.18),
            step_width=0.30,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        # 困难上楼梯
        "stairs_up_hard": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.16, 0.20),
            step_width=0.28,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


# ============================================================================
#                           统一场景配置类
# ============================================================================
@configclass
class UnifiedSceneCfg(InteractiveSceneCfg):
    """
    统一场景配置 - 包含平地和楼梯的混合地形
    """
    
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=UNIFIED_TERRAIN_CFG,
        max_init_terrain_level=UNIFIED_TERRAIN_CFG.num_rows - 1,
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

    # 高度扫描器（模式1/3使用，模式0/2时输出置零）
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.2, 0.0, 0.8)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

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
#                           事件配置类（域随机化）
# ============================================================================
@configclass
class UnifiedEventCfg:
    """统一事件配置 - 包含模式切换逻辑"""

    # 启动时：物理材质随机化
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # 启动时：关节参数随机化
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # 重置时：基座状态初始化
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
            },
        },
    )

    # 重置时：关节状态初始化
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # 重置时：随机选择模式（核心！）
    reset_mode = EventTerm(
        func=mdp.reset_mode_randomly,  # 需要在 mdp 中实现
        mode="reset",
        params={
            "num_modes": 4,
            "mode_probabilities": [0.25, 0.25, 0.25, 0.25],  # 等概率
        },
    )

    # 间隔：外部推力干扰
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )


# ============================================================================
#                           命令配置类
# ============================================================================
@configclass
class UnifiedCommandsCfg:
    """统一命令配置"""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(-math.pi, math.pi),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.5, 0.5),
            heading=(-math.pi, math.pi),
        ),
    )


# ============================================================================
#                           动作配置类
# ============================================================================
@configclass
class UnifiedActionsCfg:
    """统一动作配置 - 关节位置控制"""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )


# ============================================================================
#                           统一观测配置类
# ============================================================================
@configclass
class UnifiedObservationsCfg:
    """
    统一观测配置 - 包含 mode_flag 和条件 height_scan

    观测向量结构:
        [本体感知] + [速度命令] + [mode_flag] + [height_scan(条件)]

    盲模式(0/2)时 height_scan 输出置零
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """策略网络观测组"""

        # 基础角速度
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

        # ========== 条件策略核心：模式标志 ==========
        mode_flag = ObsTerm(
            func=mdp.mode_flag,
            params={"num_modes": 4},
        )

        # ========== 条件 height_scan（盲模式时置零）==========
        height_scan = ObsTerm(
            func=mdp.conditional_height_scan,  # 需要在 mdp 中实现
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
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

        # 模式标志
        mode_flag = ObsTerm(
            func=mdp.mode_flag,
            params={"num_modes": 4},
        )

        # 评论家始终使用 height_scan（无噪声，特权信息）
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.history_length = 5

    critic: CriticCfg = CriticCfg()


# ============================================================================
#                           统一奖励配置类
# ============================================================================
@configclass
class UnifiedRewardsCfg:
    """
    统一奖励配置 - 支持多模式的自适应奖励

    奖励函数根据 mode_flag 自动调整：
        - 模式0/1 (平地): 主要关注速度跟踪
        - 模式2/3 (楼梯): 额外增加上楼梯奖励
    """

    # ========== 任务奖励 ==========
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

    # 模式自适应的向上进展奖励（楼梯模式时激活）
    upward_progress = RewTerm(
        func=mdp.upward_progress,
        weight=0.5,  # 楼梯模式时有效
        params={},
    )

    # ========== 存活奖励 ==========
    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # ========== 正则化惩罚 ==========
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # ========== 姿态奖励 ==========
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link"), "target_height": 0.72},
    )

    # ========== 步态奖励 ==========
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )

    # ========== 安全惩罚 ==========
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*hip.*_link", ".*knee_link"]),
            "threshold": 1.0,
        },
    )


# ============================================================================
#                           终止条件配置类
# ============================================================================
@configclass
class UnifiedTerminationsCfg:
    """统一终止条件配置"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["torso_link", ".*hip.*"]),
            "threshold": 1.0,
        },
    )


# ============================================================================
#                           课程学习配置类
# ============================================================================
@configclass
class UnifiedCurriculumCfg:
    """统一课程学习配置"""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


# ============================================================================
#                           统一环境配置主类
# ============================================================================
@configclass
class UnifiedEnvCfg(ManagerBasedRLEnvCfg):
    """
    统一条件策略环境配置

    训练一个能处理4种模式的策略：
        - 模式0: 平地盲走
        - 模式1: 平地带传感器
        - 模式2: 楼梯盲爬
        - 模式3: 楼梯带传感器
    """

    scene: UnifiedSceneCfg = UnifiedSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: UnifiedObservationsCfg = UnifiedObservationsCfg()
    actions: UnifiedActionsCfg = UnifiedActionsCfg()
    commands: UnifiedCommandsCfg = UnifiedCommandsCfg()
    rewards: UnifiedRewardsCfg = UnifiedRewardsCfg()
    terminations: UnifiedTerminationsCfg = UnifiedTerminationsCfg()
    events: UnifiedEventCfg = UnifiedEventCfg()
    curriculum: UnifiedCurriculumCfg = UnifiedCurriculumCfg()

    def __post_init__(self):
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
class UnifiedPlayEnvCfg(UnifiedEnvCfg):
    """统一环境演示配置"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges

