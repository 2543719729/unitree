"""
===============================================================================
Unitree G1 29DOF 人形机器人盲爬楼梯强化学习环境配置文件

本文件定义了用于训练 Unitree G1 人形机器人进行盲爬楼梯任务的完整环境配置。
盲爬模式：只依赖本体感知（关节角度、速度、IMU），不使用地形传感器（height_scan）

主要特点:
    1. 楼梯地形配置: 使用 MeshInvertedPyramidStairsTerrainCfg（上楼梯）
    2. 纯本体感知: 不使用 height_scan 观测
    3. 楼梯专用奖励函数: 向上进展奖励、固定目标高度惩罚
    4. 调整命令空间: 限制横向和旋转速度，专注于前进上楼
    5. 参考 Cassie 盲爬楼梯论文的方法

适用场景:
    - 学习鲁棒的盲爬能力
    - 即使传感器失效也能保持基本爬楼能力
    - 作为带传感器爬楼梯的预训练阶段
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
from isaaclab.sensors import ContactSensorCfg
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
        # 基础平地（50%）- 大幅增加！先学会站立和行走
        # [修复] 原值 15% 太少，机器人还没学会站立就要学爬楼梯
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.50),

        # ========== 上楼梯（使用 InvertedPyramid，从边缘向中心爬升）==========
        # 简单上楼梯（25%）- 低阶高，宽踏面
        # [修复] 原值 35%，减少到 25%
        "stairs_up_easy": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.08, 0.12),  # 8-12cm 阶高
            step_width=0.35,                  # 35cm 踏面宽度
            platform_width=2.0,               # 2m 顶部平台
            border_width=1.0,                 # 1m 边界
            holes=False,
        ),

        # 中等上楼梯（15%）- 标准室内楼梯
        # [修复] 原值 30%，减少到 15%
        "stairs_up_medium": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.10, 0.16),  # 10-16cm 阶高
            step_width=0.32,                  # 32cm 踏面宽度
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),

        # 困难上楼梯（10%）- 较高阶梯
        # [修复] 原值 20%，减少到 10%
        "stairs_up_hard": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.14, 0.18),  # 14-18cm 阶高
            step_width=0.30,                  # 30cm 踏面宽度
            platform_width=1.5,
            border_width=1.0,
            holes=False,
        ),
    },
)


# ============================================================================
#                           盲爬楼梯场景配置类（无高度扫描）
# ============================================================================
@configclass
class StairBlindSceneCfg(InteractiveSceneCfg):
    """
    盲爬楼梯场景配置类（无 height_scanner）

    只依赖本体感知（关节角度、速度、IMU），不使用地形感知
    参考 Cassie 盲爬楼梯论文的方法
    """

    # ======================== 地形配置 ========================
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=STAIR_TERRAIN_CFG,
        # [修复] 从最简单的地形开始！原值是 num_rows - 1 = 9
        # 这确保机器人先在平地上学会站立和行走，再逐步挑战楼梯
        max_init_terrain_level=0,
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
            "mass_distribution_params": (-2.0, 2.0),
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
            "velocity_range": (-0.0, 0.0),  # 减小初始关节速度
        },
    )

    # ======================== 间隔事件 ========================
    # 楼梯上减小推力干扰（盲爬模式更保守，因为没有地形感知）
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),  # 降低推力频率（比带传感器模式更宽松）
        params={
            "velocity_range": {
                "x": (-0.3, 0.3),  # 减小推力强度（比带传感器模式更小）
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

    # [修复] 减小动作幅度，训练初期避免剧烈动作导致失衡
    # 原值 0.25 太大，随机动作会导致关节大幅移动
    # 0.15 更保守，让机器人有机会学习平衡
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.15,
        use_default_offset=True,
    )


# ============================================================================
#                     盲爬楼梯观测配置类（无 height_scan）
# ============================================================================
@configclass
class StairBlindObservationsCfg:
    """
    盲爬楼梯观测配置类

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
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )

        # 关节速度
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        # 上一步动作
        last_action = ObsTerm(func=mdp.last_action)

        # [修复] 添加步态相位观测，帮助策略学习周期性步态
        # 这在 marching_env_cfg 中存在，对步态学习非常重要
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 1.0})

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
#                     盲爬楼梯奖励配置类（不依赖 height_scanner）
# ============================================================================
@configclass
class StairBlindRewardsCfg:
    """
    盲爬楼梯奖励配置类

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

    # [修复] 大幅增加 alive 奖励权重
    # 原值 2.0 不足以激励存活，机器人没有动力保持平衡
    # 5.0 让存活成为最重要的目标，优先学会站立
    alive = RewTerm(func=mdp.is_alive, weight=5.0)

    # 向上进展奖励 - 楼梯任务核心奖励
    upward_progress = RewTerm(
        func=mdp.upward_progress,
        weight=1.5,
    )

    # ====================== 基座运动正则化 ======================
    # 降低 Z 轴速度惩罚，因为上楼梯时 Z 轴速度自然会增加
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)#惩罚Z轴线速度（上下颠簸
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)#惩罚X/Y轴角速度（左右摇晃）

    # ====================== 关节运动正则化 ======================
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)  #惩罚关节速度
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7) #惩罚关节加速度
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05) #惩罚动作率
    # [修复] 原值 -5.0 太高，训练初期关节容易超限导致大量惩罚
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0) #惩罚关节位置限制
    energy = RewTerm(func=mdp.energy, weight=-2e-5) #惩罚能量消耗

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
    # 增强姿态惩罚，保持躯干直立（替代 base_height_l2 的作用）
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)

    # 盲爬模式：移除 base_height_l2
    # 原因：在楼梯上，机器人的绝对高度会随着攀爬而增加
    # 固定目标高度 0.78m 会在高处产生错误惩罚
    # 通过增强 flat_orientation_l2 和添加膝关节惩罚来间接约束高度

    # 新增：膝关节弯曲惩罚，防止蹲着走
    joint_deviation_knees = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.9,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_joint"])
        },
    )

    # ====================== 步态奖励 ======================
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "period": 1.0,
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

    # 增加抬腿高度，适合跨越台阶
    # [修复] 目标高度从 0.20 降低到 0.12m
    # 原因：过高的目标高度导致机器人重心不稳，容易倾倒
    # 0.12m 足以跨越 8-12cm 的简单楼梯，更高难度的楼梯通过课程学习逐步挑战
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.2,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.20,  # [修复] 从 0.20 降低到 0.12
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

    # [修复] 使用带保护期的终止条件
    # grace_steps=10 表示 episode 开始的前 10 步不触发终止
    # 这给机器人时间从初始姿态稳定下来，避免因初始抖动就终止
    base_height = DoneTerm(
        func=mdp.root_height_below_terrain_minimum_with_grace,
        params={
            "minimum_height": 0.15,  # 相对地形高度阈值
            "grace_steps": 10,       # 保护期 10 步
        },
    )

    # [修复] 使用带保护期的姿态终止条件
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation_with_grace,
        params={
            "limit_angle": 1.3,      # 74° 倾斜角阈值
            "grace_steps": 10,       # 保护期 10 步
        },
    )


# ============================================================================
#                           课程学习配置类
# ============================================================================
@configclass
class StairCurriculumCfg:
    """
    盲爬楼梯任务课程学习配置
    
    [升级] 使用自适应课程学习 adaptive_terrain_levels
    
    功能：
        - 自动检测训练阶段（基于 episode_length 和 terrain_level）
        - 动态调整奖励权重和终止条件参数
        - 根据阶段自动切换底层课程策略（survival/climb）
    
    训练阶段：
        Stage 0: 初始探索期 (mean_episode_length < 100)
        Stage 1: 站立稳定期 (100 ≤ mean_episode_length < 300)
        Stage 2: 行走学习期 (300 ≤ mean_episode_length < 600)
        Stage 3: 楼梯适应期 (mean_episode_length ≥ 600, terrain_level < 3)
        Stage 4: 楼梯精通期 (mean_episode_length ≥ 600, terrain_level ≥ 3)
    
    详细规划文档：docs/adaptive_training_plan.md
    """

    # [升级] 使用自适应课程学习（自动检测阶段并调整参数）
    terrain_levels = CurrTerm(func=mdp.adaptive_terrain_levels)
    
    # 速度命令课程学习（保持不变）
    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)


# ============================================================================
#                  盲爬楼梯环境配置类（无 height_scan）
# ============================================================================
@configclass
class StairBlindEnvCfg(ManagerBasedRLEnvCfg):
    """
    盲爬楼梯环境配置类

    只使用本体感知，不使用 height_scanner
    参考 Cassie 盲爬楼梯论文的方法
    """

    # ======================== 场景配置 ========================
    scene: StairBlindSceneCfg = StairBlindSceneCfg(num_envs=4096, env_spacing=2.5)

    # ======================== MDP 配置 ========================
    #盲爬楼梯观测配置类（无 height_scan）
    observations: StairBlindObservationsCfg = StairBlindObservationsCfg()
    #动作配置类
    actions: StairActionsCfg = StairActionsCfg()
    #命令配置类
    commands: StairCommandsCfg = StairCommandsCfg()

    # ======================== 核心配置 ========================
    #奖励配置类
    rewards: StairBlindRewardsCfg = StairBlindRewardsCfg()
    #终止条件配置类
    terminations: StairTerminationsCfg = StairTerminationsCfg()
    #事件配置类
    events: StairEventCfg = StairEventCfg()
    #课程学习配置类
    curriculum: StairCurriculumCfg = StairCurriculumCfg()

    def __post_init__(self):
        """
        后初始化方法 - 在配置对象创建后自动调用
        
        该方法用于设置仿真的核心参数，包括：
            1. 控制频率和时间步长
            2. Episode 时长
            3. 物理引擎参数
            4. 传感器更新周期
            5. 课程学习开关
        
        这些参数会覆盖父类的默认值，确保环境按预期运行。
        """
        
        # ======================== 控制频率配置 ========================
        # decimation: 降采样因子，控制策略执行频率与物理仿真频率的比例
        #
        # 计算公式：策略频率 = 仿真频率 / decimation
        #
        # 当前配置：
        #   - 仿真频率 = 1 / 0.005 = 200 Hz
        #   - 策略频率 = 200 / 4 = 50 Hz
        #
        # 含义：物理仿真每秒运行200次，但策略网络每秒只执行50次
        #       每次策略输出的动作会被保持4个仿真步
        self.decimation = 4
        
        # ======================== Episode 时长配置 ========================
        # episode_length_s: 每个训练 Episode 的最大时长（秒）
        #
        # 计算：
        #   - 每个策略步的时间 = sim.dt × decimation = 0.005 × 4 = 0.02 秒
        #   - 最大步数 = episode_length_s / 0.02 = 20.0 / 0.02 = 1000 步
        #
        # 影响：
        #   - 值越大，机器人有更多时间完成任务
        #   - 值越小，训练迭代更快，但可能学不到长期行为
        self.episode_length_s = 20.0
        
        # ======================== 物理仿真配置 ========================
        # sim.dt: 物理仿真的时间步长（秒）
        #
        # 当前值 0.005 秒 = 5 毫秒，对应 200 Hz 的仿真频率
        #
        # 影响：
        #   - 值越小，物理仿真越精确，但计算量越大
        #   - 值越大，仿真越快，但可能出现物理不稳定（穿透、抖动）
        #
        # 推荐范围：0.001 ~ 0.01 秒（100 ~ 1000 Hz）
        # 机器人仿真通常使用 0.005 秒（200 Hz）
        self.sim.dt = 0.005
        
        # sim.render_interval: 渲染间隔
        #
        # 设置为 decimation，表示每 decimation 个仿真步渲染一次
        # 即渲染频率与策略频率相同（50 Hz）
        self.sim.render_interval = self.decimation
        
        # sim.physics_material: 默认物理材质
        #
        # 将地形的物理材质（摩擦系数、弹性系数）设置为仿真的默认材质
        # 这确保了机器人与地面接触时使用正确的物理属性
        self.sim.physics_material = self.scene.terrain.physics_material
        
        # sim.physx.gpu_max_rigid_patch_count: PhysX GPU 刚体 patch 数量上限
        #
        # 计算：10 × 2^15 = 10 × 32768 = 327,680
        #
        # 作用：
        #   - 控制 GPU 上可以同时处理的刚体接触点数量
        #   - 大规模并行仿真（如 4096 个环境）需要更大的值
        #
        # 如果出现 "GPU rigid body patch count exceeded" 错误，需要增大此值
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        
        # ======================== 传感器更新周期配置 ========================
        # contact_forces.update_period: 接触力传感器的更新周期
        #
        # 设置为 sim.dt，表示每个仿真步都更新接触力数据
        # 即接触力传感器以 200 Hz 的频率更新
        #
        # 这对于步态检测和碰撞惩罚非常重要，需要高频率的接触信息
        self.scene.contact_forces.update_period = self.sim.dt
        
        # 注意：盲爬模式不使用 height_scanner（高度扫描器）
        # 如果有 height_scanner，通常设置为：
        # self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        # 即以策略频率（50 Hz）更新，因为高度扫描不需要那么高的频率
        
        # ======================== 课程学习配置 ========================
        # 检查是否启用了地形难度课程学习
        #
        # 课程学习的作用：
        #   - 训练初期使用简单地形（如平地）
        #   - 随着策略性能提升，自动切换到更难的地形
        #   - 帮助策略逐步学习，避免一开始就面对困难任务
        #
        # 逻辑说明：
        #   1. 检查 curriculum 配置中是否定义了 terrain_levels
        #   2. 如果定义了，启用地形生成器的课程学习功能
        #   3. 如果没有定义，关闭课程学习，使用固定难度
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            # 如果配置了地形难度课程
            if self.scene.terrain.terrain_generator is not None:
                # 启用地形生成器的课程学习模式
                # 地形生成器会根据机器人性能自动调整难度等级
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            # 如果没有配置地形难度课程
            if self.scene.terrain.terrain_generator is not None:
                # 关闭课程学习，使用固定的地形难度分布
                self.scene.terrain.terrain_generator.curriculum = False


# 用来演示的配置
@configclass
class StairBlindPlayEnvCfg(StairBlindEnvCfg):
    """盲爬楼梯演示环境配置"""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = 8

        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
