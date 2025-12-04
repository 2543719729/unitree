"""
===============================================================================
Unitree G1 29DOF 人形机器人速度跟踪强化学习环境配置文件

本文件定义了用于训练 Unitree G1 人形机器人进行速度跟踪任务的完整环境配置。
主要包含以下配置模块:
    1. 场景配置 (RobotSceneCfg): 地形、机器人、传感器、灯光
    2. 事件配置 (EventCfg): 域随机化、重置逻辑、干扰
    3. 命令配置 (CommandsCfg): 速度指令生成
    4. 动作配置 (ActionsCfg): 关节位置控制动作空间
    5. 观测配置 (ObservationsCfg): 策略网络和评论家网络的观测
    6. 奖励配置 (RewardsCfg): 多项奖励函数定义
    7. 终止条件配置 (TerminationsCfg): Episode终止条件
    8. 课程学习配置 (CurriculumCfg): 训练难度递增策略
    9. 环境配置 (RobotEnvCfg): 整合所有配置的主类
===============================================================================
"""

import math  # 数学库，用于数学计算（如平方根）

# ======================== Isaac Lab 核心模块导入 ========================
import isaaclab.sim as sim_utils  # 仿真工具，包含物理材质、灯光等配置
import isaaclab.terrains as terrain_gen  # 地形生成器模块
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # 资产配置类：机器人关节体、基础资产
from isaaclab.envs import ManagerBasedRLEnvCfg  # 基于Manager的强化学习环境配置基类
from isaaclab.managers import CurriculumTermCfg as CurrTerm  # 课程学习项配置
from isaaclab.managers import EventTermCfg as EventTerm  # 事件项配置（域随机化、重置等）
from isaaclab.managers import ObservationGroupCfg as ObsGroup  # 观测组配置
from isaaclab.managers import ObservationTermCfg as ObsTerm  # 观测项配置
from isaaclab.managers import RewardTermCfg as RewTerm  # 奖励项配置
from isaaclab.managers import SceneEntityCfg  # 场景实体配置（用于引用场景中的对象）
from isaaclab.managers import TerminationTermCfg as DoneTerm  # 终止条件项配置
from isaaclab.scene import InteractiveSceneCfg  # 交互式场景配置基类
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns  # 传感器配置：接触力传感器、射线投射器
from isaaclab.terrains import TerrainImporterCfg  # 地形导入器配置
from isaaclab.utils import configclass  # 配置类装饰器
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR  # 资产目录路径常量
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise  # 加性均匀噪声配置

# ======================== 自定义模块导入 ========================
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG  # G1机器人29自由度配置
from unitree_rl_lab.tasks.locomotion import mdp  # MDP相关函数（奖励、观测、事件等）

# ============================================================================
#                           地形生成器配置
# ============================================================================
# 定义训练时使用的地形配置，支持课程学习中的地形难度递增
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),  # 每个地形块的尺寸 (长度, 宽度)，单位：米
    border_width=20.0,  # 地形边界宽度，防止机器人走出边界，单位：米
    num_rows=9,  # 地形行数，用于课程学习的难度等级数量
    num_cols=21,  # 地形列数，每个难度等级的地形变体数量
    horizontal_scale=0.1,  # 水平分辨率，高度图采样间隔，单位：米   ？？
    vertical_scale=0.005,  # 垂直分辨率，高度值缩放因子，单位：米  ？？
    slope_threshold=0.75,  # 斜坡阈值，用于法线计算和可行走区域判定
    difficulty_range=(0.0, 1.0),  # 难度范围 (最小难度, 最大难度)，用于课程学习
    use_cache=False,  # 是否使用缓存地形，False表示每次重新生成
    sub_terrains={
        # 子地形配置字典，定义不同类型地形及其生成比例
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),  # 平坦地形，占比50%
    },
)


# ============================================================================
#                           场景配置类
# ============================================================================
@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """
    机器人仿真场景配置类
    
    定义了完整的仿真环境，包括:
        - 地形配置：支持平面或程序化生成的复杂地形
        - 机器人配置：Unitree G1 人形机器人
        - 传感器配置：高度扫描器、接触力传感器
        - 灯光配置：天空穹顶灯光
    """
    """
    terrain  
    
    """
    # ======================== 地形配置 ========================
    terrain = TerrainImporterCfg(

        prim_path="/World/ground",  # 地形在USD场景中的路径
        
        terrain_type="generator",  # 地形类型: "plane"(平面) 或 "generator"(程序生成)
        
        terrain_generator=COBBLESTONE_ROAD_CFG,  # 地形生成器配置，None表示使用默认平面
        
        max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,  # 初始化时的最大地形难度等级
        
        collision_group=-1,  # 碰撞组ID，-1表示与所有物体碰撞
        
        # 物理材质配置：定义地形的摩擦和弹性特性
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",  # 摩擦力组合模式：相乘
            restitution_combine_mode="multiply",  # 恢复系数组合模式：相乘
            static_friction=1.0,  # 静摩擦系数
            dynamic_friction=1.0,  # 动摩擦系数
        ),
        
        # 视觉材质配置：地形的外观纹理
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            #NVIDIA MDL 材质文件路径，这是一个白色大理石砖纹理
            project_uvw=True,  # 使用投影UV映射
            texture_scale=(0.25, 0.25),  # 纹理缩放比例
        ),
        
        debug_vis=False,  # 是否显示调试可视化
    )

    # ======================== 机器人配置 ========================
    # 使用预定义的G1机器人配置，并设置其在各环境中的路径
    # {ENV_REGEX_NS} 是环境命名空间的占位符，会被替换为具体环境的路径
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # ======================== 传感器配置 ========================
    # 高度扫描器：用于感知机器人周围地形高度，实现地形感知
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",  # 传感器安装位置：躯干链接
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),  # 射线起点偏移，从高处向下投射
        ray_alignment="yaw",  # 射线对齐方式：跟随机器人偏航角  什么
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),  # 网格采样模式：分辨率0.1m，范围1.6mx1.0m
        debug_vis=False,  # 是否显示射线调试可视化
        mesh_prim_paths=["/World/ground"],  # 射线检测的目标网格路径
    )

    # 接触力传感器：检测机器人各部位的接触力，用于步态检测和碰撞惩罚
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",  # 检测机器人所有部位的接触力
        history_length=3,  # 保存3帧历史数据
        track_air_time=True,  # 追踪腾空时间，用于步态奖励
    )

    # ======================== 灯光配置 ========================
    # 天空穹顶灯光：提供全局环境光照
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",  # 灯光在USD场景中的路径
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,  # 灯光强度
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",  # HDR天空纹理
        ),
    )


# ============================================================================
#                           事件配置类（域随机化）
# ============================================================================
@configclass
class EventCfg:
    """
    事件配置类 - 域随机化和环境重置
    
    包含三种类型的事件:
        1. startup (启动时): 仿真开始时执行一次，用于初始化随机化
        2. reset (重置时): 每次Episode重置时执行，用于状态初始化
        3. interval (间隔): 按时间间隔周期性执行，用于持续干扰
    
    域随机化的目的是提高策略的泛化能力，使其能适应真实世界的不确定性。
    """

    # ======================== 启动时事件 (startup) ========================
    # 这些事件在仿真开始时执行一次，为每个环境设置不同的物理参数

    # 物理材质随机化：随机化机器人各部位的摩擦系数
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # 随机化刚体材质的函数
        mode="startup",  # 启动时执行
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),  # 目标：机器人所有body
            "static_friction_range": (0.3, 1.0),  # 静摩擦系数范围
            "dynamic_friction_range": (0.3, 1.0),  # 动摩擦系数范围
            "restitution_range": (0.0, 0.0),  # 恢复系数范围（弹性）
            "num_buckets": 64,  # 离散化桶数，用于GPU优化
        },
    )

    # 基座质量随机化：在躯干上添加随机额外质量，模拟负载变化
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,  # 随机化刚体质量的函数
        mode="startup",  # 启动时执行
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),  # 目标：躯干链接
            "mass_distribution_params": (-1.0, 3.0),  # 质量变化范围：-1kg到+3kg
            "operation": "add",  # 操作类型：添加（而非替换）
        },
    )

    # ======================== 重置时事件 (reset) ========================
    # 这些事件在每个Episode开始时执行，初始化机器人状态

    # 外力/力矩扰动：在躯干上施加随机外力（当前设为0，可开启）
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,  # 施加外力/力矩的函数
        mode="reset",  # 重置时执行
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),  # 目标：躯干链接
            "force_range": (0.0, 0.0),  # 外力范围（N），当前禁用
            "torque_range": (-0.0, 0.0),  # 力矩范围（Nm），当前禁用
        },
    )

    # 基座状态重置：随机化机器人的初始位置和朝向
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,  # 均匀随机重置根状态的函数
        mode="reset",  # 重置时执行
        params={
            # 位姿随机化范围
            "pose_range": {
                "x": (-0.5, 0.5),  # X位置偏移范围（米）
                "y": (-0.5, 0.5),  # Y位置偏移范围（米）
                "yaw": (-3.14, 3.14),  # 偏航角范围（弧度），完整360度
            },
            # 速度随机化范围（当前全部设为0）
            "velocity_range": {
                "x": (0.0, 0.0),  # X方向线速度
                "y": (0.0, 0.0),  # Y方向线速度
                "z": (0.0, 0.0),  # Z方向线速度
                "roll": (0.0, 0.0),  # 横滚角速度
                "pitch": (0.0, 0.0),  # 俯仰角速度
                "yaw": (0.0, 0.0),  # 偏航角速度
            },
        },
    )

    # 关节状态重置：重置机器人关节到默认位置附近
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,  # 按比例重置关节的函数
        mode="reset",  # 重置时执行
        params={
            "position_range": (1.0, 1.0),  # 位置缩放范围（1.0表示使用默认位置）
            "velocity_range": (-1.0, 1.0),  # 速度随机化范围
        },
    )

    # ======================== 间隔事件 (interval) ========================
    # 这些事件在训练过程中周期性执行，提供持续的干扰

    # 推力干扰：周期性地给机器人施加推力，测试平衡恢复能力
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,  # 通过设置速度来施加推力的函数
        mode="interval",  # 间隔执行模式
        interval_range_s=(5.0, 5.0),  # 执行间隔：固定5秒
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),  # X方向速度变化范围（m/s）
                "y": (-0.5, 0.5),  # Y方向速度变化范围（m/s）
            }
        },
    )


# ============================================================================
#                           命令配置类
# ============================================================================
@configclass
class CommandsCfg:
    """
    命令配置类 - 速度指令生成
    
    定义机器人需要跟踪的目标速度指令，这些指令在训练过程中会被
    周期性地重新采样，为机器人提供多样化的运动目标。
    """

    # 基座速度命令：生成机器人需要跟踪的线速度和角速度目标
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",  # 目标资产名称
        resampling_time_range=(10.0, 10.0),  # 指令重采样时间间隔（秒），固定10秒
        rel_standing_envs=0.02,  # 站立环境比例：2%的环境目标速度为0（站立不动）
        rel_heading_envs=1.0,  # 航向跟踪环境比例（暂未使用）
        heading_command=False,  # 是否使用航向命令模式
        debug_vis=True,  # 是否显示速度命令的调试可视化（箭头）
        # 初始速度范围：训练初期使用较小的速度范围，便于学习基础行走
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1),  # X方向线速度范围（m/s）：前后
            lin_vel_y=(-0.1, 0.1),  # Y方向线速度范围（m/s）：左右
            ang_vel_z=(-0.1, 0.1),  # Z轴角速度范围（rad/s）：转向
        ),
        # 最终速度范围：课程学习后期使用的完整速度范围
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),  # X方向线速度范围：后退0.5m/s，前进1.0m/s
            lin_vel_y=(-0.3, 0.3),  # Y方向线速度范围：左右0.3m/s
            ang_vel_z=(-0.2, 0.2),  # Z轴角速度范围：左右转0.2rad/s
        ),
    )


# ============================================================================
#                           动作配置类
# ============================================================================
@configclass
class ActionsCfg:
    """
    动作配置类 - 关节控制动作空间
    
    定义策略网络输出的动作空间。本配置使用关节位置控制，
    策略输出的动作值会被缩放后作为目标关节位置发送给PD控制器。
    """

    # 关节位置动作：策略输出关节位置目标
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",  # 目标资产名称
        joint_names=[".*"],  # 控制的关节：所有关节（使用正则表达式匹配）
        scale=0.25,  # 动作缩放因子：策略输出乘以0.25得到位置偏移
        use_default_offset=True,  # 使用默认关节位置作为偏移基准
    )


# ============================================================================
#                           观测配置类
# ============================================================================
@configclass
class ObservationsCfg:
    """
    观测配置类 - 策略网络和评论家网络的输入
    
    定义两组观测:
        1. PolicyCfg: 策略网络的观测，添加噪声模拟真实传感器
        2. CriticCfg: 评论家网络的观测（特权信息），无噪声，用于训练
    
    使用非对称Actor-Critic架构：
        - Actor（策略）只能访问带噪声的真实可测量信息
        - Critic（评论家）可以访问仿真中的特权信息，帮助训练
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """
        策略网络观测组
        
        这些观测将被送入策略网络，用于生成动作。
        所有观测都添加了噪声，模拟真实传感器的测量误差。
        注意：观测项的顺序会被保留，影响最终观测向量的拼接顺序。
        """

        # ============== 本体感知观测（Proprioception） ==============
        # 基座角速度：IMU测量的角速度（带噪声）
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,  # 获取基座角速度的函数
            scale=0.2,  # 缩放因子，将角速度值归一化
            noise=Unoise(n_min=-0.2, n_max=0.2),  # 加性均匀噪声范围
        )

        # 重力投影：重力向量在机器人坐标系下的投影，反映机器人姿态
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,  # 获取投影重力的函数
            noise=Unoise(n_min=-0.05, n_max=0.05),  # 加性均匀噪声
        )

        # ============== 命令观测 ==============
        # 速度命令：当前需要跟踪的目标速度 [vx, vy, wz]
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,  # 获取生成命令的函数
            params={"command_name": "base_velocity"},  # 指定命令名称
        )

        # ============== 关节状态观测 ==============
        # 关节位置偏差：当前关节位置相对于默认位置的偏差
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,  # 获取相对关节位置的函数
            noise=Unoise(n_min=-0.01, n_max=0.01),  # 加性噪声，模拟编码器噪声
        )

        # 关节速度：当前关节角速度
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,  # 获取关节速度的函数
            scale=0.05,  # 缩放因子，归一化速度值
            noise=Unoise(n_min=-1.5, n_max=1.5),  # 加性噪声，模拟速度测量噪声
        )

        # ============== 动作历史观测 ==============
        # 上一步动作：帮助策略学习平滑的动作序列
        last_action = ObsTerm(func=mdp.last_action)  # 获取上一步动作

        # 步态相位（可选，已注释）：周期性步态时钟信号
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})

        def __post_init__(self):
            """观测组后初始化配置"""
            self.history_length = 5  # 观测历史长度：保存最近5帧观测，形成时序信息
            self.enable_corruption = True  # 启用噪声干扰，模拟传感器噪声
            self.concatenate_terms = True  # 将所有观测项拼接成单一向量

    # 策略观测组实例
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """
        评论家网络观测组（特权信息）
        
        评论家可以访问仿真中的完美信息（无噪声），
        用于在训练时提供更准确的价值估计。
        部署时不使用评论家，因此可以使用真实环境中无法获取的信息。
        """

        # 基座线速度：仿真中的真实线速度（真实环境中难以直接测量）
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)  # 无噪声的线速度

        # 基座角速度：无噪声版本
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)

        # 重力投影：无噪声版本
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        # 速度命令
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        # 关节位置偏差：无噪声版本
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)

        # 关节速度：无噪声版本
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)

        # 上一步动作
        last_action = ObsTerm(func=mdp.last_action)

        # 步态相位（可选，已注释）
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})

        # 高度扫描（可选，已注释）：地形高度图，用于地形感知
        # height_scanner = ObsTerm(func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 5.0),
        # )

        def __post_init__(self):
            """评论家观测组后初始化配置"""
            self.history_length = 5  # 观测历史长度

    # 评论家观测组实例（特权观测）
    critic: CriticCfg = CriticCfg()


# ============================================================================
#                           奖励配置类
# ============================================================================
@configclass
class RewardsCfg:
    """
    奖励配置类 - 强化学习奖励函数
    
    奖励函数是RL训练的核心，决定了策略学习的目标。
    本配置包含多种奖励项，分为以下几类:
        1. 任务奖励（正向）：鼓励完成速度跟踪任务
        2. 正则化惩罚（负向）：惩罚不自然或高能耗的行为
        3. 姿态奖励：保持良好的身体姿态
        4. 步态奖励：学习自然的行走步态
        5. 安全惩罚：避免不期望的碰撞
    
    权重(weight)的正负决定是奖励还是惩罚，绝对值决定重要程度。
    """

    # ====================== 任务奖励（Task Rewards） ======================
    # 这些奖励直接关联到任务目标：跟踪速度命令

    # 线速度跟踪奖励：在机器人偏航坐标系下跟踪XY平面线速度
    # 使用指数函数：exp(-error²/std²)，误差越小奖励越高
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,  # 线速度跟踪函数（指数形式）
        weight=1.0,  # 权重：正值表示奖励
        params={
            "command_name": "base_velocity",  # 命令名称
            "std": math.sqrt(0.25),  # 标准差：控制奖励曲线的陡峭程度，≈0.5
        },
    )

    # 角速度跟踪奖励：跟踪Z轴（偏航）角速度命令
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,  # 角速度跟踪函数（指数形式）
        weight=0.5,  # 权重：比线速度权重低，优先保证线速度跟踪
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),  # 标准差
        },
    )

    # 存活奖励：每步给予小额奖励，鼓励策略保持机器人不摔倒
    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # ====================== 基座运动正则化惩罚 ======================
    # 惩罚不必要的身体运动，使行走更平稳

    # Z轴线速度惩罚：抑制垂直方向的弹跳
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)

    # XY轴角速度惩罚：抑制横滚和俯仰方向的角速度（身体摇晃）
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    # ====================== 关节运动正则化惩罚 ======================
    # 惩罚剧烈的关节运动，促进平滑、节能的动作

    # 关节速度惩罚：抑制过快的关节运动
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)

    # 关节加速度惩罚：抑制关节运动的突变（抖动）
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)

    # 动作变化率惩罚：惩罚连续动作之间的差异，促进平滑控制
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)

    # 关节位置限位惩罚：惩罚接近关节极限的情况
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)

    # 能量消耗惩罚：惩罚高功率消耗，鼓励节能行走
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    # ====================== 关节偏差惩罚 ======================
    # 惩罚特定关节偏离默认位置，保持自然姿态

    # 手臂关节偏差惩罚：保持手臂在默认位置附近（自然垂放）
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,  # L1范数偏差惩罚
        weight=-0.1,  # 较小权重，允许适度摆臂
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",  # 肩关节
                    ".*_elbow_joint",  # 肘关节
                    ".*_wrist_.*",  # 腕关节
                ],
            )
        },
    )

    # 腰部关节偏差惩罚：保持躯干直立
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1,  # 较大权重，强制躯干直立
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist.*",  # 腰部关节
                ],
            )
        },
    )

    # 腿部关节偏差惩罚：限制髋关节的横向运动
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_roll_joint",  # 髋横滚关节
                    ".*_hip_yaw_joint",  # 髋偏航关节
                ],
            )
        },
    )

    # ====================== 姿态奖励 ======================
    # 保持良好的身体姿态

    # 水平姿态惩罚：惩罚躯干倾斜，保持躯干水平
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)

    # 基座高度惩罚：保持合适的站立高度
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-10,  # 较大权重，严格约束高度
        params={"target_height": 0.78},  # 目标高度：0.78米
    )

    # ====================== 步态奖励 ======================
    # 引导学习自然的双足行走步态

    # 步态周期奖励：鼓励周期性的交替抬腿
    gait = RewTerm(
        func=mdp.feet_gait,  # 步态奖励函数
        weight=0.5,  # 中等权重
        params={
            "period": 0.8,  # 步态周期：0.8秒（一个完整的左右脚循环）
            "offset": [0.0, 0.5],  # 左右脚相位偏移：0和0.5，表示交替（相差半周期）
            "threshold": 0.55,  # 接触力阈值：判定脚是否着地
            "command_name": "base_velocity",  # 速度命令名（用于判断是否需要行走）
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),  # 脚踝接触传感器
        },
    )

    # 脚部滑动惩罚：惩罚脚在地面上的滑动（应该是稳定支撑）
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),  # 脚踝body
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),  # 接触传感器
        },
    )

    # 脚部离地高度奖励：鼓励摆动腿抬到合适高度
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,  # 脚部离地奖励函数
        weight=1.0,
        params={
            "std": 0.05,  # 标准差
            "tanh_mult": 2.0,  # tanh乘数，控制奖励曲线形状
            "target_height": 0.1,  # 目标抬腿高度：0.1米
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),  # 脚踝body
        },
    )

    # ====================== 安全惩罚 ======================
    # 惩罚危险或不期望的行为

    # 非期望碰撞惩罚：惩罚除脚以外部位与地面的接触（摔倒、磕碰）
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,  # 接触力阈值（N）
            # 排除脚踝的所有body，使用负向前瞻正则表达式
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )


# ============================================================================
#                           终止条件配置类
# ============================================================================
@configclass
class TerminationsCfg:
    """
    终止条件配置类 - Episode终止判定
    
    定义何时结束当前训练Episode。终止条件分为两类:
        1. time_out=True: 正常超时终止，不算失败（用于bootstrap）
        2. time_out=False: 异常终止（摔倒等），算作失败
    
    合理的终止条件能够:
        - 避免无意义的仿真时间
        - 帮助策略学习避免危险状态
        - 提供明确的失败信号
    """

    # 超时终止：达到最大Episode时长后正常结束
    time_out = DoneTerm(
        func=mdp.time_out,  # 超时检测函数
        time_out=True,  # 标记为超时类型，不算失败
    )

    # 高度过低终止：机器人基座高度低于阈值时终止（摔倒）
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum,  # 高度检测函数
        params={"minimum_height": 0.2},  # 最小高度阈值：0.2米
    )

    # 姿态异常终止：机器人倾斜角度过大时终止（即将摔倒）
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,  # 姿态检测函数
        params={"limit_angle": 0.8},  # 最大允许倾斜角：0.8弧度（约46度）
    )


# ============================================================================
#                           课程学习配置类
# ============================================================================
@configclass
class CurriculumCfg:
    """
    课程学习配置类 - 训练难度递增策略
    
    课程学习通过逐步增加任务难度来帮助策略学习:
        1. 地形难度课程：从简单地形逐步过渡到复杂地形
        2. 速度命令课程：从低速命令逐步增加到高速命令
    
    这种方式比直接在困难环境中训练更容易收敛。
    """

    # 地形难度课程：根据机器人性能自动调整地形难度等级
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

    # 速度命令课程：逐步扩大速度命令的范围
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


# ============================================================================
#                           主环境配置类
# ============================================================================
@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """
    机器人环境主配置类 - 整合所有配置模块
    
    这是环境配置的顶层类，继承自ManagerBasedRLEnvCfg。
    它将所有子配置组合在一起，并定义仿真的基本参数。
    
    配置模块:
        - scene: 场景配置（地形、机器人、传感器）
        - observations: 观测配置
        - actions: 动作配置
        - commands: 命令配置
        - rewards: 奖励配置
        - terminations: 终止条件配置
        - events: 事件配置（域随机化）
        - curriculum: 课程学习配置
    """

    # ======================== 场景配置 ========================
    # num_envs: 并行环境数量，决定了批处理大小
    # env_spacing: 环境间距，避免不同环境中的机器人相互干扰
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)

    # ======================== 基本MDP配置 ========================
    observations: ObservationsCfg = ObservationsCfg()  # 观测空间配置
    actions: ActionsCfg = ActionsCfg()  # 动作空间配置
    commands: CommandsCfg = CommandsCfg()  # 命令生成配置

    # ======================== MDP核心配置 ========================
    rewards: RewardsCfg = RewardsCfg()  # 奖励函数配置
    terminations: TerminationsCfg = TerminationsCfg()  # 终止条件配置
    events: EventCfg = EventCfg()  # 域随机化事件配置
    curriculum: CurriculumCfg = CurriculumCfg()  # 课程学习配置

    def __post_init__(self):
        """
        后初始化方法 - 设置仿真参数和传感器更新频率
        
        该方法在配置对象创建后自动调用，用于:
            1. 设置仿真时间参数
            2. 配置物理引擎参数
            3. 设置传感器更新周期
            4. 启用/禁用课程学习
        """
        # ======================== 通用设置 ========================
        # decimation: 控制频率降采样，策略频率 = 仿真频率 / decimation
        # 仿真频率200Hz (dt=0.005)，策略频率50Hz (decimation=4)
        self.decimation = 4

        # Episode最大时长（秒）
        self.episode_length_s = 20.0

        # ======================== 仿真设置 ========================
        # 仿真时间步长：0.005秒 = 200Hz
        self.sim.dt = 0.005

        # 渲染间隔：每decimation个仿真步渲染一次
        self.sim.render_interval = self.decimation

        # 使用地形的物理材质作为默认材质
        self.sim.physics_material = self.scene.terrain.physics_material

        # PhysX GPU参数：增加刚体patch数量上限，支持大规模并行仿真
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # ======================== 传感器更新周期 ========================
        # 接触力传感器：每个仿真步更新（200Hz）
        self.scene.contact_forces.update_period = self.sim.dt

        # 高度扫描器：每个策略步更新（50Hz）
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # ======================== 课程学习设置 ========================
        # 检查是否启用地形难度课程学习
        # 如果启用，地形生成器会生成难度递增的地形
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


# ============================================================================
#                           测试/演示环境配置类
# ============================================================================
@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    """
    演示/测试环境配置类 - 用于策略评估和可视化

    继承自RobotEnvCfg，但修改了部分参数使其更适合演示:
        - 减少并行环境数量（便于可视化）
        - 减少地形块数量（加快加载）
        - 使用完整速度范围（测试策略能力上限）
    """

    def __post_init__(self):
        """
        后初始化方法 - 覆盖训练配置，适配演示需求
        """
        # 首先调用父类的后初始化
        super().__post_init__()

        # 减少环境数量：32个足够用于演示和评估
        self.scene.num_envs = 32

        # 减少地形块数量：加快加载速度
        self.scene.terrain.terrain_generator.num_rows = 2  # 2个难度等级
        self.scene.terrain.terrain_generator.num_cols = 10  # 每个等级10个变体

        # 使用完整速度范围：测试策略的极限能力
        # 将初始范围设置为limit_ranges，跳过课程学习的渐进过程
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


# ============================================================================
#                  阶段2：带传感器行走（平地，有 height_scan）
# ============================================================================
@configclass
class HeightScanSceneCfg(RobotSceneCfg):
    """
    带高度扫描的场景配置（阶段2）

    与 RobotSceneCfg 相同，但修正 height_scanner 的 z 偏移
    使其能够正确感知地形
    """

    # 高度扫描器：修正 z 偏移，使其能正确感知地形
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.2, 0.0, 0.8)),  # 修正：从 20.0 改为 0.8
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class HeightScanObservationsCfg:
    """
    带高度扫描的观测配置（阶段2）

    在原有观测基础上添加 height_scan 观测
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """策略网络观测组 - 包含高度扫描"""

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )

        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

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

        # 高度扫描 - 阶段2新增观测
        height_scan = ObsTerm(
            func=mdp.height_scan,
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

        # 评论家也使用高度扫描（无噪声）
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.history_length = 5

    critic: CriticCfg = CriticCfg()


@configclass
class RobotHeightScanEnvCfg(ManagerBasedRLEnvCfg):
    """
    带高度扫描的平地行走环境配置（阶段2）

    在平地上训练机器人使用 height_scan 观测
    为后续楼梯任务做准备
    """

    scene: HeightScanSceneCfg = HeightScanSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: HeightScanObservationsCfg = HeightScanObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

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
class RobotHeightScanPlayEnvCfg(RobotHeightScanEnvCfg):
    """带高度扫描的平地行走演示环境配置"""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10

        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
