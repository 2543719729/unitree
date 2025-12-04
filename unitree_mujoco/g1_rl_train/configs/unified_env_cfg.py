"""
===============================================================================
Unitree G1 29DOF MuJoCo 统一环境配置文件

仿照 IsaacLab unified_env_cfg.py 的配置结构
支持4模式条件策略训练
===============================================================================
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import math


@dataclass
class SimulationCfg:
    """仿真配置"""
    sim_dt: float = 0.005  # 仿真时间步
    control_decimation: int = 4  # 控制降采样
    episode_length_s: float = 20.0  # 回合时长
    render_fps: int = 50  # 渲染帧率


@dataclass
class TerrainCfg:
    """地形配置"""
    terrain_type: str = "flat"  # "flat", "stairs", "mixed"
    terrain_size: Tuple[float, float] = (8.0, 8.0)
    
    # 平地配置
    flat_proportion: float = 0.4
    
    # 楼梯配置
    stairs_proportion: float = 0.6
    stair_height_range: Tuple[float, float] = (0.08, 0.20)
    stair_width_range: Tuple[float, float] = (0.28, 0.35)


@dataclass
class ModeCfg:
    """模式配置 - 4模式条件策略"""
    num_modes: int = 4
    mode_probabilities: List[float] = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])
    
    # 模式定义:
    #   0: 平地盲走 (无 height_scan)
    #   1: 平地带传感器 (有 height_scan)
    #   2: 楼梯盲爬 (无 height_scan)
    #   3: 楼梯带传感器 (有 height_scan)


@dataclass
class CommandsCfg:
    """速度命令配置"""
    resampling_time_range: Tuple[float, float] = (10.0, 10.0)
    rel_standing_envs: float = 0.02
    heading_command: bool = True
    heading_control_stiffness: float = 0.5
    
    # 命令范围 (初始)
    lin_vel_x_range: Tuple[float, float] = (0.0, 0.0)
    lin_vel_y_range: Tuple[float, float] = (0.0, 0.0)
    ang_vel_z_range: Tuple[float, float] = (0.0, 0.0)
    heading_range: Tuple[float, float] = (-math.pi, math.pi)
    
    # 命令范围 (上限 - 课程学习后)
    limit_lin_vel_x_range: Tuple[float, float] = (-0.5, 1.0)
    limit_lin_vel_y_range: Tuple[float, float] = (-0.3, 0.3)
    limit_ang_vel_z_range: Tuple[float, float] = (-0.5, 0.5)


@dataclass
class ActionsCfg:
    """动作配置"""
    action_scale: float = 0.25
    use_default_offset: bool = True


@dataclass
class PDControlCfg:
    """PD控制器配置"""
    # 腿部
    leg_kp: float = 50.0
    leg_kd: float = 3.5
    
    # 踝部
    ankle_kp: float = 30.0
    ankle_kd: float = 2.0
    
    # 腰部
    waist_kp: float = 40.0
    waist_kd: float = 3.0
    
    # 手臂
    arm_kp: float = 25.0
    arm_kd: float = 2.0
    
    # 手腕
    wrist_kp: float = 10.0
    wrist_kd: float = 1.0


@dataclass
class ObservationsCfg:
    """观测配置"""
    history_length: int = 5
    enable_corruption: bool = True
    concatenate_terms: bool = True
    
    # 观测噪声
    base_ang_vel_scale: float = 0.2
    base_ang_vel_noise: Tuple[float, float] = (-0.2, 0.2)
    
    projected_gravity_noise: Tuple[float, float] = (-0.05, 0.05)
    
    joint_pos_rel_noise: Tuple[float, float] = (-0.01, 0.01)
    
    joint_vel_rel_scale: float = 0.05
    joint_vel_rel_noise: Tuple[float, float] = (-1.5, 1.5)
    
    # Height scan
    use_height_scan: bool = True
    height_scan_resolution: float = 0.1
    height_scan_size: Tuple[float, float] = (1.6, 1.0)  # 16x11 = 176 点，加上边缘 = 187
    height_scan_noise: Tuple[float, float] = (-0.1, 0.1)
    height_scan_clip: Tuple[float, float] = (-1.0, 1.0)


@dataclass
class RewardsCfg:
    """奖励配置"""
    # 任务奖励
    track_lin_vel_xy_weight: float = 1.0
    track_lin_vel_xy_std: float = 0.5  # sqrt(0.25)
    
    track_ang_vel_z_weight: float = 0.5
    track_ang_vel_z_std: float = 0.5
    
    upward_progress_weight: float = 0.5  # 楼梯模式
    
    # 存活奖励
    alive_weight: float = 0.15
    
    # 正则化惩罚
    lin_vel_z_l2_weight: float = -2.0
    ang_vel_xy_l2_weight: float = -0.05
    flat_orientation_l2_weight: float = -1.0
    action_rate_l2_weight: float = -0.01
    joint_acc_l2_weight: float = -2.5e-7
    
    # 姿态奖励
    base_height_l2_weight: float = -0.5
    target_height: float = 0.72
    
    # 步态奖励
    feet_air_time_weight: float = 0.25
    feet_air_time_threshold: float = 0.5
    
    # 安全惩罚
    undesired_contacts_weight: float = -1.0
    undesired_contacts_threshold: float = 1.0


@dataclass
class TerminationsCfg:
    """终止条件配置"""
    time_out: bool = True
    
    base_contact: bool = True
    base_contact_bodies: List[str] = field(default_factory=lambda: ["torso_link", "pelvis"])
    base_contact_threshold: float = 1.0
    
    height_threshold: float = 0.3  # 低于此高度视为跌倒
    orientation_threshold: float = -0.5  # 重力投影 z 分量阈值


@dataclass 
class DomainRandomizationCfg:
    """域随机化配置"""
    enabled: bool = True
    
    # 物理材质
    static_friction_range: Tuple[float, float] = (0.3, 1.0)
    dynamic_friction_range: Tuple[float, float] = (0.3, 1.0)
    restitution_range: Tuple[float, float] = (0.0, 0.0)
    
    # 质量
    add_base_mass_range: Tuple[float, float] = (-1.0, 3.0)
    
    # 初始状态
    pose_range: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "yaw": (-3.14, 3.14),
    })
    velocity_range: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (-0.5, 0.5),
        "roll": (-0.5, 0.5),
        "pitch": (-0.5, 0.5),
        "yaw": (-0.5, 0.5),
    })
    
    # 关节
    joint_position_range: Tuple[float, float] = (0.5, 1.5)
    
    # 外部扰动
    push_interval_s: Tuple[float, float] = (10.0, 15.0)
    push_velocity_range: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
    })


@dataclass
class CurriculumCfg:
    """课程学习配置"""
    enabled: bool = True
    terrain_levels: bool = True
    
    # 命令范围课程
    command_curriculum: bool = True
    command_curriculum_threshold: float = 0.7  # 奖励阈值


@dataclass
class UnifiedEnvCfg:
    """
    统一环境配置主类
    
    训练一个能处理4种模式的策略：
        - 模式0: 平地盲走
        - 模式1: 平地带传感器
        - 模式2: 楼梯盲爬
        - 模式3: 楼梯带传感器
    """
    # 环境数量
    num_envs: int = 4096
    env_spacing: float = 2.5
    
    # 子配置
    simulation: SimulationCfg = field(default_factory=SimulationCfg)
    terrain: TerrainCfg = field(default_factory=TerrainCfg)
    mode: ModeCfg = field(default_factory=ModeCfg)
    commands: CommandsCfg = field(default_factory=CommandsCfg)
    actions: ActionsCfg = field(default_factory=ActionsCfg)
    pd_control: PDControlCfg = field(default_factory=PDControlCfg)
    observations: ObservationsCfg = field(default_factory=ObservationsCfg)
    rewards: RewardsCfg = field(default_factory=RewardsCfg)
    terminations: TerminationsCfg = field(default_factory=TerminationsCfg)
    domain_randomization: DomainRandomizationCfg = field(default_factory=DomainRandomizationCfg)
    curriculum: CurriculumCfg = field(default_factory=CurriculumCfg)


@dataclass
class UnifiedPlayEnvCfg(UnifiedEnvCfg):
    """统一环境演示配置"""
    num_envs: int = 32
    
    def __post_init__(self):
        # 使用完整命令范围
        self.commands.lin_vel_x_range = self.commands.limit_lin_vel_x_range
        self.commands.lin_vel_y_range = self.commands.limit_lin_vel_y_range
        self.commands.ang_vel_z_range = self.commands.limit_ang_vel_z_range


# 预定义配置
def get_default_config() -> UnifiedEnvCfg:
    """获取默认配置"""
    return UnifiedEnvCfg()


def get_flat_blind_config() -> UnifiedEnvCfg:
    """获取平地盲走配置"""
    cfg = UnifiedEnvCfg()
    cfg.mode.mode_probabilities = [1.0, 0.0, 0.0, 0.0]
    cfg.observations.use_height_scan = False
    return cfg


def get_flat_sensor_config() -> UnifiedEnvCfg:
    """获取平地带传感器配置"""
    cfg = UnifiedEnvCfg()
    cfg.mode.mode_probabilities = [0.0, 1.0, 0.0, 0.0]
    return cfg


def get_stairs_blind_config() -> UnifiedEnvCfg:
    """获取楼梯盲爬配置"""
    cfg = UnifiedEnvCfg()
    cfg.mode.mode_probabilities = [0.0, 0.0, 1.0, 0.0]
    cfg.terrain.terrain_type = "stairs"
    cfg.observations.use_height_scan = False
    return cfg


def get_stairs_sensor_config() -> UnifiedEnvCfg:
    """获取楼梯带传感器配置"""
    cfg = UnifiedEnvCfg()
    cfg.mode.mode_probabilities = [0.0, 0.0, 0.0, 1.0]
    cfg.terrain.terrain_type = "stairs"
    return cfg


if __name__ == "__main__":
    # 测试配置
    cfg = get_default_config()
    print(f"Num envs: {cfg.num_envs}")
    print(f"Simulation dt: {cfg.simulation.sim_dt}")
    print(f"Mode probabilities: {cfg.mode.mode_probabilities}")
    print(f"Command vel x range: {cfg.commands.limit_lin_vel_x_range}")
