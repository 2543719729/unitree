"""Unitree 执行器物理模型

本模块实现了 Unitree 机器人执行器的真实物理特性，包括：
- 扭矩-速度曲线建模
- 摩擦力模型（静摩擦 + 动摩擦）
- 电机惯量计算
- 各型号执行器的预定义配置
"""

from __future__ import annotations

import torch
from dataclasses import MISSING

from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class UnitreeActuator(DelayedPDActuator):
    """Unitree 执行器类 - 实现真实的扭矩-速度曲线

    扭矩-速度特性曲线定义如下：

            扭矩限制 (N·m)
                ^
    Y2──────────|              (反向扭矩：制动时)
                |──────────────Y1  (同向扭矩：加速时)
                |              │\
                |              │ \  (线性衰减区)
                |              │  \
                |              |   \
    ------------+--------------|------> 速度 (rad/s)
                              X1   X2

    参数说明:
    - Y1: 同向峰值扭矩（扭矩与速度方向相同，电机加速）
    - Y2: 反向峰值扭矩（扭矩与速度方向相反，电机制动，通常 Y2 > Y1）
    - X1: 全扭矩最大速度（T-N 曲线拐点，超过此速度扭矩开始下降）
    - X2: 空载最大速度（电机无负载时的理论最大速度）

    摩擦模型参数:
    - Fs: 静摩擦系数（低速时的摩擦力）
    - Fd: 动摩擦系数（速度相关的摩擦力）
    - Va: 摩擦激活速度（静摩擦完全激活的速度阈值）
    
    物理意义:
    这个模型比简单的 PD 控制更接近真实电机特性，考虑了：
    1. 电机在不同速度下的扭矩限制
    2. 加速和制动时的不对称性
    3. 真实的摩擦损耗
    """

    cfg: UnitreeActuatorCfg  # 执行器配置

    armature: torch.Tensor
    """执行器关节的电枢惯量 (Armature Inertia)
    
    形状: (num_envs, num_joints)
    
    物理公式:
        armature = J_rotor + J_gear1 * i1² + J_gear2 * (i1 * i2)²
    
    其中:
        - J_rotor: 转子惯量
        - J_gear1, J_gear2: 各级齿轮惯量
        - i1, i2: 各级减速比
    
    电枢惯量影响:
        - 响应速度：惯量越大，响应越慢
        - 能量消耗：惯量越大，加速时能耗越高
        - 稳定性：适当的惯量有助于系统稳定
    """

    def __init__(self, cfg: UnitreeActuatorCfg, *args, **kwargs):
        """初始化 Unitree 执行器
        
        解析配置参数并初始化扭矩-速度曲线的各个关键点
        """
        super().__init__(cfg, *args, **kwargs)

        # 保存当前关节速度（用于判断速度方向）
        self._joint_vel = torch.zeros_like(self.computed_effort)
        
        # 解析扭矩-速度曲线参数
        self._effort_y1 = self._parse_joint_parameter(cfg.Y1, 1e9)        # 同向峰值扭矩
        self._effort_y2 = self._parse_joint_parameter(cfg.Y2, cfg.Y1)    # 反向峰值扭矩（默认等于Y1）
        self._velocity_x1 = self._parse_joint_parameter(cfg.X1, 1e9)     # 全扭矩最大速度
        self._velocity_x2 = self._parse_joint_parameter(cfg.X2, 1e9)     # 空载最大速度
        
        # 解析摩擦模型参数
        self._friction_static = self._parse_joint_parameter(cfg.Fs, 0.0)   # 静摩擦系数
        self._friction_dynamic = self._parse_joint_parameter(cfg.Fd, 0.0)  # 动摩擦系数
        self._activation_vel = self._parse_joint_parameter(cfg.Va, 0.01)   # 摩擦激活速度

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        """计算实际输出扭矩
        
        步骤:
        1. 保存当前关节速度
        2. 调用父类计算期望扭矩（PD控制）
        3. 应用摩擦模型
        4. 应用扭矩-速度限制
        
        Args:
            control_action: 控制动作（目标位置/速度）
            joint_pos: 当前关节位置
            joint_vel: 当前关节速度
            
        Returns:
            修正后的控制动作（仅包含扭矩）
        """
        # 1. 保存当前关节速度（用于后续的速度方向判断）
        self._joint_vel[:] = joint_vel
        
        # 2. 调用父类的 PD 控制器计算期望扭矩
        control_action = super().compute(control_action, joint_pos, joint_vel)

        # 3. 应用摩擦模型减少扭矩
        # 摩擦力 = 静摩擦(tanh平滑) + 动摩擦(线性)
        # tanh函数用于平滑过渡，避免零速度附近的不连续
        self.applied_effort -= (
            self._friction_static * torch.tanh(joint_vel / self._activation_vel) + self._friction_dynamic * joint_vel
        )

        # 4. 清空位置和速度命令，仅输出扭矩
        control_action.joint_positions = None
        control_action.joint_velocities = None
        control_action.joint_efforts = self.applied_effort

        return control_action

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        """根据扭矩-速度曲线裁剪扭矩
        
        逻辑:
        1. 判断扭矩和速度是否同向
           - 同向：使用 Y1（加速扭矩限制）
           - 反向：使用 Y2（制动扭矩限制，通常更大）
        
        2. 判断当前速度是否超过拐点 X1
           - 未超过：使用峰值扭矩
           - 超过：根据线性衰减计算扭矩限制
        
        Args:
            effort: 期望输出扭矩
            
        Returns:
            裁剪后的扭矩
        """
        # 1. 检查扭矩和速度是否同向（正为同向）
        same_direction = (self._joint_vel * effort) > 0
        max_effort = torch.where(same_direction, self._effort_y1, self._effort_y2)
        
        # 2. 检查关节速度是否小于全扭矩最大速度（拐点）
        max_effort = torch.where(
            self._joint_vel.abs() < self._velocity_x1, 
            max_effort,                              # 低速区：使用峰值扭矩
            self._compute_effort_limit(max_effort)   # 高速区：线性衰减
        )
        
        # 3. 裁剪到 [-max_effort, max_effort] 范围
        return torch.clip(effort, -max_effort, max_effort)

    def _compute_effort_limit(self, max_effort):
        """计算高速区的扭矩限制（线性衰减）
        
        数学模型:
            在 [X1, X2] 区间，扭矩从 max_effort 线性衰减到 0
            
            斜率: k = -max_effort / (X2 - X1)
            扭矩: T = k * (v - X1) + max_effort
        
        物理意义:
            电机在高速时由于反电动势增大，可输出扭矩降低
        
        Args:
            max_effort: 峰值扭矩 (Y1 或 Y2)
            
        Returns:
            当前速度下的扭矩限制
        """
        # 计算线性衰减的斜率（负值）
        k = -max_effort / (self._velocity_x2 - self._velocity_x1)
        
        # 计算当前速度对应的扭矩限制
        limit = k * (self._joint_vel.abs() - self._velocity_x1) + max_effort
        
        # 确保扭矩限制不小于0
        return limit.clip(min=0.0)


@configclass
class UnitreeActuatorCfg(DelayedPDActuatorCfg):
    """Unitree 执行器配置基类
    
    定义了 Unitree 执行器的扭矩-速度曲线参数和摩擦模型参数
    所有具体型号的执行器配置都继承自此类
    """

    class_type: type = UnitreeActuator

    # ========== 扭矩-速度曲线参数 ==========
    X1: float = 1e9
    """全扭矩最大速度 (T-N 曲线拐点) 单位: rad/s
    
    物理意义: 超过此速度后，可输出扭矩开始线性衰减
    默认 1e9 表示无限制（不考虑速度影响）
    """

    X2: float = 1e9
    """空载最大速度（无负载测试） 单位: rad/s
    
    物理意义: 电机在无负载情况下的理论最大转速
    到达此速度时，可输出扭矩降为 0
    """

    Y1: float = MISSING
    """同向峰值扭矩（扭矩与速度同向） 单位: N·m
    
    物理意义: 电机加速时的最大扭矩
    必须指定，无默认值
    """

    Y2: float | None = None
    """反向峰值扭矩（扭矩与速度反向） 单位: N·m
    
    物理意义: 电机制动时的最大扭矩，通常大于 Y1
    如果不指定，默认等于 Y1
    """

    # ========== 摩擦模型参数 ==========
    Fs: float = 0.0
    """静摩擦系数 (Static Friction)
    
    物理意义: 低速时的摩擦力大小
    摩擦力 = Fs * tanh(v / Va)
    """

    Fd: float = 0.0
    """动摩擦系数 (Dynamic Friction)
    
    物理意义: 与速度成正比的摩擦力
    摩擦力 = Fd * v
    """

    Va: float = 0.01
    """摩擦激活速度 单位: rad/s
    
    物理意义: 静摩擦完全激活的速度阈值
    使用 tanh 函数平滑过渡，避免零速度附近的不连续
    """


@configclass
class UnitreeActuatorCfg_M107_15(UnitreeActuatorCfg):
    """M107-15 执行器配置
    
    应用: 中型四足机器人髋关节
    峰值扭矩: 150 N·m (同向) / 182.8 N·m (反向)
    额定功率: ~2.1 kW
    """
    X1 = 14.0      # 全扭矩最大速度: 14 rad/s
    X2 = 25.6      # 空载最大速度: 25.6 rad/s
    Y1 = 150.0     # 同向峰值扭矩: 150 N·m
    Y2 = 182.8     # 反向峰值扭矩: 182.8 N·m

    armature = 0.063259741  # 电枢惯量: 0.0633 kg·m²


@configclass
class UnitreeActuatorCfg_M107_24(UnitreeActuatorCfg):
    """M107-24 执行器配置
    
    应用: 重型四足/人形机器人膝关节
    峰值扭矩: 240 N·m (同向) / 292.5 N·m (反向)
    特点: 大扭矩，中速
    """
    X1 = 8.8       # 全扭矩最大速度: 8.8 rad/s
    X2 = 16        # 空载最大速度: 16 rad/s
    Y1 = 240       # 同向峰值扭矩: 240 N·m
    Y2 = 292.5     # 反向峰值扭矩: 292.5 N·m

    armature = 0.160478022  # 电枢惯量: 0.160 kg·m²


@configclass
class UnitreeActuatorCfg_Go2HV(UnitreeActuatorCfg):
    """Go2-HV 执行器配置
    
    应用: Go2 四足机器人全身关节
    峰值扭矩: 20.2 N·m
    特点: 高速低扭矩，适合敏捷运动
    """
    X1 = 13.5      # 全扭矩最大速度: 13.5 rad/s
    X2 = 30        # 空载最大速度: 30 rad/s (高速)
    Y1 = 20.2      # 同向峰值扭矩: 20.2 N·m
    Y2 = 23.4      # 反向峰值扭矩: 23.4 N·m


@configclass
class UnitreeActuatorCfg_N7520_14p3(UnitreeActuatorCfg):
    """N7520-14.3 执行器配置 (用 p 代替小数点)
    
    应用: 人形机器人髋关节 pitch/yaw、腰部 yaw
    峰值扭矩: 71 N·m
    特点: 中等扭矩，较高速度，有摩擦模型
    """
    X1 = 22.63     # 全扭矩最大速度: 22.63 rad/s
    X2 = 35.52     # 空载最大速度: 35.52 rad/s
    Y1 = 71        # 同向峰值扭矩: 71 N·m
    Y2 = 83.3      # 反向峰值扭矩: 83.3 N·m

    Fs = 1.6       # 静摩擦系数: 1.6 N·m
    Fd = 0.16      # 动摩擦系数: 0.16 N·m·s/rad

    # 电枢惯量计算:
    # | 转子    | 0.489e-4 kg·m²
    # | 一级齿轮 | 0.098e-4 kg·m² | 减速比 | 4.5
    # | 二级齿轮 | 0.533e-4 kg·m² | 减速比 | 48/22+1
    # 总惯量 = J_rotor + J_gear1 * i1² + J_gear2 * (i1*i2)²
    armature = 0.01017752  # 0.0102 kg·m²


@configclass
class UnitreeActuatorCfg_N7520_22p5(UnitreeActuatorCfg):
    """N7520-22.5 执行器配置
    
    应用: 人形机器人髋关节 roll、膝关节
    峰值扭矩: 111 N·m
    特点: 大扭矩，中速，适合承重关节
    """
    X1 = 14.5      # 全扭矩最大速度: 14.5 rad/s
    X2 = 22.7      # 空载最大速度: 22.7 rad/s
    Y1 = 111.0     # 同向峰值扭矩: 111 N·m
    Y2 = 131.0     # 反向峰值扭矪: 131 N·m

    Fs = 2.4       # 静摩擦系数: 2.4 N·m (较大)
    Fd = 0.24      # 动摩擦系数: 0.24 N·m·s/rad

    # 电枢惯量计算:
    # | 转子    | 0.489e-4 kg·m²
    # | 一级齿轮 | 0.109e-4 kg·m² | 减速比 | 4.5
    # | 二级齿轮 | 0.738e-4 kg·m² | 减速比 | 5.0
    armature = 0.025101925  # 0.0251 kg·m² (较大惯量)


@configclass
class UnitreeActuatorCfg_N5010_16(UnitreeActuatorCfg):
    """N5010-16 执行器配置
    
    应用: 轻量级手臂关节
    峰值扭矩: 9.5 N·m
    特点: 小扭矩，高速
    """
    X1 = 27.0      # 全扭矩最大速度: 27 rad/s (高速)
    X2 = 41.5      # 空载最大速度: 41.5 rad/s
    Y1 = 9.5       # 同向峰值扭矩: 9.5 N·m
    Y2 = 17.0      # 反向峰值扭矩: 17 N·m

    # 电枢惯量计算:
    # | 转子    | 0.084e-4 kg·m²
    # | 一级齿轮 | 0.015e-4 kg·m² | 减速比 | 4
    # | 二级齿轮 | 0.068e-4 kg·m² | 减速比 | 4
    armature = 0.0021812  # 0.0022 kg·m² (小惯量)


@configclass
class UnitreeActuatorCfg_N5020_16(UnitreeActuatorCfg):
    """N5020-16 执行器配置
    
    应用: 人形机器人肩膀、手臂、踝部
    峰值扭矩: 24.8 N·m
    特点: 中等扭矩，高速，低摩擦
    """
    X1 = 30.86     # 全扭矩最大速度: 30.86 rad/s (高速)
    X2 = 40.13     # 空载最大速度: 40.13 rad/s
    Y1 = 24.8      # 同向峰值扭矩: 24.8 N·m
    Y2 = 31.9      # 反向峰值扭矩: 31.9 N·m

    Fs = 0.6       # 静摩擦系数: 0.6 N·m (低摩擦)
    Fd = 0.06      # 动摩擦系数: 0.06 N·m·s/rad

    # 电枢惯量计算:
    # | 转子    | 0.139e-4 kg·m²
    # | 一级齿轮 | 0.017e-4 kg·m² | 减速比 | 46/18+1
    # | 二级齿轮 | 0.169e-4 kg·m² | 减速比 | 56/16+1
    armature = 0.003609725  # 0.0036 kg·m²


@configclass
class UnitreeActuatorCfg_W4010_25(UnitreeActuatorCfg):
    """W4010-25 执行器配置
    
    应用: 人形机器人灵巧手腕 (wrist pitch/yaw)
    峰值扭矩: 4.8 N·m
    特点: 小扭矩，精细控制，适合末端执行器
    """
    X1 = 15.3      # 全扭矩最大速度: 15.3 rad/s
    X2 = 24.76     # 空载最大速度: 24.76 rad/s
    Y1 = 4.8       # 同向峰值扭矩: 4.8 N·m (最小)
    Y2 = 8.6       # 反向峰值扭矩: 8.6 N·m

    Fs = 0.6       # 静摩擦系数: 0.6 N·m
    Fd = 0.06      # 动摩擦系数: 0.06 N·m·s/rad

    # 电枢惯量计算:
    # | 转子    | 0.068e-4 kg·m²
    # | 一级齿轮 |                | 减速比 | 5
    # | 二级齿轮 |                | 减速比 | 5
    # 总减速比 = 25
    armature = 0.00425  # 0.00425 kg·m²
