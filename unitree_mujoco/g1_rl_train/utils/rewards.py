"""
===============================================================================
Unitree G1 29DOF MuJoCo 奖励函数模块

仿照 IsaacLab mdp 的奖励函数实现
===============================================================================
"""

import numpy as np
from typing import Dict, Tuple


class RewardCalculator:
    """奖励计算器"""
    
    def __init__(self, cfg):
        """
        Args:
            cfg: RewardsCfg 配置对象
        """
        self.cfg = cfg
        
    def compute_rewards(
        self,
        base_lin_vel: np.ndarray,
        base_ang_vel: np.ndarray,
        projected_gravity: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        action: np.ndarray,
        prev_action: np.ndarray,
        velocity_command: np.ndarray,
        base_height: float,
        prev_base_height: float,
        current_mode: int,
        control_dt: float,
        yaw: float,
        contact_forces: np.ndarray = None,
        feet_air_time: np.ndarray = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算所有奖励
        
        Returns:
            total_reward: 总奖励
            reward_info: 各项奖励详情
        """
        reward_info = {}
        total_reward = 0.0
        
        # ========== 任务奖励 ==========
        # 速度跟踪奖励 (yaw frame)
        track_lin_vel = self.track_lin_vel_xy_yaw_frame_exp(
            base_lin_vel, velocity_command, yaw
        )
        reward_info["track_lin_vel_xy"] = track_lin_vel
        total_reward += track_lin_vel
        
        track_ang_vel = self.track_ang_vel_z_exp(base_ang_vel, velocity_command)
        reward_info["track_ang_vel_z"] = track_ang_vel
        total_reward += track_ang_vel
        
        # 向上进展奖励 (楼梯模式)
        if current_mode in [2, 3]:
            upward = self.upward_progress(base_height, prev_base_height, control_dt)
            reward_info["upward_progress"] = upward
            total_reward += upward
        else:
            reward_info["upward_progress"] = 0.0
        
        # ========== 存活奖励 ==========
        alive = self.is_alive()
        reward_info["alive"] = alive
        total_reward += alive
        
        # ========== 正则化惩罚 ==========
        lin_vel_z = self.lin_vel_z_l2(base_lin_vel)
        reward_info["lin_vel_z_l2"] = lin_vel_z
        total_reward += lin_vel_z
        
        ang_vel_xy = self.ang_vel_xy_l2(base_ang_vel)
        reward_info["ang_vel_xy_l2"] = ang_vel_xy
        total_reward += ang_vel_xy
        
        flat_orientation = self.flat_orientation_l2(projected_gravity)
        reward_info["flat_orientation_l2"] = flat_orientation
        total_reward += flat_orientation
        
        action_rate = self.action_rate_l2(action, prev_action)
        reward_info["action_rate_l2"] = action_rate
        total_reward += action_rate
        
        joint_acc = self.joint_acc_l2(joint_vel)
        reward_info["joint_acc_l2"] = joint_acc
        total_reward += joint_acc
        
        # ========== 姿态奖励 ==========
        base_height_reward = self.base_height_l2(base_height)
        reward_info["base_height_l2"] = base_height_reward
        total_reward += base_height_reward
        
        # ========== 步态奖励 ==========
        if feet_air_time is not None:
            feet_air = self.feet_air_time_reward(feet_air_time, velocity_command)
            reward_info["feet_air_time"] = feet_air
            total_reward += feet_air
        else:
            reward_info["feet_air_time"] = 0.0
        
        # ========== 安全惩罚 ==========
        if contact_forces is not None:
            undesired = self.undesired_contacts(contact_forces)
            reward_info["undesired_contacts"] = undesired
            total_reward += undesired
        else:
            reward_info["undesired_contacts"] = 0.0
        
        return total_reward, reward_info
    
    def track_lin_vel_xy_yaw_frame_exp(
        self,
        base_lin_vel: np.ndarray,
        velocity_command: np.ndarray,
        yaw: float,
    ) -> float:
        """
        跟踪线速度 (yaw frame) - 指数形式
        """
        # 转换到 yaw frame
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        vel_x = base_lin_vel[0] * cos_yaw + base_lin_vel[1] * sin_yaw
        vel_y = -base_lin_vel[0] * sin_yaw + base_lin_vel[1] * cos_yaw
        
        error = (vel_x - velocity_command[0])**2 + (vel_y - velocity_command[1])**2
        std = self.cfg.track_lin_vel_xy_std
        reward = np.exp(-error / (std**2)) * self.cfg.track_lin_vel_xy_weight
        
        return reward
    
    def track_ang_vel_z_exp(
        self,
        base_ang_vel: np.ndarray,
        velocity_command: np.ndarray,
    ) -> float:
        """
        跟踪角速度 - 指数形式
        """
        error = (base_ang_vel[2] - velocity_command[2])**2
        std = self.cfg.track_ang_vel_z_std
        reward = np.exp(-error / (std**2)) * self.cfg.track_ang_vel_z_weight
        
        return reward
    
    def upward_progress(
        self,
        base_height: float,
        prev_base_height: float,
        control_dt: float,
    ) -> float:
        """
        向上进展奖励 (用于楼梯)
        """
        height_progress = (base_height - prev_base_height) / control_dt
        reward = np.clip(height_progress, 0, 1) * self.cfg.upward_progress_weight
        return reward
    
    def is_alive(self) -> float:
        """存活奖励"""
        return self.cfg.alive_weight
    
    def lin_vel_z_l2(self, base_lin_vel: np.ndarray) -> float:
        """Z轴线速度惩罚"""
        return base_lin_vel[2]**2 * self.cfg.lin_vel_z_l2_weight
    
    def ang_vel_xy_l2(self, base_ang_vel: np.ndarray) -> float:
        """XY角速度惩罚"""
        return (base_ang_vel[0]**2 + base_ang_vel[1]**2) * self.cfg.ang_vel_xy_l2_weight
    
    def flat_orientation_l2(self, projected_gravity: np.ndarray) -> float:
        """姿态惩罚 (保持水平)"""
        return (projected_gravity[0]**2 + projected_gravity[1]**2) * self.cfg.flat_orientation_l2_weight
    
    def action_rate_l2(self, action: np.ndarray, prev_action: np.ndarray) -> float:
        """动作变化率惩罚"""
        return np.sum((action - prev_action)**2) * self.cfg.action_rate_l2_weight
    
    def joint_acc_l2(self, joint_vel: np.ndarray) -> float:
        """关节加速度惩罚 (使用速度近似)"""
        return np.sum(joint_vel**2) * self.cfg.joint_acc_l2_weight
    
    def base_height_l2(self, base_height: float) -> float:
        """基座高度惩罚"""
        error = (base_height - self.cfg.target_height)**2
        return error * self.cfg.base_height_l2_weight
    
    def feet_air_time_reward(
        self,
        feet_air_time: np.ndarray,
        velocity_command: np.ndarray,
    ) -> float:
        """
        脚部空中时间奖励
        
        鼓励交替步态
        """
        # 只在移动时奖励
        vel_norm = np.sqrt(velocity_command[0]**2 + velocity_command[1]**2)
        if vel_norm < 0.1:
            return 0.0
        
        # 计算奖励
        reward = 0.0
        for air_time in feet_air_time:
            if air_time > self.cfg.feet_air_time_threshold:
                reward += air_time - self.cfg.feet_air_time_threshold
        
        return reward * self.cfg.feet_air_time_weight
    
    def undesired_contacts(self, contact_forces: np.ndarray) -> float:
        """
        不期望的接触惩罚
        
        惩罚除脚以外的身体部位接触地面
        """
        # contact_forces 应该是除脚以外的接触力
        if contact_forces is None or len(contact_forces) == 0:
            return 0.0
        
        # 检查是否超过阈值
        max_force = np.max(np.abs(contact_forces))
        if max_force > self.cfg.undesired_contacts_threshold:
            return self.cfg.undesired_contacts_weight
        
        return 0.0


def compute_simple_rewards(
    base_lin_vel: np.ndarray,
    base_ang_vel: np.ndarray,
    projected_gravity: np.ndarray,
    action: np.ndarray,
    prev_action: np.ndarray,
    velocity_command: np.ndarray,
    base_height: float,
    yaw: float,
    target_height: float = 0.72,
) -> Tuple[float, Dict[str, float]]:
    """
    简化版奖励计算函数
    
    用于快速测试
    """
    reward_info = {}
    total_reward = 0.0
    
    # 速度跟踪
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    vel_x = base_lin_vel[0] * cos_yaw + base_lin_vel[1] * sin_yaw
    vel_y = -base_lin_vel[0] * sin_yaw + base_lin_vel[1] * cos_yaw
    
    lin_vel_error = (vel_x - velocity_command[0])**2 + (vel_y - velocity_command[1])**2
    track_lin_vel = np.exp(-lin_vel_error / 0.25)
    reward_info["track_lin_vel"] = track_lin_vel
    total_reward += track_lin_vel
    
    ang_vel_error = (base_ang_vel[2] - velocity_command[2])**2
    track_ang_vel = np.exp(-ang_vel_error / 0.25) * 0.5
    reward_info["track_ang_vel"] = track_ang_vel
    total_reward += track_ang_vel
    
    # 存活
    alive = 0.15
    reward_info["alive"] = alive
    total_reward += alive
    
    # 惩罚项
    lin_vel_z = base_lin_vel[2]**2 * (-2.0)
    reward_info["lin_vel_z"] = lin_vel_z
    total_reward += lin_vel_z
    
    ang_vel_xy = (base_ang_vel[0]**2 + base_ang_vel[1]**2) * (-0.05)
    reward_info["ang_vel_xy"] = ang_vel_xy
    total_reward += ang_vel_xy
    
    flat_orientation = (projected_gravity[0]**2 + projected_gravity[1]**2) * (-1.0)
    reward_info["flat_orientation"] = flat_orientation
    total_reward += flat_orientation
    
    action_rate = np.sum((action - prev_action)**2) * (-0.01)
    reward_info["action_rate"] = action_rate
    total_reward += action_rate
    
    height_error = (base_height - target_height)**2 * (-0.5)
    reward_info["base_height"] = height_error
    total_reward += height_error
    
    return total_reward, reward_info


if __name__ == "__main__":
    # 测试奖励计算
    from configs.unified_env_cfg import RewardsCfg
    
    cfg = RewardsCfg()
    calculator = RewardCalculator(cfg)
    
    # 模拟数据
    base_lin_vel = np.array([0.5, 0.0, 0.0])
    base_ang_vel = np.array([0.0, 0.0, 0.1])
    projected_gravity = np.array([0.0, 0.0, -1.0])
    joint_pos = np.zeros(29)
    joint_vel = np.zeros(29)
    action = np.zeros(29)
    prev_action = np.zeros(29)
    velocity_command = np.array([0.5, 0.0, 0.0])
    base_height = 0.72
    prev_base_height = 0.72
    current_mode = 0
    control_dt = 0.02
    yaw = 0.0
    
    total, info = calculator.compute_rewards(
        base_lin_vel, base_ang_vel, projected_gravity,
        joint_pos, joint_vel, action, prev_action,
        velocity_command, base_height, prev_base_height,
        current_mode, control_dt, yaw
    )
    
    print(f"Total reward: {total:.4f}")
    for key, value in info.items():
        print(f"  {key}: {value:.4f}")
