"""
===============================================================================
Unitree G1 29DOF MuJoCo 强化学习环境

仿照 IsaacLab unified_env_cfg.py 实现的 MuJoCo Gym 环境
支持4模式条件策略训练：
    - 模式0: 平地盲走 (无 height_scan)
    - 模式1: 平地带传感器 (有 height_scan)
    - 模式2: 楼梯盲爬 (无 height_scan)
    - 模式3: 楼梯带传感器 (有 height_scan)
===============================================================================
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Tuple, Any
import os


class G1_29DOF_Env(gym.Env):
    """
    G1 29DOF MuJoCo 强化学习环境
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    # 关节名称列表 (与 g1_29dof.xml 中的 actuator 顺序一致)
    JOINT_NAMES = [
        # 左腿 (6)
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        # 右腿 (6)
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        # 腰部 (3)
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        # 左臂 (7)
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        # 右臂 (7)
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ]
    
    # 脚部 link 名称 (用于接触检测)
    FOOT_LINKS = ["left_ankle_roll_link", "right_ankle_roll_link"]
    
    # 默认站立姿态
    DEFAULT_JOINT_POS = {
        # 左腿
        "left_hip_pitch_joint": -0.1, "left_hip_roll_joint": 0.0, "left_hip_yaw_joint": 0.0,
        "left_knee_joint": 0.3, "left_ankle_pitch_joint": -0.2, "left_ankle_roll_joint": 0.0,
        # 右腿
        "right_hip_pitch_joint": -0.1, "right_hip_roll_joint": 0.0, "right_hip_yaw_joint": 0.0,
        "right_knee_joint": 0.3, "right_ankle_pitch_joint": -0.2, "right_ankle_roll_joint": 0.0,
        # 腰部
        "waist_yaw_joint": 0.0, "waist_roll_joint": 0.0, "waist_pitch_joint": 0.0,
        # 左臂
        "left_shoulder_pitch_joint": 0.0, "left_shoulder_roll_joint": 0.3, "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.5, "left_wrist_roll_joint": 0.0, "left_wrist_pitch_joint": 0.0, "left_wrist_yaw_joint": 0.0,
        # 右臂
        "right_shoulder_pitch_joint": 0.0, "right_shoulder_roll_joint": -0.3, "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 0.5, "right_wrist_roll_joint": 0.0, "right_wrist_pitch_joint": 0.0, "right_wrist_yaw_joint": 0.0,
    }
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        # 仿真参数
        sim_dt: float = 0.005,
        control_decimation: int = 4,
        episode_length_s: float = 20.0,
        # 模式配置
        num_modes: int = 4,
        mode_probabilities: list = None,
        # 命令范围
        cmd_lin_vel_x_range: Tuple[float, float] = (-0.5, 1.0),
        cmd_lin_vel_y_range: Tuple[float, float] = (-0.3, 0.3),
        cmd_ang_vel_z_range: Tuple[float, float] = (-0.5, 0.5),
        # 观测配置
        history_length: int = 5,
        use_height_scan: bool = True,
        height_scan_points: int = 187,  # 对应 GridPatternCfg(resolution=0.1, size=[1.6, 1.0])
        # 动作配置
        action_scale: float = 0.25,
        # PD 控制器参数
        kp: float = 50.0,
        kd: float = 3.5,
        # 域随机化
        domain_randomization: bool = True,
        # 地形类型
        terrain_type: str = "flat",  # "flat", "stairs"
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.sim_dt = sim_dt
        self.control_decimation = control_decimation
        self.control_dt = sim_dt * control_decimation
        self.episode_length_s = episode_length_s
        self.max_episode_steps = int(episode_length_s / self.control_dt)
        
        # 模式配置
        self.num_modes = num_modes
        self.mode_probabilities = mode_probabilities or [0.25, 0.25, 0.25, 0.25]
        self.current_mode = 0
        
        # 命令范围
        self.cmd_lin_vel_x_range = cmd_lin_vel_x_range
        self.cmd_lin_vel_y_range = cmd_lin_vel_y_range
        self.cmd_ang_vel_z_range = cmd_ang_vel_z_range
        
        # 观测配置
        self.history_length = history_length
        self.use_height_scan = use_height_scan
        self.height_scan_points = height_scan_points
        
        # 动作配置
        self.action_scale = action_scale
        self.kp = kp
        self.kd = kd
        
        # 域随机化
        self.domain_randomization = domain_randomization
        self.terrain_type = terrain_type
        
        # 加载 MuJoCo 模型
        self._load_model()
        
        # 关节索引映射
        self._setup_joint_indices()
        
        # 定义观测和动作空间
        self._setup_spaces()
        
        # 初始化历史缓冲
        self._init_history_buffer()
        
        # 内部状态
        self.step_count = 0
        self.last_action = np.zeros(self.num_actions)
        self.prev_action = np.zeros(self.num_actions)
        self.velocity_command = np.zeros(3)  # [vx, vy, wz]
        self.prev_base_height = 0.0
        
        # 渲染相关
        self.viewer = None
        
    def _load_model(self):
        """加载 MuJoCo 模型"""
        # 获取模型路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "../../unitree_robots/g1/scene_29dof.xml")
        
        if not os.path.exists(model_path):
            # 尝试其他路径
            model_path = os.path.join(current_dir, "../../../unitree_robots/g1/scene_29dof.xml")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"无法找到模型文件: {model_path}")
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # 设置仿真时间步
        self.model.opt.timestep = self.sim_dt
        
    def _setup_joint_indices(self):
        """设置关节索引映射"""
        self.num_actions = len(self.JOINT_NAMES)
        
        # 获取关节在 qpos 中的索引
        self.joint_qpos_indices = []
        self.joint_qvel_indices = []
        self.actuator_indices = []
        
        for joint_name in self.JOINT_NAMES:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id == -1:
                raise ValueError(f"找不到关节: {joint_name}")
            
            qpos_adr = self.model.jnt_qposadr[joint_id]
            qvel_adr = self.model.jnt_dofadr[joint_id]
            
            self.joint_qpos_indices.append(qpos_adr)
            self.joint_qvel_indices.append(qvel_adr)
        
        # 获取 actuator 索引
        for i, joint_name in enumerate(self.JOINT_NAMES):
            actuator_name = joint_name.replace("_joint", "")
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            if act_id == -1:
                # 尝试直接用关节名
                act_id = i
            self.actuator_indices.append(act_id)
        
        # 获取脚部 body 索引
        self.foot_body_ids = []
        for link_name in self.FOOT_LINKS:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, link_name)
            if body_id != -1:
                self.foot_body_ids.append(body_id)
        
        # 躯干 body 索引
        self.torso_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.pelvis_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        
        # 默认关节位置
        self.default_joint_pos = np.array([self.DEFAULT_JOINT_POS[name] for name in self.JOINT_NAMES])
        
    def _setup_spaces(self):
        """设置观测和动作空间"""
        # 计算观测维度
        # 基础观测 (单步):
        #   - base_ang_vel: 3
        #   - projected_gravity: 3
        #   - velocity_commands: 3
        #   - joint_pos_rel: 29
        #   - joint_vel_rel: 29
        #   - last_action: 29
        #   - mode_flag: 4 (one-hot)
        #   - height_scan (条件): 187 or 0
        
        single_obs_dim = 3 + 3 + 3 + self.num_actions + self.num_actions + self.num_actions + self.num_modes
        if self.use_height_scan:
            single_obs_dim += self.height_scan_points
        
        # 考虑历史
        self.obs_dim = single_obs_dim * self.history_length
        
        # 观测空间
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        
        # 动作空间 (关节位置偏移)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32
        )
        
    def _init_history_buffer(self):
        """初始化历史观测缓冲"""
        single_obs_dim = self.obs_dim // self.history_length
        self.obs_history = np.zeros((self.history_length, single_obs_dim), dtype=np.float32)
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境"""
        super().reset(seed=seed)
        
        # 重置 MuJoCo 数据
        mujoco.mj_resetData(self.model, self.data)
        
        # 随机选择模式
        self.current_mode = np.random.choice(self.num_modes, p=self.mode_probabilities)
        
        # 设置初始姿态
        self._reset_robot_state()
        
        # 随机速度命令
        self._sample_velocity_command()
        
        # 域随机化
        if self.domain_randomization:
            self._apply_domain_randomization()
        
        # 重置内部状态
        self.step_count = 0
        self.last_action = np.zeros(self.num_actions)
        self.prev_action = np.zeros(self.num_actions)
        self.prev_base_height = self._get_base_height()
        
        # 重置历史缓冲
        self._init_history_buffer()
        
        # 前进仿真确保稳定
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        # 获取初始观测
        obs = self._get_observation()
        info = {"mode": self.current_mode}
        
        return obs, info
    
    def _reset_robot_state(self):
        """重置机器人状态"""
        # 设置基座位置和姿态
        self.data.qpos[0:3] = [0.0, 0.0, 0.793]  # 位置
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # 四元数 (wxyz)
        
        # 添加随机扰动
        self.data.qpos[0] += np.random.uniform(-0.5, 0.5)
        self.data.qpos[1] += np.random.uniform(-0.5, 0.5)
        yaw_noise = np.random.uniform(-np.pi, np.pi)
        # 简化: 只添加 yaw 扰动
        self.data.qpos[3:7] = self._euler_to_quat(0, 0, yaw_noise)
        
        # 设置关节位置
        for i, idx in enumerate(self.joint_qpos_indices):
            noise = np.random.uniform(0.5, 1.5) if self.domain_randomization else 1.0
            self.data.qpos[idx] = self.default_joint_pos[i] * noise
        
        # 设置速度 (带噪声)
        self.data.qvel[:] = 0.0
        if self.domain_randomization:
            self.data.qvel[0:3] = np.random.uniform(-0.5, 0.5, 3)
            self.data.qvel[3:6] = np.random.uniform(-0.5, 0.5, 3)
    
    def _euler_to_quat(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """欧拉角转四元数 (wxyz)"""
        cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
        cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
        cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return np.array([w, x, y, z])
    
    def _sample_velocity_command(self):
        """采样速度命令"""
        self.velocity_command = np.array([
            np.random.uniform(*self.cmd_lin_vel_x_range),
            np.random.uniform(*self.cmd_lin_vel_y_range),
            np.random.uniform(*self.cmd_ang_vel_z_range),
        ])
    
    def _apply_domain_randomization(self):
        """应用域随机化"""
        # 随机化摩擦系数
        for geom_id in range(self.model.ngeom):
            self.model.geom_friction[geom_id, 0] = np.random.uniform(0.3, 1.0)
        
        # 可以添加更多随机化: 质量、惯性等
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行一步动作"""
        # 保存上一步动作
        self.prev_action = self.last_action.copy()
        self.last_action = action.copy()
        
        # 计算目标关节位置
        target_pos = self.default_joint_pos + action * self.action_scale
        
        # 执行 PD 控制
        for _ in range(self.control_decimation):
            self._apply_pd_control(target_pos)
            mujoco.mj_step(self.model, self.data)
        
        self.step_count += 1
        
        # 计算奖励
        reward, reward_info = self._compute_reward()
        
        # 检查终止条件
        terminated, truncated = self._check_termination()
        
        # 获取观测
        obs = self._get_observation()
        
        info = {
            "mode": self.current_mode,
            "step": self.step_count,
            **reward_info,
        }
        
        # 更新前一步高度
        self.prev_base_height = self._get_base_height()
        
        return obs, reward, terminated, truncated, info
    
    def _apply_pd_control(self, target_pos: np.ndarray):
        """应用 PD 控制"""
        for i, (qpos_idx, qvel_idx, act_idx) in enumerate(
            zip(self.joint_qpos_indices, self.joint_qvel_indices, self.actuator_indices)
        ):
            pos_error = target_pos[i] - self.data.qpos[qpos_idx]
            vel = self.data.qvel[qvel_idx]
            torque = self.kp * pos_error - self.kd * vel
            
            # 裁剪力矩到执行器限制
            ctrl_range = self.model.actuator_ctrlrange[act_idx]
            torque = np.clip(torque, ctrl_range[0], ctrl_range[1])
            
            self.data.ctrl[act_idx] = torque
    
    def _get_observation(self) -> np.ndarray:
        """获取观测"""
        # 基座角速度 (在身体坐标系)
        base_ang_vel = self._get_base_angular_velocity() * 0.2
        
        # 重力投影
        projected_gravity = self._get_projected_gravity()
        
        # 速度命令
        vel_cmd = self.velocity_command.copy()
        
        # 关节位置偏差
        joint_pos = self._get_joint_positions()
        joint_pos_rel = joint_pos - self.default_joint_pos
        
        # 关节速度
        joint_vel = self._get_joint_velocities() * 0.05
        
        # 上一步动作
        last_action = self.last_action.copy()
        
        # 模式标志 (one-hot)
        mode_flag = np.zeros(self.num_modes)
        mode_flag[self.current_mode] = 1.0
        
        # 组合单步观测
        single_obs = np.concatenate([
            base_ang_vel,       # 3
            projected_gravity,  # 3
            vel_cmd,            # 3
            joint_pos_rel,      # 29
            joint_vel,          # 29
            last_action,        # 29
            mode_flag,          # 4
        ])
        
        # Height scan (条件)
        if self.use_height_scan:
            # 盲模式 (0, 2) 时置零
            if self.current_mode in [0, 2]:
                height_scan = np.zeros(self.height_scan_points)
            else:
                height_scan = self._get_height_scan()
            
            # 添加噪声
            height_scan += np.random.uniform(-0.1, 0.1, self.height_scan_points)
            height_scan = np.clip(height_scan, -1.0, 1.0)
            
            single_obs = np.concatenate([single_obs, height_scan])
        
        # 更新历史缓冲
        self.obs_history = np.roll(self.obs_history, -1, axis=0)
        self.obs_history[-1] = single_obs
        
        # 展平历史
        obs = self.obs_history.flatten().astype(np.float32)
        
        return obs
    
    def _get_base_angular_velocity(self) -> np.ndarray:
        """获取基座角速度 (在身体坐标系)"""
        # 世界坐标系角速度
        world_ang_vel = self.data.qvel[3:6].copy()
        
        # 转换到身体坐标系
        quat = self.data.qpos[3:7]  # wxyz
        rot_mat = np.zeros((3, 3))
        mujoco.mju_quat2Mat(rot_mat.flatten(), quat)
        
        body_ang_vel = rot_mat.T @ world_ang_vel
        
        return body_ang_vel
    
    def _get_projected_gravity(self) -> np.ndarray:
        """获取重力在身体坐标系的投影"""
        quat = self.data.qpos[3:7]  # wxyz
        rot_mat = np.zeros((3, 3))
        mujoco.mju_quat2Mat(rot_mat.flatten(), quat)
        
        # 世界坐标系重力 [0, 0, -1]
        gravity = np.array([0.0, 0.0, -1.0])
        
        # 投影到身体坐标系
        projected = rot_mat.T @ gravity
        
        return projected
    
    def _get_joint_positions(self) -> np.ndarray:
        """获取关节位置"""
        return np.array([self.data.qpos[idx] for idx in self.joint_qpos_indices])
    
    def _get_joint_velocities(self) -> np.ndarray:
        """获取关节速度"""
        return np.array([self.data.qvel[idx] for idx in self.joint_qvel_indices])
    
    def _get_base_linear_velocity(self) -> np.ndarray:
        """获取基座线速度"""
        return self.data.qvel[0:3].copy()
    
    def _get_base_height(self) -> float:
        """获取基座高度"""
        return self.data.qpos[2]
    
    def _get_height_scan(self) -> np.ndarray:
        """获取高度扫描 (简化实现)"""
        # 在实际实现中，这里应该使用 ray casting
        # 简化为返回零向量，用户可以扩展
        return np.zeros(self.height_scan_points)
    
    def _compute_reward(self) -> Tuple[float, Dict[str, float]]:
        """计算奖励"""
        reward_info = {}
        total_reward = 0.0
        
        # ========== 任务奖励 ==========
        # 速度跟踪奖励
        base_vel = self._get_base_linear_velocity()
        base_ang_vel = self.data.qvel[3:6]
        
        # 转换到 yaw frame
        yaw = self._get_yaw()
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        vel_x = base_vel[0] * cos_yaw + base_vel[1] * sin_yaw
        vel_y = -base_vel[0] * sin_yaw + base_vel[1] * cos_yaw
        
        lin_vel_error = (vel_x - self.velocity_command[0])**2 + (vel_y - self.velocity_command[1])**2
        track_lin_vel = np.exp(-lin_vel_error / 0.25) * 1.0
        reward_info["track_lin_vel"] = track_lin_vel
        total_reward += track_lin_vel
        
        ang_vel_error = (base_ang_vel[2] - self.velocity_command[2])**2
        track_ang_vel = np.exp(-ang_vel_error / 0.25) * 0.5
        reward_info["track_ang_vel"] = track_ang_vel
        total_reward += track_ang_vel
        
        # 向上进展奖励 (楼梯模式)
        if self.current_mode in [2, 3]:
            height_progress = (self._get_base_height() - self.prev_base_height) / self.control_dt
            upward_progress = np.clip(height_progress, 0, 1) * 0.5
            reward_info["upward_progress"] = upward_progress
            total_reward += upward_progress
        
        # ========== 存活奖励 ==========
        alive = 0.15
        reward_info["alive"] = alive
        total_reward += alive
        
        # ========== 正则化惩罚 ==========
        # Z 轴速度惩罚
        lin_vel_z = base_vel[2]**2 * (-2.0)
        reward_info["lin_vel_z"] = lin_vel_z
        total_reward += lin_vel_z
        
        # XY 角速度惩罚
        ang_vel_xy = (base_ang_vel[0]**2 + base_ang_vel[1]**2) * (-0.05)
        reward_info["ang_vel_xy"] = ang_vel_xy
        total_reward += ang_vel_xy
        
        # 姿态惩罚
        projected_gravity = self._get_projected_gravity()
        flat_orientation = (projected_gravity[0]**2 + projected_gravity[1]**2) * (-1.0)
        reward_info["flat_orientation"] = flat_orientation
        total_reward += flat_orientation
        
        # 动作变化率惩罚
        action_rate = np.sum((self.last_action - self.prev_action)**2) * (-0.01)
        reward_info["action_rate"] = action_rate
        total_reward += action_rate
        
        # 关节加速度惩罚
        joint_vel = self._get_joint_velocities()
        joint_acc = np.sum(joint_vel**2) * (-2.5e-7)
        reward_info["joint_acc"] = joint_acc
        total_reward += joint_acc
        
        # ========== 姿态奖励 ==========
        # 高度惩罚
        base_height = self._get_base_height()
        target_height = 0.72
        height_error = (base_height - target_height)**2 * (-0.5)
        reward_info["base_height"] = height_error
        total_reward += height_error
        
        # ========== 步态奖励 ==========
        # 脚部空中时间奖励 (简化)
        feet_air_time = 0.0  # 需要接触检测支持
        reward_info["feet_air_time"] = feet_air_time
        total_reward += feet_air_time
        
        # ========== 安全惩罚 ==========
        # 不期望的接触惩罚 (简化)
        undesired_contacts = 0.0  # 需要接触检测支持
        reward_info["undesired_contacts"] = undesired_contacts
        total_reward += undesired_contacts
        
        return total_reward, reward_info
    
    def _get_yaw(self) -> float:
        """从四元数获取 yaw 角"""
        quat = self.data.qpos[3:7]  # wxyz
        w, x, y, z = quat
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return yaw
    
    def _check_termination(self) -> Tuple[bool, bool]:
        """检查终止条件"""
        terminated = False
        truncated = False
        
        # 超时
        if self.step_count >= self.max_episode_steps:
            truncated = True
        
        # 跌倒检测
        base_height = self._get_base_height()
        if base_height < 0.3:
            terminated = True
        
        # 姿态过大
        projected_gravity = self._get_projected_gravity()
        if projected_gravity[2] > -0.5:  # 倾斜过大
            terminated = True
        
        return terminated, truncated
    
    def render(self):
        """渲染"""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            # 返回渲染图像
            width, height = 640, 480
            renderer = mujoco.Renderer(self.model, height, width)
            renderer.update_scene(self.data)
            return renderer.render()
        
        return None
    
    def close(self):
        """关闭环境"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# 注册 Gym 环境
def register_env():
    """注册 Gym 环境"""
    gym.register(
        id="G1-29DOF-v0",
        entry_point="envs.g1_29dof_env:G1_29DOF_Env",
        max_episode_steps=1000,
    )


if __name__ == "__main__":
    # 测试环境
    env = G1_29DOF_Env(render_mode="human")
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action shape: {env.action_space.shape}")
    print(f"Initial mode: {info['mode']}")
    
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
