"""
===============================================================================
MDP（马尔可夫决策过程）组件模块
===============================================================================

本模块定义了强化学习环境的核心 MDP 组件。

主要模块:
    - commands: 速度命令生成和课程学习配置
    - curriculums: 训练难度自适应调整
    - events: 环境重置、域随机化、模式切换事件
    - observations: 观测函数（步态相位、模式标志、高度扫描等）
    - rewards: 奖励函数（跟踪、姿态、步态、楼梯攀爬等）

设计原则:
    - 所有函数接受 env 作为第一个参数
    - 函数应该是纯函数（无副作用，除了必要的状态缓存）
    - 返回值为 torch.Tensor，形状为 (num_envs, ...)

扩展方法:
    继承 Isaac Lab 的基础 MDP 函数，并添加 Unitree 专用的函数。
===============================================================================
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403

# 导入自定义 MDP 组件
from .commands import *  # noqa: F401, F403
from .curriculums import *  # noqa: F401, F403
from .events import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
