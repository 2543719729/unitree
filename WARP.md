# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Overview

This repository contains reinforcement learning environments for Unitree robots built on top of Isaac Lab. It supports training locomotion policies for **Go2** (quadruped), **H1** (humanoid), and **G1-29dof** (humanoid) robots using RSL-RL (PPO algorithm).

The repository structure:
- `unitree_rl_lab/` - Main RL training code and environments
- `IsaacLab/` - NVIDIA Isaac Lab framework (dependency)
- `unitree_ros/` - Robot URDF descriptions from Unitree

## Common Commands

### Environment Setup
```bash
# Activate conda environment (required before any operations)
conda activate env_isaaclab

# Install unitree_rl_lab in editable mode
./unitree_rl_lab/unitree_rl_lab.sh -i
```

### Training
```bash
# Train with default settings (headless)
./unitree_rl_lab/unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity

# Or directly with Python
python unitree_rl_lab/scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity

# Train with specific parameters
python unitree_rl_lab/scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity --num_envs 4096 --max_iterations 10000

# Resume training from checkpoint
python unitree_rl_lab/scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity --resume --load_run <run_folder_name>

# Train with video recording
python unitree_rl_lab/scripts/rsl_rl/train.py --task Unitree-G1-29dof-Velocity --video --video_interval 2000

# Use wandb/tensorboard logging
python unitree_rl_lab/scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity --logger wandb
```

### Inference/Play
```bash
# Play trained policy with visualization
./unitree_rl_lab/unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity

# Or directly
python unitree_rl_lab/scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity

# Play specific checkpoint
python unitree_rl_lab/scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity --checkpoint <path_to_model.pt>

# Record video during play
python unitree_rl_lab/scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity --video --video_length 500
```

### Utility Commands
```bash
# List all available tasks
./unitree_rl_lab/unitree_rl_lab.sh -l

# Run with GUI (remove --headless)
python unitree_rl_lab/scripts/rsl_rl/train.py --task Unitree-G1-29dof-Velocity
```

### Sim2Sim Deployment (Mujoco)
```bash
# Build the robot controller
cd unitree_rl_lab/deploy/robots/g1_29dof
mkdir build && cd build
cmake .. && make

# Run in Mujoco simulation
./g1_ctrl
```

### Sim2Real Deployment
```bash
# Deploy to physical robot (replace eth0 with your network interface)
./g1_ctrl --network eth0
```

## Available Tasks (G1 29-DOF Training Stages)

The G1 robot uses a 4-stage curriculum training approach:

1. **`Unitree-G1-29dof-Velocity`** - Blind walking on flat terrain (no height_scan)
2. **`Unitree-G1-29dof-Velocity-HeightScan`** - Walking with height scanner on flat terrain
3. **`Unitree-G1-29dof-Stair-Blind`** - Blind stair climbing (no height_scan)
4. **`Unitree-G1-29dof-Stair`** - Stair climbing with height scanner

Additional tasks:
- **`Unitree-G1-29dof-Unified`** - Multi-mode unified policy
- **`Unitree-G1-29dof-Marching`** - Marching in place

## Code Architecture

### Source Structure
```
unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/
├── assets/robots/          # Robot configurations (unitree.py, unitree_actuators.py)
├── tasks/
│   ├── locomotion/
│   │   ├── agents/         # PPO algorithm configs (rsl_rl_ppo_cfg.py)
│   │   ├── mdp/            # MDP components:
│   │   │   ├── commands/   # Velocity command generation
│   │   │   ├── curriculums.py    # Training curriculum
│   │   │   ├── events.py         # Domain randomization, resets
│   │   │   ├── observations.py   # Observation functions
│   │   │   ├── rewards.py        # Reward functions
│   │   │   └── terminations.py   # Episode termination
│   │   └── robots/         # Per-robot environment configs
│   │       ├── g1/29dof/   # G1 humanoid configs
│   │       ├── go2/        # Go2 quadruped configs
│   │       └── h1/         # H1 humanoid configs
│   └── mimic/              # Imitation learning tasks
└── utils/                  # Deployment config export utilities
```

### Key Design Patterns

**Environment Configuration**: Each robot/task has a `*_env_cfg.py` file defining:
- `RobotSceneCfg`: Terrain, robot, sensors, lighting
- `EventCfg`: Domain randomization (mass, friction, external forces)
- `CommandsCfg`: Velocity command generation
- `ActionsCfg`: Joint position control action space
- `ObservationsCfg`: Policy and critic observations
- `RewardsCfg`: Multi-term reward functions
- `TerminationsCfg`: Episode termination conditions
- `CurriculumCfg`: Training difficulty progression

**Task Registration**: Tasks are registered via `gymnasium.register()` in each robot's `__init__.py`:
```python
gym.register(
    id="Unitree-G1-29dof-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
```

**MDP Functions**: All MDP components in `mdp/` follow the pattern:
- First argument is always `env`
- Returns `torch.Tensor` with shape `(num_envs, ...)`
- Functions should be pure (no side effects except necessary state caching)

### Robot Model Configuration

Robot models are defined in `assets/robots/unitree.py`. Two paths must be configured:
```python
UNITREE_MODEL_DIR = "e:/Aunitree/unitree_model"  # USD model files
UNITREE_ROS_DIR = "e:/Aunitree/unitree_ros"      # URDF model files
```

The G1-29DOF robot uses URDF by default:
```python
spawn=UnitreeUrdfFileCfg(
    asset_path=f"{UNITREE_ROS_DIR}/robots/g1_description/g1_29dof_rev_1_0.urdf",
)
```

### PPO Algorithm Configuration

Two PPO configs are provided in `agents/rsl_rl_ppo_cfg.py`:
- `BasePPORunnerCfg`: Standard config for most tasks
- `StairBlindPPORunnerCfg`: Optimized for blind stair climbing with:
  - Longer trajectories (32 steps vs 24)
  - Larger networks [512, 256, 256, 128]
  - Higher discount factor (0.995)
  - More learning epochs (8 vs 5)

### Training Output

Training logs and checkpoints are saved to:
```
unitree_rl_lab/logs/rsl_rl/<experiment_name>/<timestamp>/
├── params/          # Environment and agent configs (YAML)
├── videos/          # Training videos (if enabled)
├── model_*.pt       # Checkpoints
└── exported/        # JIT/ONNX exported models (after play.py)
```

## Development Notes

- Activate `env_isaaclab` conda environment before running any commands
- The codebase uses Chinese comments in some files; key architectural docs are in Chinese
- Code follows `isort` and `pyright` configurations in `pyproject.toml`
- All observations should return tensors with shape `(num_envs, ...)`
- Domain randomization parameters are defined in `EventCfg` within each environment config
