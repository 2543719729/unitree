## Unitree G1 自动上楼梯任务：配置修改总体规划

> 本文基于《机器人上楼梯配置调研》与现有 `Unitree-G1-29dof-Velocity` 任务代码（`velocity_env_cfg.py` 等），给出一个**从当前水平到“能在仿真中稳定上楼梯”**的详细配置修改路线图。本文只做规划，不直接修改代码，方便你逐步选择和实施。

---

## 0. 当前基线与目标假设

- **当前基线**（已存在）
  - 任务：`Unitree-G1-29dof-Velocity`（速度跟踪，平地/简单地形）。
  - 主要配置文件：
    - `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py`
    - `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/*.py`
  - 地形：`TerrainGeneratorCfg` + `MeshPlaneTerrainCfg`（平地），暂未启用楼梯。
  - 观测：本体状态 + 速度命令，`height_scan` 尚未加入策略观测。
  - 奖励：偏向“速度跟踪 + 行走稳定”，未显式鼓励“向上攀爬楼梯”。

- **目标**
  1. 在 Isaac Lab 中，G1 能在**固定参数楼梯**场景中稳定上楼梯（不摔倒、不严重打滑）。
  2. 观测空间中包含**楼梯高度信息**（先用 RayCaster 高度扫描近似 LiDAR）。
  3. 奖励函数显式鼓励“**往上/往前通过楼梯**”的进展，并强约束平衡与足端安全。
  4. 训练过程中，采用**课程学习**：从低阶梯 → 标准楼梯 → 更高/更陡楼梯。
  5. 为后续 sim2real 做准备：控制频率、传感信息形式尽量与真实 G1 接近。

接下来按“阶段–任务–需要修改的配置项”分层规划。

---

## 1. 阶段一：在 Isaac Lab 中引入标准楼梯地形

### 1.1 选择地形类型与参数

**文件与类：**
- `velocity_env_cfg.py`
  - `COBBLESTONE_ROAD_CFG: TerrainGeneratorCfg`
  - `RobotSceneCfg.terrain: TerrainImporterCfg`

**规划：**

1. **保留平地配置作为基线**，但为楼梯任务单独创建一个新的 `TerrainGeneratorCfg`，例如：
   - 新增：`STAIR_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(...)`
   - 放在 `COBBLESTONE_ROAD_CFG` 附近，便于切换。

2. 在 `STAIR_TERRAIN_CFG.sub_terrains` 中使用 **mesh 楼梯地形**：
   - `terrain_gen.MeshPyramidStairsTerrainCfg`：上楼（从四周向中心上升，或沿一个方向上升）。
   - （可选）`terrain_gen.MeshInvertedPyramidStairsTerrainCfg`：从中心向外下楼。

3. **楼梯几何参数建议**（与报告中“典型室内楼梯”一致）：
   - `step_height_range = (0.15, 0.20)`  
     - 室内阶高通常 15–20cm，先从略低值开始：如 `(0.12, 0.18)`，再课程增加。
   - `step_width = 0.30`  
     - 台阶进深 30cm 左右。
   - `platform_width = 2.0`  
     - 顶部/中间平台宽度 2m，保证机器人有缓冲区。
   - `border_width = 1.0`  
     - 为楼梯周围留 1m 平台，防止机器人走出可视范围立即跌落。
   - `holes = False`  
     - 初期不要空洞，以降低难度。

4. `TerrainGeneratorCfg` 其它参数建议：
   - `size = (8.0, 8.0)`：单块地形足够容纳几级楼梯 + 平台。
   - `num_rows` / `num_cols`：
     - 训练：保持 `num_rows=9, num_cols=21` 以支撑课程学习。
     - 可视化小批量测试（`RobotPlayEnvCfg`）：将 `num_rows=2, num_cols=4~10` 加快加载。
   - `difficulty_range = (0.0, 1.0)`：后续用于**阶高课程学习**。

5. **在 `RobotSceneCfg.terrain` 中切换使用楼梯地形**：
   - `terrain_generator=STAIR_TERRAIN_CFG`。
   - 保持 `terrain_type="generator"`，`max_init_terrain_level` 根据楼梯行数设置（如 `num_rows-1`）。

> 决策点 A：
> - 若你希望 **只在楼梯上训练**，则把原平地 `COBBLESTONE_ROAD_CFG` 留作参考，不在当前任务中使用。
> - 若希望混合“平地 + 楼梯”，可以在一个 `TerrainGeneratorCfg` 下配置多个 `sub_terrains`，并用不同 `proportion` 同时生成（但会增加训练复杂度）。

---

## 2. 阶段二：观测空间中加入楼梯高度信息（模拟 LiDAR/深度相机）

报告建议：仿真中用**RayCaster 高度扫描**近似 LiDAR/深度映射，以降低训练代价并便于 sim2real（后续用点云/深度图处理成同样的高度数组）。

### 2.1 传感器配置检查

**文件与类：**
- `velocity_env_cfg.py`
  - `RobotSceneCfg.height_scanner: RayCasterCfg`

**规划：**

1. 确认已经存在：
   - `prim_path="{ENV_REGEX_NS}/Robot/torso_link"` 或 `base`：建议使用接近质心的位置。
   - `offset.pos = (0.0, 0.0, 20.0)`：从高处向下扫描。
   - `ray_alignment="yaw"`：与机器人航向对齐即可。
   - `pattern_cfg = patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0])`：
     - 分辨率 0.1m、范围 1.6m × 1.0m，覆盖前方若干台阶。
   - `mesh_prim_paths=["/World/ground"]`：必须与地形 prim 匹配。

2. 若希望更聚焦于楼梯前方：
   - 可略微向前偏移：`offset.pos=(0.2, 0.0, 20.0)`，让射线更多落在楼梯上。

### 2.2 将 `height_scan` 加入观测空间

**文件与类：**
- `velocity_env_cfg.py`
  - `ObservationsCfg.PolicyCfg`
  - `ObservationsCfg.CriticCfg`
- `isaaclab/envs/mdp/observations.py`
  - `height_scan(env, sensor_cfg, offset=0.5)`

**规划：**

1. 在 **策略观测组** 中添加地形高度扫描：
   - 在 `ObservationsCfg.PolicyCfg` 中新增：
     - `height_scan = ObsTerm(func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner")}, noise=Unoise(...), clip=(-1.0, 1.0))`
   - 噪声与裁剪：
     - `noise`：`n_min=-0.1, n_max=0.1`，模拟 LiDAR 噪声。
     - `clip=(-1.0, 1.0)`：确保数值稳定。
   - 保持 `history_length` 与其他观测一致（当前为 5），形成时序信息。

2. 在 **评论家观测组** 中也加入高度扫描（无噪声）：
   - 方便评论家利用更精确的地形信息学习。

3. 注意：加入 `height_scan` 后，策略输入维度会显著增大：
   - 若使用 `GridPatternCfg(resolution=0.1, size=[1.6,1.0])`，大约 17×11=187 维；再乘以 `history_length`。
   - 需要检查 / 调整 PPO/RSL-RL 的网络宽度（隐藏层尺寸）以容纳更多输入特征。

> 决策点 B：
> - 若你打算**短期内只做“盲爬楼”（不看地形，只靠触觉/IMU）**，可以暂时不启用 `height_scan`，将重点放在奖励 shaping 和楼梯几何参数上；后续在需要更强泛化能力时再加入。
> - 若目标是**最终实机部署**，强烈建议从一开始就设计与真实传感管线匹配的 `height_scan`/点云特征格式。

---

## 3. 阶段三：专门针对“上楼梯”的奖励函数设计

报告指出：上楼梯任务的关键是**进展奖励 + 平衡约束 + 足端安全**。当前 `RewardsCfg` 偏向“速度跟踪平地行走”，需要做定制化调整。

**文件与类：**
- `velocity_env_cfg.py`: `RewardsCfg`
- `tasks/locomotion/mdp/rewards.py` & `isaaclab/envs/mdp/rewards.py`

### 3.1 进展奖励（向上 + 向前）

1. **纵向进展奖励（向上爬）**
   - 新增奖励项：
     - `stair_upward_progress = RewTerm(func=<自定义>, weight>0)`
   - 实现思路（在 `mdp/rewards.py` 中实现函数）：
     - 利用 `asset.data.root_pos_w[:, 2]`（高度）或质心高度，计算每步高度增量 `Δh`；
     - 对正向高度增量给正奖励，如：`reward = clamp(Δh, 0, h_max) / h_norm`；
     - 为避免“跳起来再落回去骗奖励”，可以使用**累计高度**（当前高度相对于起始高度的差）并加入折扣。

2. **前向进展奖励（沿楼梯方向）**
   - 若楼梯沿世界 x 轴方向，可以使用 `root_pos_w[:, 0]` 的增量；
   - 或在楼梯方向定义单位向量 `e_stair`，对 `Δpos · e_stair` 进行奖励；
   - 注意：若仍需要“速度命令”兼容，需在奖励中区分“爬楼模式”与“平地行走模式”。

### 3.2 保留/调整的稳定性与正则项

当前已存在的项中，多数仍适用楼梯场景：

- `flat_orientation_l2`：惩罚身体倾斜，需保留，但在上楼时**允许略向前倾**，可以放宽权重或仅约束 roll（左右倾斜）。
- `base_height_l2`：原本以固定高度为目标，在楼梯场景中需要：
  - 使用 `mdp.base_height_l2` 的 **RayCaster 修正版本**（加 `sensor_cfg`，将地形高度作为参考），否则会惩罚爬升。
  - 对目标高度 `target_height` 设置为“相对于脚下台阶”的高度，而非世界绝对高度。
- 关节速度/加速度/能量等正则项保持不变即可。

### 3.3 足端安全：避免踢台阶棱角 + 足底稳定接触

1. **抬脚高度奖励**（当前已有 `feet_clearance`）：
   - 需要检查参数是否匹配楼梯高度：
     - `target_height ≈ step_height + margin（如 +5~10cm）`；
     - `std` 和 `tanh_mult` 控制曲线陡峭程度，可以在实际训练中根据是否频繁踢到阶角调整。

2. **脚部滑动惩罚**（`feet_slide`）：
   - 在楼梯场景中尤为重要：防止脚踏上台阶后发生大幅滑动。
   - 若发现训练中仍易滑，可以适当增大该项权重。

3. **非期望碰撞惩罚**（`undesired_contacts`）：
   - 保持脚以外的部位与楼梯接触即惩罚；
   - 可在楼梯训练初期降低权重，避免策略“过度保守不敢迈步”；待基本会走后再升高权重。

### 3.4 课程学习中的奖励权重渐变

规划：
1. 初期：主要奖励“保持平衡 + 迈上低台阶”，进展奖励权重可小一些，稳定性更重要；
2. 中后期：逐步放大奖励中“纵向进展”的权重，引导策略走得更快、更果断；
3. 可以通过 `CurriculumCfg` 或直接在训练脚本中调整 `reward_scales` 实现分阶段权重切换。

> 决策点 C：
> - 优先选择“增量式改动”：先只加一个简单的 `upward_progress` 奖励，观察是否有明显上楼意图，再慢慢增加复杂奖励，避免一次性改动太多无从调试。

---

## 4. 阶段四：命令空间与任务模式设计

报告中区分：
- 室内导航：先通过 SLAM/NVBlox/Nav2 导航到楼梯口；
- 爬楼控制：在楼梯模式下，用专门的控制策略（FSM 或 RL）攀登楼梯。

在当前 Isaac Lab 任务中，可以先只做“**纯爬楼任务**”，暂不考虑地图导航。

**文件与类：**
- `velocity_env_cfg.py`: `CommandsCfg`

### 4.1 为楼梯任务定义更合适的速度命令

1. 在楼梯爬升任务中：
   - 不再需要大范围的横向和旋转速度命令；
   - 重点在“沿楼梯方向缓慢前进 + 稍微调整姿态”。

2. 调整 `CommandsCfg.base_velocity`：
   - 训练爬楼时可以使用：
     - `lin_vel_x`：略正的小范围，如 `(0.1, 0.5)`，鼓励向前/向上走；
     - `lin_vel_y`：接近 0，如 `(-0.05, 0.05)`，避免横向摆动过大；
     - `ang_vel_z`：`(-0.1, 0.1)`，只用于轻微对正楼梯。
   - 若希望更简单，可在楼梯任务中**不使用速度命令**（恒定目标），让 RL 只学“稳定上楼”这一单一目标。

3. 根据是否需要“多任务”：
   - **方案 1：专用楼梯任务 env**：
     - 新建一个 `StairClimbEnvCfg`/`StairPlayEnvCfg`，其 `CommandsCfg` 固定为适合上楼的命令范围。
   - **方案 2：在现有 velocity 任务中增加“爬楼模式标志”**：
     - 在观测中加入一个二值标志位 `is_climbing_mode`；
     - 在奖励和命令采样中根据该标志切换不同逻辑（复杂度较高，建议后做）。

---

## 5. 阶段五：课程学习与地形难度规划

**文件与类：**
- `velocity_env_cfg.py`: `CurriculumCfg`
- `tasks/locomotion/mdp/curriculums.py`

### 5.1 楼梯几何课程

设计思路：从**低矮、宽台阶** 到 **标准楼梯** 再到 **略超标准的困难楼梯**。

1. 在 `STAIR_TERRAIN_CFG` 中，将 `difficulty` 映射到：
   - `step_height_range(d)`：
     - 低难度：`0.08–0.12m`
     - 中难度：`0.12–0.16m`
     - 高难度：`0.16–0.20m`
   - `step_width(d)`：
     - 低难度：`0.35m`（更宽，容易踩准）
     - 高难度：`0.25–0.30m`（更窄，需要更准落脚）。

2. 在 `CurriculumCfg.terrain_levels` 对应函数中：
   - 参考 `mdp.terrain_levels_vel` 的写法，根据机器人成功率或奖励自动提升 `terrain_level`；
   - 将 `terrain_level` 转成上述几何参数，逐级增加难度。

### 5.2 课程节奏

规划建议：

1. **阶段 0：平地行走（已完成）**
2. **阶段 1：单级“高台阶”**
   - 仅在平地前放置一个孤立的台阶（可用简单方块代替楼梯）；
   - 检查机器人是否能抬脚上这一阶并在上方保持平衡。
3. **阶段 2：两级楼梯**
   - 连续两级台阶，测试机器人是否能在连续多步中保持稳定。
4. **阶段 3：完整楼梯（3–5 级）**
   - 使用 `MeshPyramidStairsTerrainCfg` 生成的小型楼梯；
   - 在此基础上做 domain randomization（见下一节）。

> 决策点 D：
> - 你可以选择先用“手工建模的单级/两级阶”做快速实验（不改 TerrainGeneratorCfg），只在 USD 场景中摆放 box asset；当策略初步可行后，再把逻辑迁移到“程序生成楼梯”地形中。

---

## 6. 阶段六：域随机化与鲁棒性增强（为 sim2real 做准备）

**文件与类：**
- `velocity_env_cfg.py`: `EventCfg`

报告强调，为了减少 sim2real gap，需要对摩擦、质量、传感噪声等进行**域随机化**。你当前的 `EventCfg` 已经包含部分随机化，可以有针对性增强：

1. **摩擦随机化**
   - 已有：`physics_material` 事件配置，随机 `static_friction` / `dynamic_friction`。
   - 在楼梯任务中可略微扩大范围，尤其是向下调整一部分，以模拟“滑一点的楼梯”。

2. **质量与重心随机化**
   - `add_base_mass`：继续保留，模拟背负物/载荷。
   - 若发现爬楼时重心偏差敏感，可以额外对上肢关节质量做轻微随机化（需要在 `SceneEntityCfg` 中指定 body_names）。

3. **外力扰动**
   - `push_robot`：可以在楼梯上适当减小干扰强度／频率，避免还没学会走就被推倒；
   - 待策略成熟后，再逐步加大，测试鲁棒性。

4. **传感噪声**
   - 当前在观测配置中已有 `Unoise`；
   - 对 `height_scan` 观测建议保持适度噪声，以免策略过度依赖理想高度值。

---

## 7. 阶段七：仿真到实机部署的接口预规划（只做配置层面的准备）

虽然真正 sim2real 需要 ROS2 + Unitree SDK + Isaac ROS/NVBlox，需要单独工程，但在当前阶段可以在 Isaac Lab 配置中**提前对齐以下几点**：

1. **控制频率**
   - 确认 `RobotEnvCfg.__post_init__` 中：
     - `self.sim.dt`、`self.decimation` 组合出的策略频率与未来现实控制频率接近（例如 50Hz）。

2. **观测格式**
   - 保持策略使用的 `height_scan`/IMU/关节状态形式，与未来在 ROS 中处理出来的观测向量一致：
     - 射线数量、排序方式不轻易变动；
     - 对输入做标准化（均值/方差）时，记录下这些统计量，以便部署时复用。

3. **动作输出范围**
   - `ActionsCfg.JointPositionAction.scale` 等参数与 Unitree SDK 中期望的关节指令范围匹配；
   - 对动作进行合理的剪裁，防止 RL 输出过大命令在实机上伤害硬件。

---

## 8. 建议的实施顺序总结（可执行 Checklist）

1. **环境与地形**
   - [ ] 在 `velocity_env_cfg.py` 中新增 `STAIR_TERRAIN_CFG`，使用 `MeshPyramidStairsTerrainCfg`；
   - [ ] 在某个测试任务（可新建 `StairEnvCfg`）中切换 `terrain_generator=STAIR_TERRAIN_CFG`；
   - [ ] 使用 `RobotPlayEnvCfg` 或单独的 PlayEnv，设置 `num_envs` 为小值，便于可视化。

2. **观测**
   - [ ] 确认 `RobotSceneCfg.height_scanner` 参数合理；
   - [ ] 在 `ObservationsCfg.PolicyCfg`/`CriticCfg` 中加入 `height_scan` 项；
   - [ ] 检查 RL 网络隐层大小，确保能处理新增观测维度。

3. **奖励**
   - [ ] 在 `mdp/rewards.py` 中实现 `stair_upward_progress` 奖励函数；
   - [ ] 修改 `base_height` 奖励为基于 RayCaster 的相对高度（不惩罚爬升）；
   - [ ] 调优 `feet_clearance` 目标高度与权重，确保不过多踢到台阶棱角。

4. **命令与任务模式**
   - [ ] 为楼梯任务缩窄 `CommandsCfg.base_velocity` 的范围，或固定前进速度；
   - [ ] 如需将楼梯与平地行走解耦，考虑新建专用 `StairClimbEnvCfg`。

5. **课程学习与随机化**
   - [ ] 在 `CurriculumCfg` 中增加围绕楼梯几何参数的课程逻辑；
   - [ ] 调整 `EventCfg` 中的摩擦、质量和外力扰动范围，形成合理的域随机化。

6. **可视化与调试**
   - [ ] 使用少量 env + 开启 `debug_vis`（地形、射线、速度箭头）观察策略学习过程；
   - [ ] 先只训练在简单楼梯上攀登，观察是否出现“有意识的上楼”行为，再逐步提升难度。

---

以上规划覆盖了从**地形建模 → 观测设计 → 奖励 shaping → 课程学习 → 域随机化 → sim2real 准备**的完整链路。你可以先选择其中一个小闭环（例如：
“只在简单 Mesh 楼梯 + 无 height_scan 下让 G1 盲走上 3 级台阶”）作为第一阶段目标，然后按本文的 Checklist 逐项落地对应配置修改。

---

# 附录：深度调研补充（基于全网检索与 GitHub 项目）

> 以下内容基于对 RSS/ICRA 论文、ETH Zurich legged_gym、RobotEra humanoid-gym、Unitree 官方仓库等的深度调研，提供**业界验证过的参数配置**和**关键研究结论**。

---

## A. 核心研究结论：盲爬楼梯（Blind Stair Climbing）

### A.1 Cassie 盲爬楼梯论文（RSS 2021）

> 论文：*"Blind Bipedal Stair Traversal via Sim-to-Real Reinforcement Learning"*  
> 作者：Siekmann et al. (Oregon State University)

**核心结论：**

1. **不需要视觉/LiDAR 也能爬楼**
   - 仅使用**本体感知（proprioceptive）**反馈（关节角度、速度、IMU）即可盲爬楼梯
   - 策略通过腿部触觉反馈自适应调整

2. **不需要修改奖励函数**
   - 只需在现有平地训练框架中**加入楼梯地形随机化**
   - 奖励函数保持与平地行走相同

3. **关键是地形随机化 + 任务随机化**
   - 训练时暴露于各种 `step_height`、`step_width` 的楼梯
   - 任务随机化增强泛化能力

4. **容错性而非完美落脚**
   - RL 策略学会"在不完美动作后及时纠正"
   - 即使踩偏、踢到阶角，也能继续前进

**对 G1 的启示：阶段一完全可以不加 `height_scan`，用盲爬方式验证基础运动能力**

---

### A.2 Vision-Based Bipedal Locomotion（2023）

**地形生成分布（训练时采样概率）：**

```python
terrain_sampling_probs = {
    "flat": 0.03,      # 平地
    "hills": 0.07,     # 起伏
    "blocks": 0.35,    # 方块障碍
    "ridges": 0.20,    # 脊地形
    "stairs": 0.35,    # 楼梯
}
# 大部分概率分配给困难地形（90%）
```

**观测空间设计：**

| 组件 | 详情 |
|------|------|
| 本体感知 | 基座姿态、角速度、关节位置/速度 |
| 地形高度图 | 1.5m × 1.0m，分辨率 5cm |
| 用户命令 | 前向/侧向速度、转向角速度 |
| 周期时钟 | sin/cos 编码的步态相位 |

**终止条件：**
- 倾斜超过 15°
- 基座高度低于 40cm
- 躯干碰撞地形

**关键 insight：地形生成分布比奖励调参更重要**

---

## B. 业界标准配置参数（legged_gym / ETH Zurich）

### B.1 地形配置

```python
# legged_gym 默认地形比例
terrain_proportions = [
    0.10,  # smooth slope
    0.10,  # rough slope
    0.35,  # stairs up
    0.25,  # stairs down
    0.20,  # discrete obstacles
]

# 课程学习
curriculum = True
max_init_terrain_level = 5
num_rows = 10
num_cols = 20
```

### B.2 奖励函数权重

```python
reward_scales = {
    # 主要跟踪
    "tracking_lin_vel": 1.0,
    "tracking_ang_vel": 0.5,

    # 稳定性惩罚
    "lin_vel_z": -2.0,
    "ang_vel_xy": -0.05,

    # 步态
    "feet_air_time": 1.0,
    "collision": -1.0,

    # 能量
    "torques": -0.00001,
    "dof_acc": -2.5e-7,
    "action_rate": -0.01,
}

only_positive_rewards = True  # 截断负奖励为 0
tracking_sigma = 0.25
```

### B.3 噪声配置

```python
noise_scales = {
    "dof_pos": 0.01,
    "dof_vel": 1.5,
    "lin_vel": 0.1,
    "ang_vel": 0.2,
    "gravity": 0.05,
    "height_measurements": 0.1,
}
```

---

## C. 域随机化最佳实践

### C.1 物理参数随机化

```python
domain_randomization = {
    # 摩擦
    "randomize_friction": True,
    "friction_range": [0.5, 1.25],

    # 质量
    "randomize_base_mass": True,
    "added_mass_range": [-1.0, 3.0],

    # 外部扰动
    "push_robots": True,
    "push_interval_s": 15,
    "max_push_vel_xy": 1.0,

    # 观测延迟（关键！）
    "add_observation_delay": True,
    "delay_ms_range": [0, 20],
}
```

### C.2 随机化优先级（从高到低）

1. 摩擦系数
2. 质量/惯性
3. 关节动力学
4. 观测噪声
5. 动作延迟

---

## D. 混合地形配置建议（平地 + 楼梯）

```python
MIXED_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    curriculum=True,
    sub_terrains={
        # 基础地形（30%）
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.15),
        "smooth_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.2)),

        # 楼梯地形（50%）
        "stairs_easy": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.08, 0.12),
            step_width=0.35,
            platform_width=2.0,
        ),
        "stairs_medium": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.12, 0.18),
            step_width=0.30,
        ),
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.10, 0.15),
            step_width=0.30,
        ),

        # 障碍地形（20%）
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.10, noise_range=(0.02, 0.08)),
        "discrete": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.10, grid_height_range=(0.02, 0.10)),
    },
)
```

---

## E. 现实不确定性考量

### E.1 真实楼梯变异性

| 参数 | 标准范围 | 训练随机化 |
|------|----------|------------|
| 阶高 | 15–20cm | 8–22cm |
| 踏面宽度 | 25–35cm | 20–40cm |
| 表面摩擦 | 瓷砖/木/地毯 | 0.4–1.0 |

### E.2 传感器不确定性

```python
sensor_uncertainty = {
    "gyro_noise": 0.001,      # rad/s/√Hz
    "joint_pos_noise": 0.01,  # rad
    "joint_vel_noise": 0.5,   # rad/s
    "depth_latency": 50,      # ms
}
```

---

## F. 训练超参数

```python
ppo_config = {
    "actor_hidden_dims": [512, 256, 128],
    "critic_hidden_dims": [512, 256, 128],
    "learning_rate": 1e-3,
    "gamma": 0.99,
    "clip_param": 0.2,
    "num_envs": 4096,
    "num_steps_per_env": 24,
}
```

**训练时间估计（RTX 4090）：**
- 平地收敛：~30 分钟
- 简单楼梯：~1 小时
- 混合地形：~2 小时

---

## G. 参考资源

### 论文
1. **Blind Bipedal Stair Traversal (RSS 2021)**: https://arxiv.org/abs/2105.08328
2. **Vision-Based Bipedal Locomotion (2023)**: https://arxiv.org/abs/2309.14594
3. **Humanoid-Gym (RSS 2024)**: https://arxiv.org/abs/2404.05695

### GitHub
1. **legged_gym**: https://github.com/leggedrobotics/legged_gym
2. **humanoid-gym**: https://github.com/roboterax/humanoid-gym
3. **unitree_rl_gym**: https://github.com/unitreerobotics/unitree_rl_gym

---

## H. 用户决策点落地

根据你的选择：
- **决策点 A：混合平地 + 楼梯** → 参考 D 节配置
- **决策点 B：盲爬优先** → 参考 A.1 Cassie 论文，先不加 height_scan
- **决策点 C：增量式改动** → 先只改地形，观察效果后再改奖励
- **决策点 D：程序生成楼梯** → 使用 `MeshPyramidStairsTerrainCfg`

**建议第一步：**
1. 创建 `MIXED_TERRAIN_CFG`（D 节配置）
2. 保持奖励函数不变
3. 用少量 env 可视化观察
4. 若策略"有上楼意图但不稳"，再调整奖励
