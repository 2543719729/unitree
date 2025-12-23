 # 楼梯盲爬任务奖励说明（stair_single_blind_cfg.py）
 
 本文解释 `unitree_rl_lab/tasks/locomotion/robots/g1/29dof/stair_single_blind_cfg.py` 中 `StairBlindRewardsCfg` 的**全部奖励项**：
 
 - **怎么算**（对应到代码的真实计算逻辑）
 - **为什么要有它**
 - **对训练的作用**（它会把策略“推向什么行为”）
 
 ---
 
 ## 1. 奖励在训练里怎么起作用？
 
 每个仿真 step 环境都会算一个总奖励（对每个并行环境各算一个标量）：
 
 `reward_total = Σ (weight_i * term_i)`
 
 其中：
 
 - **`term_i`**：某个奖励函数的输出。
 - **`weight_i`**：对应项的权重。
 - 通常“惩罚项”会返回一个**正数**（表示“越大越糟糕”），然后用**负权重**把它变成扣分。
 
 PPO 等算法会更新策略，让它在长期累计回报上变大。所以：
 
 - **权重大**的项会更强地影响学到的行为。
 - 奖励项组合的本质，就是在“前进/爬升/稳定/省力/不摔”之间做权衡。
 
 ---
 
 ## 2. 奖励项总览（按功能分组）
 
 `StairBlindRewardsCfg` 中的奖励可以理解为 7 类：
 
 - **(A) 命令跟踪**：按给定速度/转向走
 - **(B) 楼梯进展**：明确告诉它“往上/往外爬”
 - **(C) 存活与高度安全**：不摔、不蹲太低
 - **(D) 基座稳定**：少跳、少摇
 - **(E) 动作/关节正则**：动作平滑、少超限、少耗能
 - **(F) 姿态/对称**：避免奇怪扭曲动作
 - **(G) 足部/步态/接触安全**：抬脚过台阶、不打滑、不用身体撞
 
 ---
 
 ## 3. 逐项解释（对应配置里的字段名）
 
 > 提示：文中的 `cmd` 指 `base_velocity` 命令；`root_pos_w` / `root_lin_vel_w` / `root_ang_vel_b` 等量来自 IsaacLab 的资产数据。
 
 ### A. 命令跟踪类
 
 #### 1) `track_lin_vel_xy`（权重 `+3.0`）
 
 - **函数**：`mdp.track_lin_vel_xy_yaw_frame_exp`（来自 `isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py`）
 - **计算**：
   - 将世界系线速度 `root_lin_vel_w` 旋转到“仅按 yaw 对齐的机器人坐标系”（让比较更关注前/侧方向速度，而不是世界坐标）。
   - 误差：`e = || cmd_xy - vel_yaw_xy ||^2`
   - 输出：`term = exp( - e / std^2 )`
   - 本任务 `std = sqrt(0.25) = 0.5`。
 - **直观含义**：你走得越接近要求速度，分越接近 1；差得越多，分迅速接近 0。
 - **对训练的作用**：
   - 强力驱动“真的要向前走”，避免策略学成“站着不动最安全”。
 
 #### 2) `track_ang_vel_z`（权重 `+0.5`）
 
 - **函数**：`mdp.track_ang_vel_z_exp`（来自 `isaaclab/envs/mdp/rewards.py`）
 - **计算**：
   - 误差：`e = (cmd_yaw_rate - base_ang_vel_z)^2`
   - 输出：`term = exp( - e / std^2 )`（本任务 `std = 0.5`）
 - **直观含义**：命令让你转多少就转多少。
 - **对训练的作用**：
   - 抑制“为了找局部最优而绕圈/自转”的行为。
 
 ---
 
 ### B. 楼梯进展类（任务核心信号）
 
 #### 3) `upward_progress`（权重 `+3.5`）
 
 - **函数**：`mdp.upward_progress`（本仓库 `tasks/locomotion/mdp/rewards.py`）
 - **计算**（函数内部会缓存 `上一时刻高度` 与 `episode 初始高度`）：
   - 当前基座高度：`h_t = root_pos_w.z`
   - 单步高度增量：`Δh = h_t - h_{t-1}`
   - 即时上升奖励（只奖励上升）：
     - `r_inst = tanh(delta_scale * max(Δh, 0))`
   - 累计上升奖励（相对初始高度）：
     - `r_prog = tanh(progress_scale * max(h_t - h_0, 0))`
   - 输出：`term = r_inst + r_prog`（大致范围 `[0, 2]`）
   - 本任务参数：`delta_scale=50.0`，`progress_scale=1.0`。
 - **直观含义**：
   - 每一步只要真的“爬高一点”，就有奖励。
   - 整段 episode 如果累计爬得更高，也会持续加分。
 - **对训练的作用**：
   - 这是楼梯任务最直接的“你确实在上楼”的信号。
   - 能补足“只做速度跟踪但不一定能爬上去”的问题。
 
 #### 4) `height_drop_from_peak`（权重 `-0.6`）
 
 - **函数**：`mdp.height_drop_from_peak`（本仓库 `mdp/rewards.py`）
 - **计算**：
   - 维护 episode 内历史最高高度：`h_peak = max(h_peak, h_t)`
   - 下滑量（带死区）：`drop = max(h_peak - h_t - deadzone, 0)`
   - 输出：`term = tanh(scale * drop)`
   - 本任务参数：`deadzone=0.03`，`scale=20.0`。
 - **直观含义**：你爬上去后如果又滑下来/摔下来，会被扣分（而且下滑越多扣得越狠）。
 - **对训练的作用**：
   - 抑制“冲一下然后大幅后滑”的策略，逼策略去学“稳住高度”。
 
 #### 5) `radial_distance_progress`（权重 `+1.0`）
 
 - **函数**：`mdp.radial_distance_progress_delta`（本仓库 `mdp/rewards.py`）
 - **背景**：此任务地形是 `InvertedPyramidStairs`（中心低、往外高），出生点在中心，向外走通常意味着“走向更高的台阶”。
 - **计算**：
   - 当前水平位置：`p_t = root_pos_w.xy`
   - 与出生点的径向距离：`d_t = ||p_t - p_0||`
   - 单步进展：`Δd = d_t - d_{t-1}`
   - 只奖励明显的正向进展（死区）：`Δd' = max(Δd - deadzone, 0)`
   - 输出：`term = tanh(scale * Δd')`
   - 仅在命令足够大时生效：`||cmd|| > cmd_threshold`。
   - 本任务参数：`deadzone=0.001`，`scale=150.0`，`cmd_threshold=0.1`。
 - **直观含义**：只要你这一小步真的“离开出生点更远了”，就加分。
 - **对训练的作用**：
   - 专门打击“原地抖动/小范围绕圈”的局部最优，让策略必须产生真实位移。
   - 与 `track_lin_vel_xy` 互补：一个关心速度像不像命令，一个关心你到底有没有真实推进。
 
 ---
 
 ### C. 存活与高度安全
 
 #### 6) `alive`（权重 `+0.15`）
 
 - **函数**：`mdp.is_alive`（来自 `isaaclab/envs/mdp/rewards.py`）
 - **计算**：未终止时 `term=1`，终止时 `term=0`。
 - **直观含义**：活着就给一点点保底分。
 - **对训练的作用**：
   - 训练初期很重要：否则机器人频繁摔倒、episode 很短，学习信号太稀。
   - 但权重被刻意压低（0.15），防止学成“原地站着混分”。
 
 #### 7) `base_height_relative`（权重 `-0.5`）
 
 - **函数**：`mdp.base_height_relative`（本仓库 `mdp/rewards.py`）
 - **计算**：
   - episode 第一步记录参考高度 `h_ref`。
   - 目标高度：`h_target = h_ref + target_offset`（这里 `target_offset=0.0`）。
   - `only_penalize_drop=True` 时：
     - `drop = max(h_target - h_t, 0)`
     - 输出 `term = drop^2`
 - **直观含义**：你只要蹲得太低/趴下/摔倒（高度掉下去），就扣分；爬楼导致高度更高不会扣。
 - **对训练的作用**：
   - 很适合楼梯任务：它不像“固定世界高度”那样会误伤正常爬升。
 
 ---
 
 ### D. 基座稳定
 
 #### 8) `base_linear_velocity`（权重 `-0.5`）
 
 - **函数**：`mdp.lin_vel_z_l2`（来自 `isaaclab/envs/mdp/rewards.py`）
 - **计算**：`term = (v_z)^2`（基座 z 方向速度平方，通常在机体坐标系 `root_lin_vel_b` 的 z 分量）
 - **直观含义**：上下“弹跳/蹦”越厉害越扣分。
 - **对训练的作用**：抑制跳跃式、抖动式的上楼方式，让动作更稳定。
 
 #### 9) `base_angular_velocity`（权重 `-0.1`）
 
 - **函数**：`mdp.ang_vel_xy_l2`（来自 `isaaclab/envs/mdp/rewards.py`）
 - **计算**：`term = ω_x^2 + ω_y^2`（roll/pitch 角速度平方和）
 - **直观含义**：左右摇、前后点头越厉害越扣分。
 - **对训练的作用**：减少楼梯上“晃倒”的风险。
 
 ---
 
 ### E. 动作/关节正则
 
 #### 10) `joint_vel`（权重 `-0.0005`）
 
 - **函数**：`mdp.joint_vel_l2`（来自 `isaaclab/envs/mdp/rewards.py`）
 - **计算**：`term = Σ q̇_i^2`
 - **直观含义**：关节甩得越猛扣分越多。
 - **对训练的作用**：抑制高频抖动，让动作更“稳”。
 
 #### 11) `joint_acc`（权重 `-2.5e-7`）
 
 - **函数**：`mdp.joint_acc_l2`（来自 `isaaclab/envs/mdp/rewards.py`）
 - **计算**：`term = Σ q̈_i^2`
 - **直观含义**：关节加速度越大（变化越突兀）扣分越多。
 - **对训练的作用**：进一步平滑动作。
 
 #### 12) `action_rate`（权重 `-0.01`）
 
 - **函数**：`mdp.action_rate_l2`（来自 `isaaclab/envs/mdp/rewards.py`）
 - **计算**：`term = Σ (a_t - a_{t-1})^2`
 - **直观含义**：连续两步的控制指令变化越快越扣分。
 - **对训练的作用**：减少输出抖动，提升可控性。
 
 #### 13) `dof_pos_limits`（权重 `-0.7`）
 
 - **函数**：`mdp.joint_pos_limits`（来自 `isaaclab/envs/mdp/rewards.py`）
 - **计算**：关节位置超出软限位就按超出量累计：`term = Σ out_of_limits_i`
 - **直观含义**：关节别打到极限角度。
 - **对训练的作用**：避免学出“顶到限位硬撑”的动作（仿真里可能能做，但真实机器人风险大）。
 
 #### 14) `energy`（权重 `-2e-5`）
 
 - **函数**：`mdp.energy`（本仓库 `mdp/rewards.py`）
 - **计算**：`term = Σ |τ_i| * |q̇_i|`
 - **直观含义**：近似能耗/功率（用力矩和角速度衡量）。
 - **对训练的作用**：鼓励更省力、少蛮力的上楼方式。
 
 ---
 
 ### F. 姿态/默认姿态偏差与对称
 
 #### 15) `joint_deviation_arms`（权重 `-0.1`）
 
 - **函数**：`mdp.joint_deviation_l1`（来自 `isaaclab/envs/mdp/rewards.py`）
 - **计算**：对选中的手臂关节：`term = Σ |q - q_default|`
 - **直观含义**：上肢别乱甩。
 - **对训练的作用**：减少用手臂乱挥来找平衡的投机解。
 
 #### 16) `joint_deviation_waists`（权重 `-0.25`）
 
 - **函数**：同上（只针对腰部关节）。
 - **直观含义**：腰部别偏离默认太多。
 - **对训练的作用**：避免“用腰扭成麻花”来换取上楼。
 
 #### 17) `joint_deviation_legs`（权重 `-0.25`）
 
 - **函数**：同上（主要约束 `hip_roll` / `hip_yaw`）。
 - **直观含义**：腿部别过度外八/内八、过度扭转。
 - **对训练的作用**：减少怪异姿态，但仍留足够自由度跨阶。
 
 #### 18) `flat_orientation_l2`（权重 `-1.2`）
 
 - **函数**：`mdp.flat_orientation_l2`（来自 `isaaclab/envs/mdp/rewards.py`）
 - **计算**：使用投影重力向量 x/y 分量：`term = g_x^2 + g_y^2`
 - **直观含义**：身体越倾斜扣分越多。
 - **对训练的作用**：让躯干保持直立，是防摔的关键约束。
 
 #### 19) `joint_deviation_knees`（权重 `-0.3`）
 
 - **函数**：`mdp.joint_deviation_l1`（针对膝关节）
 - **直观含义**：别长期大幅弯膝（避免“蹲着走”）。
 - **对训练的作用**：抑制半蹲挪动这种常见但能效差、容易不稳的策略。
 
 #### 20) `leg_joint_symmetry`（权重 `-0.1`）
 
 - **函数**：`mdp.joint_mirror`（本仓库 `mdp/rewards.py`）
 - **计算**：对指定左右对称关节对：
   - `term = mean_over_pairs( (q_left - q_right)^2 )`
 - **直观含义**：左右腿尽量对称，不要一边“怪抬/怪扭”。
 - **对训练的作用**：减少极端不对称步态，提升稳定性与可迁移性。
 
 ---
 
 ### G. 足部/步态/接触安全
 
 #### 21) `gait`（权重 `-0.05`）
 
 - **函数**：`mdp.air_time_variance_penalty_with_cmd`（本仓库 `mdp/rewards.py`）
 - **计算**：
   - 取每只脚的 `last_air_time` 和 `last_contact_time`，分别做方差（并 clip 到 0.5）：
     - `pen = var(clip(last_air_time, 0.5)) + var(clip(last_contact_time, 0.5))`
   - 仅在 `||cmd|| > cmd_threshold` 时生效。
 - **直观含义**：两条腿的抬脚/支撑节奏别差太大。
 - **对训练的作用**：推动更均衡的步态，减少“一条腿拖地、一条腿乱蹬”。
 
 #### 22) `feet_air_time`（权重 `+0.6`）
 
 - **函数**：`mdp.feet_air_time`（来自 `isaaclab_tasks/.../velocity/mdp/rewards.py`）
 - **计算**：
   - 在脚“首次接触地面”的瞬间（`first_contact`）读取上一次腾空时间 `last_air_time`。
   - 输出：`term = Σ (last_air_time - threshold) * first_contact`
   - 若命令很小则不奖励：`||cmd_xy|| > 0.1` 才给。
   - 本任务 `threshold=0.3`。
 - **直观含义**：鼓励迈更“明确的一步”（脚要离地足够久）。
 - **对训练的作用**：对楼梯非常关键：跨台阶需要腾空摆腿。
 
 #### 23) `feet_slide`（权重 `-0.25`）
 
 - **函数**：`mdp.feet_slide_stairs`（本仓库 `mdp/rewards.py`）
 - **计算**（只在“踩实且接触稳定”时才算滑）：
   - 接触判定：接触力 `> force_threshold`
   - 稳定接触：`current_contact_time > stable_contact_time`
   - 足端水平滑动速度：`slip_speed = ||v_xy||`
   - 去掉小滑动（死区）：`slip = max(slip_speed - slip_velocity_deadzone, 0)`
   - 平滑：`slip = tanh(tanh_mult * slip)`
   - 输出：`term = Σ slip * (is_contact & stable_contact)`
   - 本任务参数：`force_threshold=1.0`，`stable_contact_time=0.03`，`slip_velocity_deadzone=0.05`，`tanh_mult=3.0`。
 - **直观含义**：脚踩住以后还在明显“擦着滑”，就扣分。
 - **对训练的作用**：抑制打滑；配合 `height_drop_from_peak` 能显著减少“上去又滑下来”。
 
 #### 24) `feet_clearance`（权重 `+0.6`）
 
 - **函数**：`mdp.foot_clearance_reward_swing`（本仓库 `mdp/rewards.py`）
 - **计算**（只对摆动脚算；并且只在有走路命令时算）：
   - 摆动脚：`is_swing = ~is_contact`
   - 高度得分（越接近目标越高）：
     - `height_score = exp( -0.5 * ((foot_z - target_height)/std)^2 )`
   - 速度得分（脚真的在摆动才给）：
     - `vel_score = tanh(tanh_mult * max(foot_vel_xy - 0.05, 0))`
   - 每只脚：`score = height_score * vel_score * is_swing`
   - 输出：对脚取平均。
   - 本任务参数：`target_height=0.12`，`std=0.05`，`tanh_mult=2.0`。
 - **直观含义**：迈步时抬脚要“抬到合适高度并且有摆动速度”。
 - **对训练的作用**：
   - 直接推动“抬脚跨台阶”，减少踢台阶（踢到就容易绊倒或触发不良接触）。
 
 #### 25) `undesired_contacts`（权重 `-0.5`）
 
 - **函数**：`mdp.undesired_contacts`（来自 `isaaclab/envs/mdp/rewards.py`）
 - **计算**：
   - 对指定的 body 集合，只要接触力超过阈值就记一次违规。
   - 输出：`term = Σ 1[ contact_force > threshold ]`
   - 本任务阈值 `threshold=1`，并且配置里排除了 ankle（脚），也就是说：**除了脚以外的身体部位碰撞都会扣分**。
 - **直观含义**：不要用膝盖、手、躯干去撞楼梯/地面。
 - **对训练的作用**：强行阻止“贴着楼梯爬/用身体顶上去”的投机策略。
 
 ---
 
 ## 4. 读懂训练过程：几个常见现象与对应奖励
 
 - **现象：机器人很喜欢原地站着或小碎步抖动**
   - **通常原因**：稳定/存活类约束压过了“前进/进展类”奖励。
   - **本配置为什么能缓解**：`track_lin_vel_xy` 与 `upward_progress` 权重都比较大，而 `alive` 权重很小。
 
 - **现象：机器人绕圈、走偏、不怎么爬高**
   - **通常原因**：只做速度跟踪可能出现“局部投机”。
   - **本配置如何处理**：用 `radial_distance_progress` 强制产生真实位移进展，并限制命令角速度范围。
 
 - **现象：爬上去后又明显滑落**
   - **相关项**：`height_drop_from_peak`（下滑扣分）、`feet_slide`（打滑扣分）。
 
 - **现象：经常踢台阶、绊倒**
   - **相关项**：`feet_clearance`（摆动脚抬高到目标高度）、`feet_air_time`（鼓励真正迈步腾空）。
