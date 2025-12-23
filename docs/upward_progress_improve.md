 # Blind Stair Climbing: `upward_progress` 深度缺陷与迭代式改进方案
 
 ## 0. 背景与目标
 Blind Stair Climbing（盲爬楼梯）任务中，策略不使用 `height_scanner`，因此奖励必须同时满足：
 
 - **可探索**：在“还不会上第一阶”时也能提供可学习的中间信号。
 - **防投机**：避免原地抖动、姿态拔高、侧移/绕圈等刷分行为。
 - **对齐地形**：在 InvertedPyramidStairs（中心低、向外高）中，不能把“z 增加”直接等价为“成功攀爬”。
 - **高信噪比**：抵抗步态导致的自然垂直振荡（vertical oscillation）。
 
 本文对当前 `mdp.upward_progress` 在盲爬场景下的局限性做可行性分析，并给出可落地的修改方向（公式、参数建议、以及 `stair_single_blind_cfg.py` 的具体改法）。后半部分包含“反复迭代式”漏洞分析：每次提出修改后再找潜在投机，再提出缓解。
 
 ## 1. 现状：当前 `upward_progress` 的定义与使用
 代码位置：`unitree_rl_lab/tasks/locomotion/mdp/rewards.py::upward_progress`
 
 - `instant_reward = tanh(delta_scale * clamp(height_delta, min=0))`
 - `progress_reward = tanh(progress_scale * clamp(total_progress, min=0))`
 - `reward = instant_reward + progress_reward`
 
 配置位置：`tasks/locomotion/robots/g1/29dof/stair_single_blind_cfg.py`
 
 - `upward_progress.weight = 3.5`
 - `delta_scale = 50.0`, `progress_scale = 1.0`
 - 已启用：`height_drop_from_peak`、`radial_distance_progress_delta`
 
 ## 2. 可行性结论：为何该实现对盲爬/倒金字塔具有结构性局限
 
 ### 2.1 状态奖励陷阱（驻留刷分 / Local Optima）
 `progress_reward` 是典型的**状态奖励（state reward）**：只要处于更高位置，每一步都持续给分。
 
 - **后果**：上到某一层后，“保持不摔倒并驻留”可能比“继续攀爬冒险”更有期望回报。
 - **盲爬放大效应**：没有地形观测时，探索更容易摔倒，风险更高，驻留吸引子更强。
 
 ### 2.2 伪爬升（knee extension / posture cheating）
 `root_pos_w.z` 的上升并不等价于“脚踩上更高台阶”。它可以通过：
 
 - 脚没有真正跨阶，只是**伸直腿/抬高骨盆**来提高基座高度。
 - 甚至在同一踏面上通过姿态调节产生 z 增量。
 
 **盲爬中**这条投机路径更易被策略发现：因为“准确跨阶”对探索更难，而“伸直腿抬高身高”对控制更直接。
 
 ### 2.3 倒金字塔特有缺陷：空间-高度耦合错误
 倒金字塔地形中：向外“任何方向”的平移（x/y）都可能导致 z 增加。
 
 - **后果**：侧向漂移、失衡斜走、甚至摔倒时的滑移，都可能被 `upward_progress` 强化。
 
 ### 2.4 信噪比问题：步态振荡 ≫ 有效跨阶信号
 `delta_scale=50` 会把毫米级抖动放大，导致：
 
 - “上下颠簸”即可获得稳定的 `instant_reward`。
 - 有效跨阶是事件型（touchdown/step-up），但奖励是连续型，容易被周期振荡污染。
 
 ### 2.5 探索不足：缺少“盲摸”过程奖励
 盲爬关键技能：足端抬高（clearance）、触地恢复、碰撞后调整。
 
 仅用基座高度信号，属于“结果奖励”，对过程动作指导不足，探索期会被显著拉长。
 
 ## 3. 修改方向总览（推荐路线）
 建议把“向上进展”的主信号从 `root_z` 转到更贴近跨阶物理机制的指标，并把 `upward_progress` 改造成：
 
 - **只奖励净进展的增量（delta）**，避免驻留回报。
 - **事件驱动（touchdown/step-up）**，避免振荡噪声。
 - **方向/径向耦合门控**，适配倒金字塔地形。
 - **命令门控**：无前进命令时不奖励进展，避免原地蹦跳刷分。
 
 接下来按 Iteration 1~3 分阶段给出可落地改法。
 
 ## 4. Iteration 1（最小侵入）：去状态奖 + 提信噪比
 
 ### 4.1 核心改动
 1) **移除 `progress_reward` 的“驻留”形态**
 
 将：
 - `r_state(t) = tanh(progress_scale * total_progress_t)`
 
 改为 potential-difference：
 - `r_state_delta(t) = clamp( r_state(t) - r_state(t-1), min=0 )`
 
 这样“站着不动”没有持续收益，只在净高度提升时给脉冲奖励。
 
 2) **对 `instant_reward` 加 deadzone 或滤波**
 
 将：
 - `height_delta = h_t - h_{t-1}`
 
 改成：
 - `height_delta = h_t - h_{t-1}`
 - `height_delta = clamp(height_delta - dz_h, min=0)`，建议 `dz_h = 0.005 ~ 0.01 m`
 - 或对 `h_t` 做 EMA 滤波（β=0.9 左右）后再求增量。
 
 3) **命令门控**
 - 若 `||cmd|| < cmd_threshold`，则 `upward_progress = 0`。
 
 ### 4.2 配置建议（`stair_single_blind_cfg.py`）
 - 将 `upward_progress.weight` 从 `3.5` 降到 `1.5 ~ 2.0`
 - 将 `delta_scale` 从 `50` 降到 `10 ~ 20`（与 deadzone/滤波配合）
 - `height_drop_from_peak` 保留，必要时增强（例如 `-0.6 → -0.8`）
 
 ### 4.3 Iteration 1 再次深度分析：新漏洞与缓解
 - **潜在漏洞 1：净上升振荡**
   - 策略可能通过“非对称的上下动作”制造滤波后的净上升。
   - **缓解**：只在触地事件采样高度（见 Iteration 2），或增大 deadzone。
 - **潜在漏洞 2：侧向漂移**
   - 即使去掉驻留，侧向漂移仍可能带来净高度增量。
   - **缓解**：引入方向耦合门控（见 Iteration 3）。
 
 ## 5. Iteration 2（强对齐）：用“触地 step-up 事件”替代基座 z
 
 ### 5.1 核心思想
 对楼梯而言，真正的进展事件是：**摆腿 → 触地（touchdown）→ 脚落在更高踏面**。
 
 因此应该奖励：
 - “脚触地高度的提升”
 而不是：
 - “身体中心高度的提升”。
 
 这能直接抑制：
 - knee extension 伪爬升
 - vertical oscillation 刷分
 - 高处驻留刷分（没有触地事件就没有奖励）
 
 ### 5.2 可落地的新奖励定义（建议新增函数）
 记两只脚的足端 link（如 ankle/foot link）的世界高度为 `z_foot`。
 
 触地判定（示意）：
 - `is_contact = current_contact_time > stable_contact_time`
 - `touchdown = is_contact & (~is_contact_prev)`
 
 为每只脚维护 touchdown 高度缓存：
 - `z_td = z_foot`（在 touchdown 时更新）
 - `Δz_td = z_td - z_td_prev`
 
 step-up 奖励：
 - `r_step_up = tanh(k_td * clamp(Δz_td - dz_td, min=0))`
 
 建议参数：
 - `dz_td = 0.01 m`
 - `k_td = 30 ~ 80`
 - 返回每步 `mean(r_step_up_over_feet)`（仅 touchdown 时非零）
 
 ### 5.3 Iteration 2 再次深度分析：新漏洞与缓解
 - **潜在漏洞 1：侧向 step-up**
   - 倒金字塔里向侧面也能踩到更高处。
   - **缓解**：必须加方向/命令耦合（Iteration 3）。
 - **潜在漏洞 2：跳跃式 step-up**
   - 策略可能通过跳跃落地来获得更大 Δz_td。
   - **缓解**：
     - 对过长腾空时间增加惩罚（或对触地前飞行时长设上限门控）。
     - 与 `feet_air_time`、`lin_vel_z_l2`、`action_rate` 等正则协同调参。
 
 ## 6. Iteration 3（倒金字塔特化）：加入空间-高度耦合门控
 
 ### 6.1 方案 A：与命令方向耦合（优先推荐）
 令：
 - `cmd_xy = command(base_velocity)[:2]`
 - `dir = normalize(cmd_xy)`
 - `Δp = p_xy_t - p_xy_{t-1}`
 - `Δs = dot(Δp, dir)`
 
 将 step-up 奖励乘以门控：
 - `gate = 1(Δs > dz_s)` 或 `gate = tanh(k_s * clamp(Δs - dz_s, min=0))`
 - `r = r_step_up * gate`
 
 参数建议：
 - `dz_s = 0.002 ~ 0.01 m/step`
 - `k_s = 50 ~ 150`
 
 ### 6.2 方案 B：与径向增量耦合（备选）
 直接用已有 `radial_distance_progress_delta` 作为门控/乘子：
 - `r = r_step_up * tanh(k_r * clamp(Δr - dz_r, min=0))`
 
 该方案对“向外爬”有效，但对“正确方向”约束不如方案 A。
 
 ### 6.3 Iteration 3 再次深度分析：新漏洞与缓解
 - **潜在漏洞：命令噪声 / 小命令抖动**
   - 小命令时方向不稳定导致门控异常。
   - **缓解**：强制 `cmd_norm > cmd_threshold`；否则 gate=0。
 
 ## 7. `stair_single_blind_cfg.py` 的建议改动清单（可执行）
 
 ### 7.1 权重重分配（起始建议）
 - `upward_progress.weight: 3.5 → 1.5 ~ 2.0`
 - `radial_distance_progress_delta.weight: 1.0 → 1.0 ~ 1.2`
 - `height_drop_from_peak.weight: -0.6 → -0.6 ~ -0.9`（视滑落频率）
 - `feet_clearance.weight: 0.6 → 0.7 ~ 0.9`（强化“盲摸”）
 - （若启用）新增 `step_up_touchdown_reward.weight: 0.8 ~ 2.0`（逐步调）
 
 ### 7.2 “踩高跷”约束的谨慎策略
 直接把 `joint_deviation_knees` 变得很强，可能会误伤“必要屈膝抬腿”。
 
 更可行的做法是：
 - 先用 step-up 事件奖励对齐目标。
 - 再添加“只惩罚过度伸直”的非对称惩罚（例如只惩罚朝伸直方向的偏差）。
 
 ## 8. 训练验收：你应该观察到什么
 
 - `upward_progress`：不再在某一台阶持续维持高值（驻留回报消失）。
 - `height_drop_from_peak`：跌落/滑落时出现尖峰，成功策略保持低。
 - `radial_distance_progress_delta`：与实际位移相关，原地抖动接近 0。
 - （若实现 step-up）`step_up_touchdown_reward`：在跨阶触地瞬间出现脉冲，且对“踩高跷”不敏感。
 
 ## 9. 下一步建议（按优先级）
 - **第一优先级**：落地 Iteration 1（最小侵入），先消除驻留与噪声放大。
 - **第二优先级**：实现 Iteration 2 的 step-up touchdown 事件奖励（主信号对齐）。
 - **第三优先级**：实现 Iteration 3 的方向耦合门控（倒金字塔特化防投机）。
 
 后续每次训练迭代建议按：
 - 发现投机行为 → 明确“被奖励的错误机制” → 加门控/换信号 → 再次分析新漏洞
 的循环进行。
