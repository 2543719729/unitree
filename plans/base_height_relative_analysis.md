# `base_height_relative` 函数问题分析

## 📍 函数位置
[`rewards.py:391-447`](unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py:391)

## 🔍 问题总结

### ❌ 问题 1：重置检测时机错误（最严重）

**原代码：**
```python
if hasattr(env, "reset_buf"):
    reset_mask = env.reset_buf > 0
    if torch.any(reset_mask):
        env._episode_start_height = torch.where(
            reset_mask, current_height, env._episode_start_height
        )
```

**问题分析：**

根据 Isaac Lab 的 [`step()`](IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:154) 执行顺序：

```
① episode_length_buf += 1        # 步数 +1
② reset_buf = termination_manager.compute()  # 计算需要重置的环境
③ reward_buf = reward_manager.compute()      # ← 奖励函数在这里被调用
④ _reset_idx(reset_env_ids)      # 执行重置
⑤ episode_length_buf[env_ids] = 0  # 重置步数
```

**时序问题：**
- 当 `reset_buf > 0` 时，环境**尚未执行重置**
- 此时 `current_height` 是**重置前**的高度（可能是摔倒后的低位置）
- 初始高度被错误设置为摔倒时的低高度

**后果：** 下一个 episode 的目标高度会非常低，导致机器人学会蹲着走。

---

### ❌ 问题 2：`episode_length_buf == 0` 检测无效

如果尝试用 `episode_length_buf == 0` 检测新 episode：

```python
new_episode_mask = env.episode_length_buf == 0  # 这样写是错的！
```

**问题：** 在奖励函数被调用时，`episode_length_buf` 已经 +1 了，所以永远不会等于 0。

**正确做法：** 使用 `episode_length_buf == 1` 检测新 episode 的第一步。

---

### ❌ 问题 3：楼梯任务的逻辑矛盾

**设计意图：** 在楼梯上爬升时，高度会增加，所以用相对高度避免错误惩罚。

**实际问题：** 如果机器人成功爬了几级楼梯，高度增加了，这个函数会**惩罚**这种增加！

```
初始高度 = 0.78m
爬了3级楼梯后高度 = 1.2m
height_error = (1.2 - 0.78)² = 0.176  ← 错误惩罚！
```

**正确做法：** 只惩罚高度**下降**（蹲下/摔倒），不惩罚高度**增加**（爬楼梯）。

---

### ❌ 问题 4：首次初始化时机不稳定

```python
if not hasattr(env, "_episode_start_height"):
    env._episode_start_height = current_height.clone()
```

**问题：** 首次调用时，机器人可能还在下落过程中，高度不稳定。

---

### ⚠️ 问题 5：在 env 上存储状态（设计问题）

```python
env._episode_start_height = ...
```

**问题：**
1. 直接修改 `env` 对象的属性，可能与其他代码冲突
2. 不符合 Isaac Lab 的设计模式

---

## ✅ 修复方案

### 方案 A：修复当前函数（推荐）

```python
def base_height_relative(
    env: ManagerBasedRLEnv,
    target_offset: float = 0.0,
    only_penalize_drop: bool = True,  # 新参数：只惩罚下降
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    相对高度惩罚：相对于 episode 开始时的高度
    
    修复版本：
    - 使用 episode_length_buf == 1 检测新 episode
    - 可选只惩罚高度下降（适合楼梯任务）
    
    Args:
        env: 环境实例
        target_offset: 相对于初始高度的目标偏移
        only_penalize_drop: 是否只惩罚高度下降（True=只惩罚蹲下/摔倒）
        asset_cfg: 机器人资产配置
    
    Returns:
        惩罚张量，形状为 (num_envs,)
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    current_height = asset.data.root_pos_w[:, 2]
    
    # 初始化参考高度
    if not hasattr(env, "_base_height_reference"):
        env._base_height_reference = current_height.clone()
    
    # 形状不一致时进行重置
    if env._base_height_reference.shape != current_height.shape:
        env._base_height_reference = current_height.clone()
    
    # 使用 episode_length_buf == 1 检测新 episode 的第一步
    # 注意：在奖励函数被调用时，episode_length_buf 已经 +1 了
    new_episode_mask = env.episode_length_buf == 1
    if torch.any(new_episode_mask):
        env._base_height_reference = torch.where(
            new_episode_mask, current_height, env._base_height_reference
        )
    
    # 计算目标高度
    target_height = env._base_height_reference + target_offset
    
    if only_penalize_drop:
        # 只惩罚低于目标高度的情况（蹲下/摔倒）
        # 不惩罚高于目标高度的情况（爬楼梯）
        height_drop = torch.clamp(target_height - current_height, min=0.0)
        return torch.square(height_drop)
    else:
        # 惩罚任何偏离目标高度的情况
        height_error = current_height - target_height
        return torch.square(height_error)
```

### 方案 B：使用滑动参考高度（更鲁棒）

```python
def base_height_sliding_reference(
    env: ManagerBasedRLEnv,
    target_offset: float = 0.0,
    update_rate: float = 0.01,  # 参考高度更新速率
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    滑动参考高度惩罚：参考高度会缓慢跟随当前高度
    
    适合楼梯任务：
    - 当机器人爬升时，参考高度会缓慢上升
    - 当机器人突然下降时（摔倒），会产生惩罚
    
    Args:
        env: 环境实例
        target_offset: 相对于参考高度的目标偏移
        update_rate: 参考高度更新速率（0-1，越大跟随越快）
        asset_cfg: 机器人资产配置
    
    Returns:
        惩罚张量，形状为 (num_envs,)
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    current_height = asset.data.root_pos_w[:, 2]
    
    # 初始化参考高度
    if not hasattr(env, "_sliding_height_reference"):
        env._sliding_height_reference = current_height.clone()
    
    # 形状不一致时进行重置
    if env._sliding_height_reference.shape != current_height.shape:
        env._sliding_height_reference = current_height.clone()
    
    # 新 episode 时重置参考高度
    new_episode_mask = env.episode_length_buf == 1
    if torch.any(new_episode_mask):
        env._sliding_height_reference = torch.where(
            new_episode_mask, current_height, env._sliding_height_reference
        )
    
    # 参考高度只向上更新（允许爬升），不向下更新（惩罚下降）
    # 这样爬楼梯时参考高度会跟着上升，但摔倒时不会跟着下降
    higher_mask = current_height > env._sliding_height_reference
    env._sliding_height_reference = torch.where(
        higher_mask,
        env._sliding_height_reference + update_rate * (current_height - env._sliding_height_reference),
        env._sliding_height_reference
    )
    
    # 只惩罚低于参考高度的情况
    target_height = env._sliding_height_reference + target_offset
    height_drop = torch.clamp(target_height - current_height, min=0.0)
    
    return torch.square(height_drop)
```

---

## 📊 执行时序图

```
Step N (正常步骤):
┌─────────────────────────────────────────────────────────────┐
│ episode_length_buf: 5 → 6                                   │
│ reset_buf: [0, 0, 0, ...]  (没有环境需要重置)               │
│ reward_manager.compute() → base_height_relative()           │
│   - episode_length_buf == 1? No                             │
│   - 使用已有的 _base_height_reference                       │
└─────────────────────────────────────────────────────────────┘

Step N+1 (环境 0 摔倒):
┌─────────────────────────────────────────────────────────────┐
│ episode_length_buf: [6, 7, 8, ...] → [7, 8, 9, ...]         │
│ reset_buf: [1, 0, 0, ...]  (环境 0 需要重置)                │
│ reward_manager.compute() → base_height_relative()           │
│   - 环境 0 的 current_height 是摔倒后的低位置               │
│   - 但我们不在这里更新参考高度！                            │
│ _reset_idx([0])                                             │
│   - 环境 0 被重置                                           │
│   - episode_length_buf[0] = 0                               │
└─────────────────────────────────────────────────────────────┘

Step N+2 (环境 0 新 episode 第一步):
┌─────────────────────────────────────────────────────────────┐
│ episode_length_buf: [0, 8, 9, ...] → [1, 9, 10, ...]        │
│ reset_buf: [0, 0, 0, ...]                                   │
│ reward_manager.compute() → base_height_relative()           │
│   - episode_length_buf[0] == 1? Yes!                        │
│   - 更新环境 0 的 _base_height_reference 为当前高度         │
│   - 此时 current_height 是重置后的正确初始高度              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 建议

1. **立即修复**：使用方案 A 修复当前函数
2. **楼梯任务**：设置 `only_penalize_drop=True`
3. **平地任务**：设置 `only_penalize_drop=False`
4. **测试验证**：添加日志打印 `_base_height_reference` 的值，确认更新时机正确
