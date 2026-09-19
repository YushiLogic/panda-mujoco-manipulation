---
exp_id: PM-B-260919-001
date: 2026-09-19
project: panda-mujoco-manipulation
day: 20
exp_type: 随机抓取任务接口与批量策略评估
platform: Windows
python: 3.10.20
mujoco: 3.12.0
observation_size: 29
actions: [WAIT, RUN_SCRIPTED_PICK]
default_evaluation_cases: 5
evaluation_success: "5/5"
task_test_result: "19 passed"
full_test_result: "160 passed in 8.25s"
anomaly: true
anomaly_status: resolved
anomaly_summary: task.reset与Day19状态机原有RESET会重复初始化同一episode
result: passed
tags: [MuJoCo, Panda, task-interface, observation, action, seed, evaluation]
---

# Day 20 日志：随机抓取任务接口与批量策略评估

## 1. 今日目标

Day19 已经完成了一次完整脚本抓取：

```text
RESET
  ↓
MOVE_ABOVE
  ↓
APPROACH
  ↓
CLOSE_GRIPPER
  ↓
VERIFY_CONTACT
  ↓
LIFT
  ↓
CHECK_SUCCESS
  ↓
DONE / FAILED
```

但是 Day19 的入口是一个专用函数：

```python
run_scripted_pick(scene, seed=42)
```

它适合直接运行固定策略，却还不是一个通用的“任务环境”。如果后续希望：

- 连续运行多个随机 episode；
- 换成新的控制策略；
- 接入模仿学习或强化学习；
- 用统一格式收集数据；
- 统计不同失败阶段；

就需要明确任务与策略之间的接口边界。

Day20 将前面已经实现的模块包装为：

```python
observation, info = task.reset(seed=42)

observation, reward, terminated, truncated, info = task.step(action)
```

今天不是重新实现抓取控制，而是为已有能力建立稳定、可重复、可评估的
上层接口。

---

## 2. 模块关系

Day20 没有替代前面的模块，而是把它们组合起来：

```text
PandaPickTask
│
├── reset(seed)
│   ├── mj_resetData
│   ├── reset_to_home
│   ├── sample_cube_pose
│   ├── reset_cube
│   └── settle_cube
│
├── get_pick_observation
│   ├── arm qpos/qvel
│   ├── gripper width
│   ├── end-effector position
│   ├── cube position/yaw
│   └── cube linear/angular velocity
│
└── step(action)
    ├── WAIT
    │   └── bias compensation + mj_step
    │
    └── RUN_SCRIPTED_PICK
        └── Day19 run_scripted_pick
```

对应职责是：

```text
Day16：随机生成和复位方块
Day18：监测接触、抬升和稳定性
Day19：决定抓取流程如何推进
Day20：定义一个episode如何开始、交互和结束
```

---

## 3. 什么是 episode

一个 episode 可以理解为一次完整任务尝试。

本项目中的一次 episode 是：

```text
生成一个随机方块场景
        ↓
机器人回到home，方块自然落稳
        ↓
策略执行动作
        ↓
任务得到成功、失败或超时结果
        ↓
本episode结束，必须再次reset才能开始下一次
```

任务对象保存以下 episode 状态：

```python
self.seed
self.episode_steps
self.terminated
self.truncated
self.has_reset
```

这些变量不是 MuJoCo 的物理状态，而是任务层对当前回合的管理状态。

---

## 4. `reset(seed)` 做了什么

### 4.1 完整复位运行时状态

首先调用：

```python
mujoco.mj_resetData(scene.model, scene.data)
scene.reset_to_home()
```

`mj_resetData()` 清除上一 episode 留下的：

- 仿真时间；
- 关节状态；
- 加速度；
- 外加力；
- 接触求解相关运行时数据。

`reset_to_home()` 再写入本项目规定的安全机器人 home 位姿和对应控制目标。

只设置 `qpos` 不足以完成 episode 复位，因为上一回合的速度、外力和时间
也可能继续影响下一回合。

### 4.2 根据 seed 生成方块初始位姿

方块随机化代码本质上是：

```python
rng = np.random.default_rng(seed)

x = rng.uniform(0.42, 0.48)
y = rng.uniform(-0.06, 0.06)
yaw = rng.uniform(-30_deg, 30_deg)
```

关系可以写成：

```text
seed → 伪随机数序列 → x、y、yaw
```

这里的“随机”是可复现的伪随机：

- 相同 seed 会生成相同的 `x、y、yaw`；
- 不同 seed 通常会生成不同的位姿；
- seed 不是方块坐标，只是随机数生成器的初始条件；
- `seed=None` 时每次结果通常不同，不适合严格复现实验。

初始 z 固定为 `0.05 m`，不是随机量。方块随后由重力自然落到约：

```text
z = 0.0199276 m
```

### 4.3 `qpos` 与 `xpos`

方块使用 free joint，因此它占用七个 `qpos`：

```text
[x, y, z, qw, qx, qy, qz]
```

`cube_joint` 本身不使用 `xpos` 这个字段。代码中的：

```python
scene.data.body("cube").xpos
```

是 cube body 的世界坐标位置，由 MuJoCo 根据 joint `qpos` 和模型层级计算。

整体关系是：

```text
seed
  ↓
生成初始x、y、yaw
  ↓
写入cube_joint qpos
  ↓
MuJoCo正向计算cube body xpos
  ↓
推进动力学直到方块落稳
  ↓
得到episode实际初始xpos
```

### 4.4 reset 的返回值

`reset()` 返回：

```python
observation, info
```

其中 observation 只保存策略可能使用的数值状态，info 保存诊断信息：

```python
{
    "seed": 42,
    "episode_steps": 0,
    "task_state": "READY",
    "cube_position": ...,
    "cube_yaw": ...,
    "settle_duration": 1.5,
}
```

---

## 5. 29维 observation

观察向量形状固定为：

```python
observation.shape == (29,)
```

布局如下：

| 下标 | 维数 | 内容 | 单位 |
|---|---:|---|---|
| `0:7` | 7 | Panda 手臂关节位置 | rad |
| `7:14` | 7 | Panda 手臂关节速度 | rad/s |
| `14` | 1 | 夹爪实际总开口 | m |
| `15:18` | 3 | 末端世界位置 | m |
| `18:21` | 3 | 方块世界位置 | m |
| `21:23` | 2 | 方块 yaw 的 sin/cos | 无量纲 |
| `23:26` | 3 | 方块线速度 | m/s |
| `26:29` | 3 | 方块角速度 | rad/s |

### 5.1 为什么 observation 只放数值

策略、神经网络和数据集通常需要固定形状的数值张量。字符串状态、失败原因
等诊断内容不适合直接作为模型输入，因此放入 info，而不是 observation。

### 5.2 为什么 yaw 使用 sin/cos

角度在 `-π` 和 `+π` 处会发生数值跳变。例如：

```text
+179° 和 -179°
```

真实朝向只相差 2°，直接使用 yaw 数值却看起来相差 358°。使用：

```python
[sin(yaw), cos(yaw)]
```

可以消除这个边界不连续问题。

### 5.3 为什么读取实际状态而不是命令

夹爪 observation 读取的是实际宽度，而不是 `ctrl`：

```text
ctrl：控制器希望夹爪到哪里
实际宽度：动力学执行后夹爪现在在哪里
```

在发生接触、负载或尚未收敛时，两者可能不同。任务观察应该描述当前物理
状态，而不是只描述控制器的愿望。

### 5.4 observation 是独立副本

返回值由新 NumPy 数组组成。修改 observation 不会反向修改：

```python
scene.data.qpos
scene.data.qvel
```

这避免外部策略因为误改 observation 而直接污染 MuJoCo 内部状态。

---

## 6. Day20 的两个动作

动作定义为：

```python
class PickAction(IntEnum):
    WAIT = 0
    RUN_SCRIPTED_PICK = 1
```

这是高层离散动作，不是七个关节的底层控制量。

### 6.1 `WAIT`

`WAIT` 不改变已有控制目标，但仍然推进动力学：

```python
for _ in range(wait_physics_steps):
    apply_arm_bias_compensation(scene)
    mujoco.mj_step(scene.model, scene.data)
```

因此：

- 仿真时间增加；
- 物体仍受重力和接触力影响；
- 上层 `episode_steps` 只增加 1；
- 一个上层 step 可以包含多个底层 `mj_step()`。

如果只增加 `episode_steps` 而不调用 `mj_step()`，MuJoCo 物理世界不会前进。

### 6.2 `RUN_SCRIPTED_PICK`

这个动作调用 Day19 的完整抓取状态机：

```python
pick_result = run_scripted_pick(
    scene,
    seed=self.seed,
    reset_scene=False,
)
```

它不是一个瞬时底层动作，而是一个高层宏动作。一次 `task.step()` 内部会
执行多个状态和大量物理步，直到得到明确成功或失败结果。

---

## 7. 为什么增加 `reset_scene=False`

Day19 的 `run_scripted_pick()` 原本自己执行 RESET。Day20 又要求使用：

```python
task.reset(seed)
task.step(RUN_SCRIPTED_PICK)
```

如果状态机在 step 内再次 reset，会出现：

```text
task.reset(seed)
    ↓ 第一次生成并落稳方块
用户/策略读取observation
    ↓
task.step(...)
    ↓ 状态机第二次重置场景
策略看到的初始状态与真正执行的状态不再属于同一个连续episode
```

即使相同 seed 会生成相同随机位姿，这仍然会：

- 重复花费 1.5 s 仿真落稳时间；
- 清除 reset 后到 step 前产生的状态变化；
- 破坏标准任务接口的语义；
- 让 `WAIT` 等动作的影响被悄悄清除。

因此 Day19 状态机增加：

```python
reset_scene: bool = True
```

兼容关系为：

```text
直接调用run_scripted_pick：reset_scene=True
    保持Day19原有行为

通过PandaPickTask调用：reset_scene=False
    使用task.reset已经准备好的当前场景
```

当 `reset_scene=False` 时，状态机仍保留逻辑上的 RESET 记录，但不会再次
随机化或推进落稳过程。

---

## 8. reward、terminated 与 truncated

`step()` 返回五项：

```python
observation, reward, terminated, truncated, info
```

### 8.1 reward

当前使用最简单的稀疏奖励：

```text
抓取成功：reward = 1.0
等待或抓取失败：reward = 0.0
```

它足以表达当前任务目标，但还不是为强化学习精心设计的奖励函数。

### 8.2 terminated

表示任务自身已经得到明确结果：

```text
抓取成功 → terminated=True
抓取失败 → terminated=True
```

失败同样属于明确结局，因此也应该 terminated。

### 8.3 truncated

表示 episode 因外部限制停止，而不是任务自然得出成功或失败：

```text
连续WAIT达到max_episode_steps
→ terminated=False
→ truncated=True
→ reason="time_limit"
```

### 8.4 两者区别

```text
terminated：任务本身结束了
truncated：任务还没有结果，但不允许继续运行了
```

例子：

| 情况 | terminated | truncated |
|---|---:|---:|
| 抓取成功 | True | False |
| 接触失败 | True | False |
| IK失败 | True | False |
| 一直WAIT直到步数上限 | False | True |

episode 结束后再次调用 `step()` 会报错，必须先调用 `reset()` 开启新回合。

---

## 9. info 与失败分类

observation 面向策略，info 面向人类调试和批量统计。

成功抓取的 info 包含：

```python
{
    "success": True,
    "final_state": "DONE",
    "failed_stage": None,
    "reason": "none",
    "failure_category": "none",
    "lift_height": 0.07729,
    "hold_duration": 0.5,
    "gripper_width": 0.03993,
    "stage_trace": (...),
}
```

详细原因被进一步归为四个批量统计类别：

```text
planning_or_motion_failure
contact_failure
lift_failure
stability_failure
```

这样不仅能统计总成功率，还能回答：

```text
失败主要发生在IK规划、接触建立、抬升还是最终稳定性？
```

单独的 `success=False` 无法提供这种诊断能力。

---

## 10. 单次任务接口探针

运行：

```powershell
& "C:\Users\29391\.conda\envs\mujoco\python.exe" `
  "examples\day20\task_scripted_pick_probe.py"
```

结果：

```text
reward: 1.0
terminated: True
truncated: False
success: True
final_state: DONE
failed_stage: None
reason: none
failure_category: none
lift_height: 0.0772903435 m
hold_duration: 0.5 s
gripper_width: 0.0399309804 m
episode_steps: 1
```

状态轨迹为：

```text
RESET
MOVE_ABOVE
APPROACH
CLOSE_GRIPPER
VERIFY_CONTACT
LIFT
CHECK_SUCCESS
DONE
```

其中 RESET 是任务流程中的逻辑记录，方块实际复位已经由 `task.reset()`
完成。

---

## 11. WAIT 截断探针

配置：

```python
PandaPickTask(
    max_episode_steps=3,
    wait_physics_steps=10,
)
```

连续三次 WAIT：

```text
step 1 → RUNNING
step 2 → RUNNING
step 3 → TRUNCATED, reason=time_limit
```

每个 WAIT 推进：

```text
10 × 0.002 s = 0.02 s
```

因此仿真时间从约 `1.50 s` 依次变为：

```text
1.52 s
1.54 s
1.56 s
```

这证明上层 episode step 与底层物理 step 是两个不同时间尺度。

---

## 12. 多 seed 批量评估

运行：

```powershell
& "C:\Users\29391\.conda\envs\mujoco\python.exe" `
  "examples\day20\evaluate_scripted.py"
```

默认评估 seed 0～4：

```text
case 00 | seed=0 | cube=(+0.4582, -0.0276) | yaw=-27.541° | success=True
case 01 | seed=1 | cube=(+0.4507, +0.0541) | yaw=-21.351° | success=True
case 02 | seed=2 | cube=(+0.4357, -0.0242) | yaw=+18.854° | success=True
case 03 | seed=3 | cube=(+0.4251, -0.0316) | yaw=+18.075° | success=True
case 04 | seed=4 | cube=(+0.4766, +0.0014) | yaw=+28.574° | success=True
```

汇总结果：

```text
success count:       5/5
success rate:        100.00%
terminated episodes: 5
truncated episodes:  0
failure categories:  {}
failed stages:       {}
```

逐回合结果保存为：

```text
results/day20/scripted_pick_evaluation.csv
```

CSV 保存了：

- seed；
- 初始方块 x、y、z、yaw；
- reward；
- terminated/truncated；
- success；
- final_state；
- failed_stage；
- reason 和 failure_category；
- 抬升高度、保持时间、夹爪宽度；
- 实际程序墙上时间。

可以增加评估数量：

```powershell
& "C:\Users\29391\.conda\envs\mujoco\python.exe" `
  "examples\day20\evaluate_scripted.py" `
  --cases 20 `
  --start-seed 0
```

5/5 只表示当前五个确定性样本全部通过，不能证明整个连续工作空间的成功率
为 100%。可靠性结论必须同时说明采样范围、样本数量和 seed 集合。

---

## 13. 自动测试

Day20 测试文件：

```text
tests/test_pick_task.py
```

共 19 项，覆盖：

1. observation 固定为 29 维；
2. observation 所有元素有限；
3. observation 是独立数组；
4. 相同 seed 完全可复现；
5. 不同 seed 改变随机初始状态；
6. WAIT 在限制前保持 RUNNING；
7. WAIT 达到限制时设置 truncated；
8. 成功抓取设置 terminated；
9. 成功奖励为 1.0；
10. reset 前不能 step；
11. episode 结束后不能继续 step；
12. 非法动作被拒绝；
13. 非法 episode 配置被拒绝；
14. 四类失败阶段可以正确归类。

结果：

```text
19 passed
```

全项目回归测试：

```text
160 passed in 8.25s
```

这说明 Day20 的任务包装没有破坏之前的运动学、控制、接触监测和抓取状态机。

---

## 14. 探针、批量评估和 pytest 的区别

三者关注点不同：

```text
探针
  → 展示某个机制如何工作，输出适合人阅读的数据

批量评估
  → 在多个任务初始条件下衡量策略表现，保存逐回合结果

pytest
  → 自动判断软件接口和关键性质是否被后续修改破坏
```

例如：

- `task_scripted_pick_probe.py` 帮助理解一次任务返回了什么；
- `evaluate_scripted.py` 统计多个 seed 的成功率和失败位置；
- `test_pick_task.py` 保证接口语义长期保持正确。

它们不是重复代码，而是在回答三个不同问题。

---

## 15. 当前局限

### 15.1 action 仍是高层宏动作

`RUN_SCRIPTED_PICK` 一次调用就执行完整固定策略。它还不是让学习算法每个时间步
输出关节速度或末端增量的连续控制环境。

### 15.2 reward 很稀疏

目前只有成功时奖励 1，其余为 0。将来可以考虑加入：

- 末端到方块距离；
- 姿态误差；
- 双侧接触；
- 抬升高度；
- 动作平滑或能耗惩罚。

但奖励设计会改变学习目标，不能为了“数值更多”而随意叠加。

### 15.3 随机化范围仍较小

当前只随机：

- 方块 x；
- 方块 y；
- 方块 yaw。

尚未随机质量、摩擦、尺寸、传感器噪声和控制延迟，因此不能把当前结果解释为
sim-to-real 鲁棒性证明。

### 15.4 当前样本量较小

默认五个 case 适合快速开发检查。要报告策略可靠性，需要更大的样本量和明确的
置信区间或失败案例分析。

---

## 16. Day20 需要掌握的内容

完成今天后，应能够解释：

1. 为什么同一个 seed 会生成同一个方块位姿；
2. cube joint 的 qpos 与 cube body xpos 有什么区别；
3. `reset()` 为什么要清除完整运行时状态；
4. 29维 observation 每个区间表示什么；
5. 为什么 yaw 使用 sin/cos 表示；
6. observation 和 info 分别服务于谁；
7. `WAIT` 为什么仍然必须调用 `mj_step()`；
8. 一个 task step 为什么可以包含多个 physics step；
9. terminated 与 truncated 的区别；
10. 为什么抓取失败也属于 terminated；
11. 为什么 task.reset 后状态机不能再次悄悄 reset；
12. 为什么批量评估不能只保存一个成功布尔值；
13. 为什么 5/5 不能直接推出真实成功率是 100%。

---

## 17. 今日结论

Day20 将项目从“能够运行一个抓取脚本”推进为“具有明确任务边界的可评估系统”：

```text
可复现随机初始化
        +
固定数值观察
        +
明确动作接口
        +
标准episode结束语义
        +
结构化诊断信息
        +
多seed批量评估
```

这一层接口是后续数据采集、策略比较、模仿学习和强化学习环境封装的基础。
