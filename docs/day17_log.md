---
exp_id: PM-B-260917-001
date: 2026-09-17
project: panda-mujoco-manipulation
day: 17
exp_type: 抓取运动原语与方块姿态对齐
platform: Windows
python: 3.10.20
mujoco: 3.12.0
acceptance_seeds: [42, 43, 44]
motion_test_result: "10 passed"
grasp_geometry_test_result: "16 passed"
full_test_result: "103 passed in 3.73s"
anomaly: true
anomaly_status: resolved
anomaly_summary: 初版固定世界方向的抓取姿态未跟随方块随机yaw
result: passed
tags: [MuJoCo, Panda, motion-primitive, trajectory, IK, grasp, yaw, pytest]
---

# Day 17 日志：抓取运动原语与方块姿态对齐

## 1. 实验目的

Day15 定义了 `pregrasp`、`grasp` 和 `lift` 抓取目标，Day16 实现了可复现的方块随机复位与落稳。Day17 的目标是把已有能力组合成可复用的抓取运动原语：

```text
方块落稳后的实际位姿
    ↓
生成pregrasp和grasp目标
    ↓
在独立规划场景中求6D IK
    ↓
生成连续关节命令
    ↓
通过ctrl和mj_step执行动力学运动
    ↓
稳定等待并检查实际位姿误差
```

今天完成两个抓取前动作：

1. `MOVE_ABOVE`：从 home 移动到方块上方的 `pregrasp` 位姿；
2. `APPROACH`：夹爪保持张开，从 `pregrasp` 下降到 `grasp` 位姿。

今天没有执行夹爪闭合、持续双侧接触确认和抬升方块。

---

## 2. 运动原语是什么

运动原语是一个输入、输出和失败原因都明确的可复用机器人基本动作。

Day17 封装的上层接口为：

```python
result = move_end_effector_to(
    control_scene,
    target_position,
    target_rotation,
)
```

调用者只提供末端目标位姿，函数内部完成：

```text
目标末端位姿
    ↓
plan_pose_target()
    ↓
目标关节角q_goal
    ↓
linear_joint_trajectory()
    ↓
连续关节命令
    ↓
execute_joint_trajectory()
    ↓
真实动力学执行
    ↓
MotionResult
```

Day17 的重点不是发明新的 IK 或控制算法，而是将 Day10～Day16 的底层能力组合成可调用、可验收、可诊断的工程接口。

---

## 3. qpos 摆放与 ctrl 控制的区别

直接写入：

```python
scene.data.qpos[:7] = q_goal
mujoco.mj_forward(...)
```

只是将模型摆放到目标关节状态。它没有：

- 中间轨迹；
- 运动时间；
- 速度与惯性；
- 执行器跟踪误差；
- 重力与接触过程。

真正的动力学运动使用：

```python
command_arm_joint_positions(scene, q_command)
apply_arm_bias_compensation(scene)
mujoco.mj_step(scene.model, scene.data)
```

其中 `command_arm_joint_positions()` 只写入 `ctrl`，实际 `qpos` 由执行器、动力学和数值积分共同产生。

因此：

```text
直接改qpos = 改变状态，常用于初始化或运动学计算
写ctrl+mj_step = 施加控制并生成物理运动
```

---

## 4. 规划场景与控制场景

### 4.1 规划场景

`plan_pose_target()` 创建独立 `PandaScene`，并复制控制场景当前 `qpos`：

```python
planning_scene.data.qpos[:] = (
    control_scene.data.qpos.copy()
)
```

随后在规划场景中调用 `solve_pose_ik()`。IK 可以修改规划场景的 `qpos`，但不会让控制场景中的机器人瞬移。

规划验证结果：

```text
success: True
reason: none
IK iterations: 32
position error: 0.001299 mm
orientation error: 0.0000936 deg
control qpos unchanged: True
control ctrl unchanged: True
control time unchanged: True
```

### 4.2 控制场景

控制场景只通过 `ctrl` 和 `mj_step()` 执行动作。它会体现：

- 执行器跟踪滞后；
- 连杆惯性；
- 重力和 bias 力；
- 接触与摩擦；
- 真实的仿真时间。

第二个动作 `APPROACH` 的规划起点是第一个动作完成后的实际 `pregrasp` 状态，而不是 home。这避免了相邻动作之间的关节目标跳变。

---

## 5. 关节空间线性轨迹

`linear_joint_trajectory()` 使用：

$$
q(\alpha)=q_{start}+\alpha(q_{goal}-q_{start}),
\qquad 0\leq\alpha\leq1
$$

对 Panda 七个关节同时插值。轨迹数组形状为：

```text
(command_count, 7)
```

其中：

- 每一行是一个时刻的七关节完整命令；
- 每一列是一个关节随时间的变化；
- 第一行等于起点；
- 最后一行等于终点。

五个命令的简化例子：

```text
             joint1  joint2  joint3
0%            0.0     0.0     0.00
25%           0.1    -0.1     0.05
50%           0.2    -0.2     0.10
75%           0.3    -0.3     0.15
100%          0.4    -0.4     0.20
```

所有关节在相同时刻完成相同比例的各自运动量。

需要注意：关节空间是直线，不代表末端在世界坐标系中严格走直线。

---

## 6. 轨迹执行、控制频率与仿真时间

`execute_joint_trajectory()` 使用两层循环：

```python
for joint_command in joint_trajectory:
    command_arm_joint_positions(scene, joint_command)

    for _ in range(physics_steps_per_command):
        apply_arm_bias_compensation(scene)
        mujoco.mj_step(scene.model, scene.data)
```

当：

```text
command_count = 101
physics_steps_per_command = 10
MuJoCo timestep = 0.002 s
```

轨迹阶段时间为：

$$
101\times10\times0.002=2.02\text{ s}
$$

`apply_arm_bias_compensation()` 在每个物理步之前调用，因为 `qfrc_bias` 会随当前 `qpos` 和 `qvel` 变化。

---

## 7. 为什么需要最终稳定阶段

首次关节轨迹动画中，joint1 的目标是 `0.2 rad`，轨迹结束瞬间的实际值为：

```text
target: 0.200000 rad
actual: 0.191058 rad
error:  0.008942 rad
```

这不是插值错误，而是位置执行器的动力学跟踪滞后。最后一条 `ctrl=0.2` 只保持了 `10×0.002=0.02 s`，机械臂尚未完全收敛。

增加：

```text
settle_steps = 250
```

即继续保持最终目标 `0.5 s`。修改后：

| 阶段 | 稳定前最大关节误差 | 稳定后最大关节误差 |
|---|---:|---:|
| 去程 | `8.942e-03 rad` | `4.849e-05 rad` |
| 回程 | `8.940e-03 rad` | `4.847e-05 rad` |

稳定阶段需要继续调用 `mj_step()`。`time.sleep()` 只会让 Python 等待，不会推进 MuJoCo 的物理状态。

---

## 8. 可视化回调

`execute_joint_trajectory()` 接受可选：

```python
step_callback: Callable[[PandaScene], None] | None
```

无界面测试时传入 `None`，仿真尽快运行。视觉演示中，每次 `mj_step()` 后调用：

```python
viewer.sync()
time.sleep(scene.model.opt.timestep)
```

这样同一套运动执行代码既能用于快速自动化测试，也能用于接近真实时间的 MuJoCo 动画。

---

## 9. PosePlan 与 MotionResult

### 9.1 PosePlan

`PosePlan` 保存规划阶段的结果：

```text
success
reason
joint_target
ik_iterations
position_error_norm
orientation_error_norm
```

规划失败时使用 `reason="ik_failed"`。

### 9.2 MotionResult

`MotionResult` 保存动力学执行后的结果：

```text
success
reason
simulation_duration
ik_iterations
target_joint_positions
final_joint_positions
final_position
final_rotation
position_error_norm
orientation_error_norm
maximum_joint_error
joint_limit_violation
gripper_width
```

支持的失败原因包括：

```text
ik_failed
non_finite_state
joint_limit_violation
position_tolerance
orientation_tolerance
```

成功时使用：

```text
success=True
reason=none
```

这比单独返回 `True/False` 更适合探针、pytest 和后续抓取状态机。

---

## 10. MOVE_ABOVE 视觉实验

seed 42 方块落稳后：

```text
cube position: [ 0.466437, -0.007335, 0.019928 ] m
pregrasp:      [ 0.466437, -0.007335, 0.124928 ] m
```

首次 `MOVE_ABOVE` 执行结果：

```text
success: True
reason: none
simulation duration: 4.52 s
IK iterations: 32
position error: 0.081341 mm
orientation error: 0.023412 deg
maximum joint error: 3.778e-04 rad
joint limit violation: False
gripper width: 0.080000 m
```

这证明机械臂可以从 home 通过动力学运动到方块上方，并在整个动作中保持夹爪张开。

---

## 11. 异常：夹爪未跟随方块 yaw

### 11.1 现象

首次 `APPROACH` 动画中，机械臂的数值验收全部通过，但视觉上夹爪与红色方块的边缘不平行。

当时的方块 yaw 约为：

```text
+21.516 deg
```

而目标旋转矩阵仍为：

```text
[[ 1,  0,  0],
 [ 0, -1,  0],
 [ 0,  0, -1]]
```

该矩阵是固定的世界 `yaw=0 deg` 俯视姿态。

### 11.2 为什么数值验收仍然通过

当时的姿态误差衡量的是：

```text
实际夹爪姿态
    对比
固定yaw=0 deg目标
```

而不是：

```text
实际夹爪姿态
    对比
方块实际yaw
```

因此，控制器准确到达了一个不完整的目标。这说明：

> 控制误差很小，只能证明机器人准确执行了目标；目标本身是否合理，还必须通过任务几何和可视化验证。

### 11.3 根本原因

Day15 的 `generate_grasp_targets()` 只接收方块位置，并假设方块 yaw 为0。Day16 新增了 `[-30 deg,+30 deg]` 随机 yaw，但当时没有真正执行机械臂抓取动作。

两个模块在 Day17 集成时才暴露信息不完整：

```text
Day15：固定世界方向的抓取姿态
Day16：方块具有随机yaw
Day17：组合后发现姿态未对齐
```

### 11.4 修复

将方块 yaw 记为 $\theta$，新的抓取目标姿态为：

$$
R_{target}=R_z(\theta)R_{down}
$$

其中：

$$
R_z(\theta)=
\begin{bmatrix}
\cos\theta&-\sin\theta&0\\
\sin\theta&\cos\theta&0\\
0&0&1
\end{bmatrix}
$$

修改后：

- 末端局部 $x$ 轴跟随方块局部 $x$ 轴；
- 夹爪张合轴与方块局部 $y$ 轴对齐；
- 末端局部 $z$ 轴仍为世界 `-z`，保持向下接近。

落稳后的实际 yaw 从 body 旋转矩阵恢复：

```python
cube_yaw = np.arctan2(
    cube_rotation[1, 0],
    cube_rotation[0, 0],
)
```

这比直接使用采样 yaw 更稳健，因为方块落地过程可能产生轻微旋转。

### 11.5 修复验证

seed 42 落稳后：

```text
cube yaw: 21.515119 deg
```

新目标矩阵：

```text
[[ 0.930321,  0.366747,  0],
 [ 0.366747, -0.930321,  0],
 [ 0,         0,        -1]]
```

轴对齐点积：

```text
gripper/cube axis alignment: 1.0000000000000016
```

理论值为1，多出的 `1.6e-15` 是浮点误差。修复后动画中夹爪水平方向与方块边缘对齐。

---

## 12. 最终视觉演示结果

seed 42 修复 yaw 后：

### MOVE_ABOVE

```text
success: True
reason: none
simulation duration: 4.52 s
IK iterations: 28
position error: 0.081255 mm
orientation error: 0.020268 deg
maximum joint error: 3.171e-04 rad
joint limit violation: False
gripper width: 0.080000 m
```

### APPROACH

```text
success: True
reason: none
simulation duration: 2.52 s
IK iterations: 4
position error: 0.024326 mm
orientation error: 0.001144 deg
maximum joint error: 6.989e-05 rad
joint limit violation: False
gripper width: 0.080000 m
```

可视化过程确认：

1. 机械臂从 home 连续运动到方块上方；
2. 夹爪姿态跟随方块落稳后的实际 yaw；
3. 机械臂短暂停留后向下接近 `grasp` 点；
4. 夹爪在两个阶段始终保持完全张开；
5. 当前没有执行闭合和抬升。

---

## 13. 三随机案例无界面验收

`motion_primitives_probe.py` 对 seed 42、43、44 分别创建全新 `PandaScene`，保证每个案例都从 home 开始。

| seed | 方块 x/y | yaw | 阶段 | IK | 位置误差 | 姿态误差 | 最大关节误差 | 夹爪宽度 |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 42 | `(+0.4664,-0.0073)` | `+21.515 deg` | MOVE_ABOVE | 28 | 0.081255 mm | 0.020268 deg | `3.171e-04 rad` | 0.080000 m |
| 42 | `(+0.4664,-0.0073)` | `+21.515 deg` | APPROACH | 4 | 0.024326 mm | 0.001144 deg | `6.989e-05 rad` | 0.080000 m |
| 43 | `(+0.4591,-0.0547)` | `-28.798 deg` | MOVE_ABOVE | 37 | 0.082330 mm | 0.190484 deg | `3.116e-03 rad` | 0.080000 m |
| 43 | `(+0.4591,-0.0547)` | `-28.798 deg` | APPROACH | 4 | 0.024454 mm | 0.001159 deg | `7.114e-05 rad` | 0.080000 m |
| 44 | `(+0.4274,-0.0290)` | `-5.654 deg` | MOVE_ABOVE | 32 | 0.097190 mm | 0.028578 deg | `3.794e-04 rad` | 0.080000 m |
| 44 | `(+0.4274,-0.0290)` | `-5.654 deg` | APPROACH | 4 | 0.076195 mm | 0.009046 deg | `7.881e-05 rad` | 0.080000 m |

汇总：

```text
cases: 3
maximum position error: 0.097190 mm
maximum orientation error: 0.190484 deg
maximum joint error: 0.00311555 rad
minimum gripper width: 0.07999999 m
total motion simulation duration: 21.12 s
```

总运动时间与理论值一致：

$$
3\times(4.52+2.52)=21.12\text{ s}
$$

结果：

```text
Day 17 motion primitives: PASSED
```

---

## 14. 验收标准

Day17 使用：

```text
位置误差 <= 2 mm
姿态误差 <= 1 deg
关节不超限
状态中无NaN/Inf
APPROACH前后夹爪宽度约0.08 m
```

实验最差值：

```text
位置误差 = 0.097190 mm < 2 mm
姿态误差 = 0.190484 deg < 1 deg
夹爪宽度最小值 = 0.07999999 m
```

最差位置误差约为2 mm门槛的二十分之一，最差姿态误差低于1 deg门槛。

---

## 15. 自动化测试

### 15.1 抓取几何回归测试

Day15 原有12项抓取几何测试。针对 yaw 对齐修复，增加：

- 目标末端水平轴跟随方块 yaw；
- 接近方向仍为世界 `-z`；
- `NaN`、`+Inf`、`-Inf` yaw 被拒绝。

结果：

```text
16 passed
```

### 15.2 运动原语测试

`tests/test_motion.py` 收集10项测试，覆盖：

- 线性关节轨迹形状、起点、中点和终点；
- 输入数组不被修改；
- 5类非法 `command_count` 被拒绝；
- IK 规划不修改控制场景；
- 轨迹执行正确推进仿真时间；
- 实际关节通过动力学接近目标；
- 完整末端运动原语可到达邻近位姿。

结果：

```text
10 passed
```

### 15.3 完整项目回归

```text
103 passed in 3.73s
```

无失败、跳过或警告。新增的运动原语和 yaw 对齐逻辑没有破坏之前的 reset、夹爪、运动学、Jacobian、IK、轨迹与方块复位功能。

---

## 16. 探针与 pytest 的职责

Day17 同时使用了动画、探针和 pytest：

### 动画

用于检查很难仅通过数字发现的语义问题。本次夹爪未与方块 yaw 对齐，就是由动画先发现的。

### 探针

负责人可读的端到端诊断：

- 显示每个随机案例；
- 显示每个动作阶段；
- 打印 IK 迭代、位姿误差、时间、夹爪宽度和失败原因；
- 汇总最差案例。

### pytest

负责快速、隔离、可自动重复的回归门禁：

- 哪一条规则失败可以精确定位；
- 以后修改其他模块时自动防止已修复问题回归；
- 适合在提交前或 CI 中一次性执行。

三者不是互相替代，而是分别解决视觉语义、端到端诊断和自动回归问题。

---

## 17. 今日实现的公共接口

### `top_down_grasp_rotation(cube_yaw)`

根据方块 yaw 生成对齐方块边缘的向下抓取姿态。

### `plan_pose_target(...)`

复制当前控制状态到独立规划场景，求解目标位姿 IK，返回 `PosePlan`。

### `linear_joint_trajectory(...)`

在起始和目标关节角之间生成包含两端点的线性插值轨迹。

### `execute_joint_trajectory(...)`

逐条写入关节位置命令、每步应用 bias 补偿并推进动力学；支持稳定阶段和可视化回调。

### `move_end_effector_to(...)`

将位姿规划、轨迹生成、动力学执行和最终验收封装为一个末端运动原语，返回 `MotionResult`。

---

## 18. 文件清单

```text
src/panda_mujoco/motion.py
    运动规划、关节轨迹、动力学执行与MotionResult

src/panda_mujoco/grasp_task.py
    增加跟随方块yaw的向下抓取姿态

examples/day17/joint_trajectory_visual_demo.py
    home到joint1目标再返回home的动画

examples/day17/motion_primitives_visual_demo.py
    MOVE_ABOVE与APPROACH的完整动画

examples/day17/motion_primitives_probe.py
    seed 42/43/44三案例无界面验收

tests/test_motion.py
    10项运动原语单元与集成测试

tests/test_grasp_task.py
    增加yaw对齐与非法yaw回归测试

docs/day17_log.md
    Day17理论、实现、异常、修复与验收记录
```

---

## 19. 当前限制

1. 当前使用关节空间线性插值，末端轨迹不保证严格为笛卡尔直线；
2. 轨迹起停处速度不连续，暂未使用 smoothstep、梯形速度或 S 曲线；
3. 动作时间使用固定 `command_count` 和 `settle_steps`，没有根据误差提前停止；
4. 规划阶段只检查 IK 和关节限位，尚未实现轨迹碰撞检查；
5. `APPROACH` 仅验收最终位姿，没有记录整段末端路径是否严格竖直；
6. 夹爪尚未闭合，没有双侧接触、接触力或抓取成功判定；
7. 当前的 yaw 对齐假设方块保持直立，没有处理任意 roll/pitch 物体姿态。

---

## 20. Day17 需要掌握的内容

完成 Day17 后，应能解释：

1. 直接写 `qpos` 为什么不等于运动控制；
2. 为什么 IK 规划和动力学执行要使用独立场景；
3. 为什么第二个动作必须从第一个动作的实际结束状态规划；
4. 关节轨迹数组的行和列分别表示什么；
5. 线性关节插值的数学原理；
6. 为什么关节空间直线不保证末端笛卡尔轨迹是直线；
7. `physics_steps_per_command` 如何决定控制更新间隔；
8. 如何由命令数、每条命令的物理步数和 timestep 计算仿真时间；
9. 为什么轨迹结束瞬间仍存在跟踪滞后；
10. 为什么稳定阶段要调用 `mj_step()` 而不是只 `sleep()`；
11. 为什么 bias 补偿需要每个物理步更新；
12. `PosePlan` 和 `MotionResult` 分别记录什么；
13. 为什么数值验收通过不一定表示任务目标合理；
14. 如何使用 $R_z(\theta)R_{down}$ 让夹爪跟随方块 yaw；
15. 如何用轴点积检查夹爪与方块是否对齐；
16. 动画、探针和 pytest 分别解决什么问题。

---

## 21. Day17 验收结论

- [x] 实现独立规划场景的末端位姿规划；
- [x] 规划不修改控制场景 `qpos`、`qvel`、`ctrl` 和时间；
- [x] 实现七关节线性轨迹生成；
- [x] 实现基于 `ctrl` 和 `mj_step()` 的动力学轨迹执行；
- [x] 每个物理步更新 bias 补偿；
- [x] 增加最终稳定阶段，显著减小跟踪误差；
- [x] 实现可选可视化回调；
- [x] 实现 `PosePlan` 和 `MotionResult`；
- [x] 实现 `move_end_effector_to()` 运动原语；
- [x] 完成 `MOVE_ABOVE` 动画与动力学验收；
- [x] 完成 `APPROACH` 动画与动力学验收；
- [x] 发现固定夹爪姿态未跟随方块 yaw；
- [x] 将抓取姿态扩展为 yaw-aware；
- [x] 轴对齐点积为1；
- [x] seed 42、43、44 的两阶段运动全部成功；
- [x] 最大位置误差0.097190 mm；
- [x] 最大姿态误差0.190484 deg；
- [x] 三案例夹爪都保持约0.08 m张开；
- [x] Day17 的10项运动测试全部通过；
- [x] 抓取几何测试扩展到16项并全部通过；
- [x] 完整项目103项测试全部通过。

**Day17：PASSED**

---

## 22. 下一步

Day17 已经将张开的夹爪稳定移动到方块两侧。下一阶段将建立闭合与抓取判定：

```text
APPROACH完成
    ↓
逐步闭合夹爪
    ↓
检查左右手指—方块接触
    ↓
要求接触和物体状态持续一段时间
    ↓
保持闭合命令并进入抬升阶段
```

后续不能只用单一时刻的接触数判定抓取成功，而需要结合双侧接触、方块位移和持续时间。
