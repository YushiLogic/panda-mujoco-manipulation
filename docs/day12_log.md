---
exp_id: PANDA-SIM-260909-001
date: 2026-09-09
system: Panda MuJoCo Manipulation
exp_type: Cartesian waypoint trajectory tracking
scene: assets/robots/panda/scene_with_cube.xml
python: 3.10.20
mujoco: 3.12.0
numpy: 2.2.6
matplotlib: 3.10.9
status: passed
test_count: 41
anomaly: true
anomaly_status: resolved
tags: [Panda, MuJoCo, Cartesian trajectory, waypoint IK, dynamics, error decomposition]
---

# Day 12：连续笛卡尔路点与轨迹误差分解

## 1. 实验目的

Day 10 实现了单个末端目标的位置 IK，Day 11 将一个 IK 关节目标接入执行器和 MuJoCo 动力学。Day 12 将“单个终点”扩展为“连续空间路径”。

本日目标为：

1. 在末端起点和目标点之间生成等间距笛卡尔路点；
2. 使用同一个规划场景依次为所有路点求 IK；
3. 验证相邻 IK 关节解保持连续；
4. 通过关节位置执行器和偏置力补偿执行整条轨迹；
5. 记录理想笛卡尔目标、运动学参考和动力学实际轨迹；
6. 将总误差拆分为几何误差和动力学误差；
7. 保存结构化实验数据并生成可复现图表；
8. 将路点生成和连续 IK 规划整理为正式模块并添加测试。

最终需要回答的问题是：

> 机械臂运动过程中约 1.25 mm 的末端误差，主要来自关节空间插值偏离直线，还是执行器的动力学跟踪滞后？

---

## 2. 实验条件

### 2.1 软件与模型

```text
Python:     3.10.20
MuJoCo:     3.12.0
NumPy:      2.2.6
Matplotlib: 3.10.9
Robot:      Franka Emika Panda
Scene:      scene_with_cube.xml
```

### 2.2 起点和目标

Panda home 姿态下的末端起点：

```text
[0.688796086, 0.000000000, 0.788705734] m
```

目标偏移：

```text
[+0.020, -0.010, +0.015] m
```

目标位置：

```text
[0.708796086, -0.010000000, 0.803705734] m
```

### 2.3 路点和控制参数

| 参数 | 数值 | 含义 |
|---|---:|---|
| `WAYPOINT_COUNT` | 6 | 包含起点和终点的笛卡尔路点数 |
| 路径段数 | 5 | 相邻路点之间的线段数 |
| `TICKS_PER_SEGMENT` | 20 | 每段使用的控制拍数 |
| `CONTROL_EVERY` | 10 | 每个控制拍包含的物理步数 |
| MuJoCo timestep | 0.002 s | 每次 `mj_step()` 推进的时间 |
| 物理频率 | 500 Hz | `1 / 0.002` |
| 控制频率 | 50 Hz | `500 / 10` |
| 单段时间 | 0.4 s | `20 / 50` |
| 运动总时间 | 2.0 s | `5 × 0.4` |
| 稳定时间 | 1.0 s | 最终目标保持500个物理步 |
| 仿真总时间 | 3.0 s | 运动2秒加稳定1秒 |

### 2.4 IK 参数

| 参数 | 数值 |
|---|---:|
| DLS damping | 0.05 |
| position tolerance | `1e-4 m` |
| maximum iterations | 50 |
| maximum joint step | `0.1 rad` |

动力学执行过程中启用 Day 11 建立的理想偏置力前馈补偿。

---

## 3. 实验一：生成笛卡尔直线路点

文件：

```text
examples/day12/cartesian_waypoint_probe.py
```

正式接口：

```python
waypoints = linear_position_waypoints(
    start_position,
    target_position,
    waypoint_count=6,
)
```

内部等价于在起点和终点之间进行等间距插值：

```text
p_i = p_start + s_i * (p_target - p_start)
```

其中：

```text
s_i = i / (N - 1)
```

### 3.1 生成的路点

```text
waypoint 0: [0.688796086,  0.000, 0.788705734]
waypoint 1: [0.692796086, -0.002, 0.791705734]
waypoint 2: [0.696796086, -0.004, 0.794705734]
waypoint 3: [0.700796086, -0.006, 0.797705734]
waypoint 4: [0.704796086, -0.008, 0.800705734]
waypoint 5: [0.708796086, -0.010, 0.803705734]
```

每段位移向量相同：

```text
[+0.004, -0.002, +0.003] m
```

即：

```text
[+4, -2, +3] mm
```

每段欧氏长度：

```text
0.005385165 m = 5.385165 mm
```

因此6个路点确实构成5段方向和长度一致的空间直线。

### 3.2 本实验验证的内容

- 路点数组形状为 `(6, 3)`；
- 第一个路点严格等于起点；
- 最后一个路点严格等于目标；
- 每段方向向量相同；
- 每段长度相同。

实验结果：

```text
Cartesian waypoint generation: PASSED
```

---

## 4. 实验二：连续路点 IK

文件：

```text
examples/day12/cartesian_waypoint_ik_probe.py
```

正式接口：

```python
plan = plan_position_waypoints(
    planning_scene,
    cartesian_waypoints,
)
```

### 4.1 为什么只使用一个规划场景

`solve_position_ik()` 会把求出的关节角写入传入场景的 `qpos`。因此依次求解时，流程为：

```text
q0 → 求p1的解q1
q1 → 求p2的解q2
q2 → 求p3的解q3
q3 → 求p4的解q4
q4 → 求p5的解q5
```

而不是每个路点都重新从 home 求解。

Panda 对三维位置任务是冗余机械臂，一个末端位置可能对应多组关节角。从上一个路点的解继续求下一个相近路点，有助于保持关节目标连续，减少不同逆解分支之间的跳变风险。

### 4.2 IK 结果

| 路点 | DLS更新次数 | 最终位置误差 | 相邻关节变化范数 |
|---:|---:|---:|---:|
| 0 | 0 | `0.000000e+00 m` | `0.000000e+00 rad` |
| 1 | 2 | `2.795266e-05 m` | `2.707440e-02 rad` |
| 2 | 2 | `3.025205e-05 m` | `2.777893e-02 rad` |
| 3 | 2 | `3.268209e-05 m` | `2.838446e-02 rad` |
| 4 | 2 | `3.545197e-05 m` | `2.903558e-02 rad` |
| 5 | 2 | `3.862944e-05 m` | `2.973902e-02 rad` |

最大路点位置误差：

```text
3.862944e-05 m = 0.03862944 mm
```

最大相邻关节变化范数：

```text
2.973902e-02 rad
```

该变化明显小于单次 IK 的 `0.1 rad` 关节步长上限，没有出现明显的关节解跳变。

实验结果：

```text
Sequential Cartesian waypoint IK: PASSED
```

---

## 5. 实验三：连续轨迹动力学执行

文件：

```text
examples/day12/cartesian_trajectory_tracking_probe.py
```

实验使用三个独立场景：

```text
planning_scene
└── 连续求解各笛卡尔路点的关节目标

reference_scene
└── 直接设置关节命令并调用 mj_forward，计算纯运动学参考

control_scene
└── 通过 ctrl、偏置力补偿和 mj_step 执行真实动力学
```

三个场景分别回答：

```text
planning_scene：每个路点应该对应什么关节角？
reference_scene：关节命令被完美实现时，末端应该在哪里？
control_scene：执行器和动力学作用后，末端实际上在哪里？
```

### 5.1 轨迹执行方式

每两个路点之间，不仅末端期望位置做线性插值，关节命令也在相邻 IK 解之间做线性插值：

```text
q_command(s) = q_start + s * (q_end - q_start)
p_desired(s) = p_start + s * (p_end - p_start)
```

其中每段：

```text
s = 1/20, 2/20, ..., 20/20
```

每个控制命令保持10个物理步，并在每个物理步前更新偏置力补偿。

---

## 6. 三类位置与三类误差

### 6.1 三类位置

在每个控制拍记录：

```text
p_desired
```

理想笛卡尔直线上的目标位置。

```text
p_reference
```

将关节命令直接放入纯运动学场景后得到的末端位置，表示关节命令被完全实现时的结果。

```text
p_actual
```

动力学控制场景实际到达的末端位置。

### 6.2 几何误差

```text
e_geometric = p_desired - p_reference
```

它衡量：

> 相邻 IK 解之间做关节空间线性插值时，形成的末端轨迹偏离理想笛卡尔直线多少。

### 6.3 动力学误差

```text
e_dynamic = p_reference - p_actual
```

它衡量：

> 实际关节没有瞬间、完全跟上关节命令所产生的末端误差。

### 6.4 总误差

```text
e_total = p_desired - p_actual
```

误差向量满足：

```text
e_total = e_geometric + e_dynamic
```

程序使用 `np.testing.assert_allclose()` 自动验证该向量关系。

需要注意：

```text
||e_total||
```

不一定等于：

```text
||e_geometric|| + ||e_dynamic||
```

因为误差向量可能指向不同方向。可以相加的是向量本身，不是它们的欧氏长度。

---

## 7. 轨迹执行结果

### 7.1 各笛卡尔路点的总误差

| 路点 | 仿真时间 | 几何误差 | 动力学误差 | 总误差 |
|---:|---:|---:|---:|---:|
| 1 | 0.4 s | `0.02795 mm` | `1.19528 mm` | `1.22082 mm` |
| 2 | 0.8 s | `0.03025 mm` | `1.22007 mm` | `1.24779 mm` |
| 3 | 1.2 s | `0.03268 mm` | `1.21993 mm` | `1.24994 mm` |
| 4 | 1.6 s | `0.03545 mm` | `1.21929 mm` | `1.25190 mm` |
| 5 | 2.0 s | `0.03863 mm` | `1.21856 mm` | `1.25416 mm` |

### 7.2 汇总指标

```text
maximum geometric error = 3.862944e-05 m = 0.038629 mm
mean geometric error    = 2.138043e-05 m = 0.021380 mm

maximum dynamic error   = 1.230719e-03 m = 1.230719 mm
mean dynamic error      = 1.171869e-03 m = 1.171869 mm

maximum total error     = 1.254162e-03 m = 1.254162 mm
mean total error        = 1.189618e-03 m = 1.189618 mm
```

运动结束时、尚未额外等待的误差：

```text
1.254162e-03 m = 1.254162 mm
```

最终目标保持1秒后的误差：

```text
3.867234e-05 m = 0.038672 mm
```

稳定后的最大关节误差：

```text
2.114107e-07 rad
```

### 7.3 结果解释

最大几何误差约为：

```text
0.039 mm
```

最大动力学误差约为：

```text
1.231 mm
```

动力学误差大约是几何误差的30倍。因此当前约 `1.25 mm` 的运动误差主要来自执行器跟踪滞后，而不是5段关节空间插值明显偏离笛卡尔直线。

机械臂停止运动并保持最终目标1秒后，误差降低到约 `0.039 mm`，说明：

- IK 最终目标正确；
- 偏置力补偿有效；
- 运动过程中存在速度相关的跟踪滞后；
- 静止后实际关节能够追上目标。

实验结果：

```text
Cartesian trajectory tracking: PASSED
```

---

## 8. 结构化数据与可视化

轨迹实验将数据保存为：

```text
outputs/day12/cartesian_trajectory_data.npz
```

数据包括：

```text
time
start_position
target_position
cartesian_waypoints
joint_targets
desired_position
reference_position
actual_position
geometric_error_norm
dynamic_error_norm
total_error_norm
settled_position
settled_final_error
```

使用 `.npz` 的原因：

- 保留 NumPy 数组的形状；
- 保留浮点精度；
- 不需要从终端字符串重新解析数据；
- 绘图时不需要重新运行仿真；
- 便于后续增加更多统计分析。

绘图脚本：

```text
figures/gen_fig_cartesian_trajectory.py
```

生成：

```text
outputs/day12/cartesian_trajectory_tracking.png
outputs/day12/cartesian_trajectory_tracking.pdf
```

图像包含两个面板：

```text
(a) 目标、运动学参考和动力学实际三维轨迹
(b) 几何、动力学和总误差随时间的变化
```

绘图约定：

- 坐标使用相对起点的毫米值；
- 使用色盲友好配色；
- PNG 使用300 DPI；
- PDF 为矢量格式；
- 竖直虚线标记每个路点时刻；
- 标注最大总误差。

`outputs/` 已被 `.gitignore` 忽略，生成数据和图片不会自动进入 Git。绘图脚本本身进入版本控制，任何使用者都可以重新生成结果。

---

## 9. 正式模块重构

新增：

```text
src/panda_mujoco/cartesian_trajectory.py
```

### 9.1 `linear_position_waypoints()`

```python
linear_position_waypoints(
    start_position,
    target_position,
    waypoint_count,
)
```

负责：

- 将输入转换为浮点数组副本；
- 检查起点和终点形状为 `(3,)`；
- 拒绝 `NaN` 和无穷大；
- 要求路点数为不小于2的整数；
- 返回包含起点和终点的等间距 `(N, 3)` 路点数组。

### 9.2 `plan_position_waypoints()`

```python
plan_position_waypoints(
    scene,
    cartesian_waypoints,
    damping=0.05,
    tolerance=1e-4,
    max_iterations=50,
    max_joint_step=0.1,
)
```

负责：

- 检查路点数组形状为 `(N, 3)`；
- 拒绝非有限数值；
- 始终使用同一个规划场景连续求解；
- 在路点失败时报告具体索引和最终误差；
- 保存每个路点的关节目标、迭代次数和误差。

### 9.3 `CartesianWaypointPlan`

规划结果使用 dataclass 保存：

```python
plan.cartesian_waypoints
plan.joint_targets
plan.ik_iterations
plan.ik_errors
```

对应形状：

```text
cartesian_waypoints: (N, 3)
joint_targets:       (N, 7)
ik_iterations:       (N,)
ik_errors:           (N,)
```

该结构比返回多个顺序不明确的数组更容易阅读，也便于以后增加姿态、速度或时间信息。

---

## 10. 自动化测试

新增：

```text
tests/test_cartesian_trajectory.py
```

新增4项测试：

1. 直线路点包含起点和终点，并且每段等间距；
2. 非法位置形状、`NaN` 和错误路点数量被拒绝；
3. 短距离可达轨迹产生有限、连续的关节目标；
4. 规划结果拥有独立数组副本，不受调用者后续修改影响。

完整测试数量由37增加到：

```text
41 passed
```

测试模块：

```text
tests/test_arm_control.py
tests/test_cartesian_trajectory.py
tests/test_gripper.py
tests/test_ik.py
tests/test_kinematics.py
tests/test_reset.py
```

---

## 11. 异常与修复记录

### 11.1 现象

将连续 IK 抽取到正式模块后，第一次运行示例时出现：

```text
waypoint 0 joint change norm = 0.142 rad
```

但 waypoint 0 就是当前 home 位置，理论上变化量应为零。

### 11.2 原因

程序先执行：

```python
plan = plan_position_waypoints(
    planning_scene,
    cartesian_waypoints,
)
```

规划完成后，`planning_scene` 已经停在最后一个路点的关节姿态。

示例随后错误地使用：

```python
planning_scene.data.qpos[:7]
```

作为 waypoint 0 的 previous joint position，实际比较变成了：

```text
第一个路点关节角 - 最后一个路点关节角
```

### 11.3 修复

改为从规划结果的第一组关节目标开始：

```python
previous_joint_positions = (
    joint_targets[0].copy()
)
```

修复后：

```text
waypoint 0 joint change norm = 0.000000e+00 rad
```

### 11.4 得到的经验

调用会修改对象状态的函数后，不能假定对象仍处于调用前状态。

需要明确区分：

```text
输入时的场景状态
规划结束后的场景状态
保存到结果对象中的历史数据
```

该问题没有影响 IK 结果和动力学轨迹，只影响示例中第一个变化量的显示基准，现已解决。

---

## 12. 今日必须掌握的内容

### 12.1 路点的含义

路点是希望末端经过的一组离散空间位置。路点本身描述几何路径，不包含电机如何运动的信息。

### 12.2 为什么连续求 IK

冗余机械臂的逆解不唯一。后一个路点从前一个路点的关节姿态开始求解，更容易得到连续的关节目标。

### 12.3 关节直线不等于末端直线

即使关节角在两个目标之间线性变化，末端位置是关节角的非线性函数：

```text
p = f(q)
```

因此：

```text
q线性变化
```

通常不严格意味着：

```text
p线性变化
```

本次路点较密、距离较短，因此几何偏差只有约 `0.039 mm`。

### 12.4 三种场景的职责

```text
planning_scene  → 求关节目标
reference_scene → 测量关节命令的纯运动学结果
control_scene   → 执行动力学并测量实际结果
```

### 12.5 三类误差

```text
几何误差 = 理想直线 - 关节命令的运动学位置
动力学误差 = 关节命令的运动学位置 - 动力学实际位置
总误差 = 理想直线 - 动力学实际位置
```

当前实验中，动力学误差是主导项。

### 12.6 运动误差和最终误差不同

```text
运动中最大误差：约1.254 mm
稳定后最终误差：约0.039 mm
```

“最终能到达”不代表“运动过程中跟踪准确”。评价轨迹控制必须记录整个运动过程，而不能只看终点。

---

## 13. 当前局限

1. 路点只包含位置，没有末端姿态；
2. 每段使用固定时间，没有速度和加速度约束；
3. 关节插值是分段线性的，段连接处速度命令可能不连续；
4. 当前使用理想偏置力补偿；
5. 没有闭环笛卡尔误差修正；
6. 没有碰撞检测和避障；
7. 没有对不同路点数量、速度和阻尼做系统对比；
8. 生成的图像和数据位于被忽略的 `outputs/`，尚未选择作品集图片进入文档目录。

---

## 14. 下一步

下一阶段可以从以下方向推进：

### 14.1 速度对比实验

改变 `TICKS_PER_SEGMENT`，比较：

```text
更慢运动 → 动力学跟踪误差是否减小
更快运动 → 跟踪误差和速度是否增大
```

这可以验证当前1.2 mm误差确实与动态滞后相关。

### 14.2 路点密度对比实验

改变 `WAYPOINT_COUNT`，比较几何误差：

```text
路点更密 → 关节插值形成的末端路径是否更接近直线
路点更稀 → 几何偏差是否增大
```

### 14.3 末端姿态控制

将 Day 9 的旋转 Jacobian 加入任务，进一步实现位置加姿态的6D IK。

### 14.4 抓取动作路径

把直线路径用于：

```text
预抓取点
→ 垂直下降
→ 闭合夹爪
→ 垂直抬升
```

---

## 15. 可复现命令

从仓库根目录执行：

```powershell
& "C:\Users\29391\.conda\envs\mujoco\python.exe" examples\day12\cartesian_waypoint_probe.py
```

```powershell
& "C:\Users\29391\.conda\envs\mujoco\python.exe" examples\day12\cartesian_waypoint_ik_probe.py
```

```powershell
& "C:\Users\29391\.conda\envs\mujoco\python.exe" examples\day12\cartesian_trajectory_tracking_probe.py
```

```powershell
& "C:\Users\29391\.conda\envs\mujoco\python.exe" figures\gen_fig_cartesian_trajectory.py
```

```powershell
& "C:\Users\29391\.conda\envs\mujoco\python.exe" -m pytest tests\test_cartesian_trajectory.py -v
```

```powershell
& "C:\Users\29391\.conda\envs\mujoco\python.exe" -m pytest -q
```

---

## 16. 原始数据与产出

### 16.1 进入版本控制的源码

```text
src/panda_mujoco/cartesian_trajectory.py
tests/test_cartesian_trajectory.py
examples/day12/cartesian_waypoint_probe.py
examples/day12/cartesian_waypoint_ik_probe.py
examples/day12/cartesian_trajectory_tracking_probe.py
figures/gen_fig_cartesian_trajectory.py
docs/day12_log.md
```

### 16.2 本地生成且被 Git 忽略的产出

```text
outputs/day12/cartesian_trajectory_data.npz
outputs/day12/cartesian_trajectory_tracking.png
outputs/day12/cartesian_trajectory_tracking.pdf
```

### 16.3 依赖更新

```text
pyproject.toml
environment.yml
```

加入：

```text
matplotlib>=3.8,<4.0
```

---

## 17. 实验结论

Day 12 成功将单个末端目标扩展为连续笛卡尔路点，并完成了从几何路径、连续 IK、关节命令到动力学实际轨迹的完整实验。

核心结论为：

1. 6个等间距路点能够描述一条20 mm、-10 mm、15 mm方向的空间直线路径；
2. 从上一个 IK 解继续求解下一个路点，可以得到连续关节目标；
3. 当前短距离、密路点设置下，关节插值的最大几何误差只有约 `0.039 mm`；
4. 运动中的最大动力学误差约为 `1.231 mm`；
5. 总运动误差约为 `1.254 mm`，主要由动力学滞后贡献；
6. 保持最终目标1秒后，误差回落到约 `0.039 mm`；
7. 只检查终点不足以评价轨迹控制，必须记录运动全过程；
8. 正式模块和新增测试已完成，全项目共 `41 passed`。

这为下一步研究速度、路点密度以及末端姿态控制提供了可复现基线。
