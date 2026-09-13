# Week 2 验收报告：Panda 末端位姿控制

## 1. 本周目标与最终结论

第二周的目标是从第一周的“能够加载并控制 Panda 各个关节”，推进到“能够在世界坐标系中指定末端目标，并通过逆运动学和动力学控制到达目标”。

本周最终建立了以下闭环工程链路：

```text
目标末端位置/姿态
    -> 计算当前末端位姿
    -> 计算位置和姿态误差
    -> 计算线速度/角速度 Jacobian
    -> DLS 求关节更新
    -> 迭代 IK 得到目标关节角
    -> 生成平滑轨迹
    -> 位置执行器 + 偏置力补偿
    -> MuJoCo 动力学执行
    -> 测量实际末端误差
    -> 批量统计与 CSV 记录
```

Week2 验收结论：

- 位置和旋转 Jacobian 已实现并通过有限差分检查；
- 位置 IK 和 6D 位姿 IK 已实现；
- IK 结果能够通过 MuJoCo 动力学执行；
- 笛卡尔直线路点能够转换成连续关节目标；
- 50 个固定随机位置目标全部通过；
- 平均动力学末端误差为 `0.027230 mm`；
- 最大动力学末端误差为 `0.098611 mm`；
- 无 NaN/Inf；
- 无关节限位违规；
- 完整项目测试为 `56 passed`。

---

## 2. 每日工作脉络

### Day 8：末端位姿和坐标变换

Day8 回答了最基础的问题：机器人末端现在位于哪里、朝向哪里。

完成内容：

- 从 `ee_center_site` 读取世界坐标系位置；
- 从 MuJoCo 的 `xmat` 读取并整理 3×3 旋转矩阵；
- 验证 `R.T @ R = I` 和 `det(R) = 1`；
- 将末端局部点转换到世界坐标系；
- 将世界坐标点逆变换回末端局部坐标系；
- 验证 joint1 绕世界 z 轴旋转对末端位姿的影响。

核心关系：

```text
p_world = R_world_ee @ p_ee + t_world_ee
p_ee = R_world_ee.T @ (p_world - t_world_ee)
```

这为后续定义“世界坐标系中的目标位置和目标姿态”提供了基础。

### Day 9：Jacobian 与有限差分验证

Day9 建立关节运动与末端瞬时运动之间的局部线性关系：

```text
v = Jp(q) qdot
omega = Jr(q) qdot
```

Panda 有 7 个手臂关节，因此：

```text
Jp shape = (3, 7)
Jr shape = (3, 7)
J  shape = (6, 7)
```

Jacobian 每一列表示当前姿态下一个关节单位运动对末端运动的瞬时影响。由于 Jacobian 依赖当前关节姿态，迭代 IK 每一步都必须重新计算 Jacobian。

使用中心有限差分验证位置 Jacobian：

```text
dp/dq_i ≈ [p(q + epsilon e_i) - p(q - epsilon e_i)] / (2 epsilon)
```

解析 Jacobian 与有限差分最大误差约为 `1.312e-10`，验证通过。

### Day 10：阻尼最小二乘与位置 IK

Day10 从 Jacobian 前向关系：

```text
delta_p ≈ Jp delta_q
```

转向逆问题：已知末端误差，求关节更新。

由于 Panda 的位置任务是 3 维、关节是 7 维，Jacobian 不是方阵，不能使用普通矩阵逆。本项目采用阻尼最小二乘：

```text
delta_q = J.T (J J.T + lambda^2 I)^-1 e
```

DLS 通过阻尼降低接近奇异位形时关节更新过大的风险。单次 DLS 只是当前姿态附近的线性近似，因此使用以下循环形成位置 IK：

```text
计算当前位置
    -> 计算误差
    -> 重新计算 Jacobian
    -> DLS 求 delta_q
    -> 限制单步关节变化
    -> 限制关节上下限
    -> mj_forward
    -> 再次计算误差
```

标准位置 IK 接口为：

```python
solve_position_ik(...)
```

### Day 11：从 IK 数学解到动力学运动

Day11 区分了三个容易混淆的量：

- `qpos`：实际广义位置；
- `ctrl`：执行器命令；
- IK target：规划器计算出的目标关节角。

IK 为了快速计算可以直接修改规划场景的 `qpos`，但真实控制场景不能把目标值直接写成实际状态。控制场景通过：

```python
command_arm_joint_positions(...)
mujoco.mj_step(...)
```

让执行器产生运动。

未使用偏置力补偿时，单目标动力学末端误差约为：

```text
9.949521 mm
```

每步应用当前 `qfrc_bias` 之后，误差降低到约：

```text
0.020929 mm
```

本项目当前采用理想模型前馈补偿：

```text
qfrc_applied = qfrc_bias
```

它不是唯一控制策略，后续可以替换为重力补偿、逆动力学、computed torque、操作空间控制或更接近真实机器人的控制器。

### Day 12：笛卡尔路点轨迹

Day12 将 Day11 的单个末端目标扩展为多个等间距笛卡尔路点：

```text
p_i = p_start + alpha_i (p_goal - p_start)
```

每个路点依次运行 IK，并以上一个路点的关节解作为下一个路点的初始姿态，得到连续的关节目标序列。

动力学跟踪结果：

```text
maximum moving tracking error = 1.254162 mm
settled final error = 0.038672 mm
```

这说明运动过程中允许存在有限跟踪滞后，而最终保持阶段能使系统收敛到目标附近。

### Day 13：旋转误差与 6D 位姿 IK

Day13 将任务从 3 维位置扩展为 6 维位姿。

相对旋转定义为：

```text
R_error = R_target R_current.T
```

再将相对旋转矩阵转换为旋转向量：

```text
e_r = theta u
```

6 维误差和 Jacobian 为：

```text
e = [e_position; e_rotation]
J = [J_position; J_rotation]
```

只使用角速度 Jacobian 可以迅速降低姿态误差，但会造成明显位置漂移。本周实验中，orientation-only 更新造成约 `178.175 mm` 的位置漂移。这说明多自由度系统中的“只完成一个子任务”可能牺牲未受约束的方向。

使用完整 6D Jacobian 后，同时控制位置和姿态。绕世界 x、y、z 三个方向 15 度的动力学验收全部通过，其中最大结果为：

```text
maximum position error = 0.003868 mm
maximum orientation error = 0.000135 degrees
maximum joint error = 9.532904e-07 rad
```

标准 6D 位姿 IK 接口为：

```python
solve_pose_ik(...)
```

### Day 14：批量评测与可复现数据

Day14 将“演示一个成功目标”升级为“在固定数据集上报告成功率”。

使用固定种子 `20260913` 在 home 附近生成 50 个目标。每个案例都从相同 home 姿态开始，先在规划场景中求 IK，再在独立控制场景中执行 3 秒动力学运动。

正式结果：

```text
IK success:                 50/50
overall success:            50/50
success rate:               100.00%
mean successful error:      0.027230 mm
maximum successful error:   0.098611 mm
mean IK iterations:         4.440
non-finite cases:           0
joint-limit violations:     0
failure reasons:            {}
```

全部逐案例结果保存到：

```text
results/day14/reach_evaluation.csv
```

---

## 3. 本周形成的代码结构

### 可复用模块

| 文件 | 职责 |
|---|---|
| `src/panda_mujoco/kinematics.py` | 末端位置、旋转和 Jacobian |
| `src/panda_mujoco/ik.py` | DLS、位置 IK、6D 位姿 IK |
| `src/panda_mujoco/rotations.py` | 旋转矩阵验证和旋转误差 |
| `src/panda_mujoco/arm_control.py` | 关节命令和偏置力补偿 |
| `src/panda_mujoco/joint_trajectory.py` | 连续关节目标插值 |
| `src/panda_mujoco/cartesian_trajectory.py` | 笛卡尔路点和顺序 IK |

### 验证层

`examples/day8` 至 `examples/day14` 保存可以独立运行的概念实验和验收脚本。它们不仅展示最终结果，也保留从坐标变换、Jacobian、单步 DLS 到完整 IK 的推导过程。

### 自动测试层

当前测试覆盖：

- 夹爪映射与动力学；
- reset 可重复性；
- Jacobian 形状和有限差分；
- DLS 已知解；
- 位置 IK 和 6D 位姿 IK；
- 旋转矩阵与旋转误差输入检查；
- 笛卡尔路点连续性；
- 关节命令和偏置力补偿。

完整结果：

```text
56 passed in 3.03s
```

---

## 4. Week2 验收表

| 验收项 | 结果 | 状态 |
|---|---:|---|
| 读取末端世界坐标位置和旋转 | 已实现 | 通过 |
| 坐标正变换与逆变换 | 往返误差约 `1e-16 m` | 通过 |
| 位置 Jacobian 有限差分 | 最大误差约 `1.312e-10` | 通过 |
| 位置 IK | 单目标最终误差约 `0.020879 mm` | 通过 |
| 笛卡尔路点动力学 | settled error `0.038672 mm` | 通过 |
| 6D 位姿 IK | 位置 `0.003842 mm`，姿态 `0.000122 deg` | 通过 |
| 多轴 6D 动力学 | x/y/z 三种旋转全部通过 | 通过 |
| 50 点位置到达成功率 | `100%` | 通过 |
| 50 点平均跟踪误差 | `0.027230 mm` | 通过 |
| NaN/Inf | `0` | 通过 |
| 关节限位违规 | `0` | 通过 |
| 完整自动测试 | `56 passed` | 通过 |

---

## 5. 本周需要真正掌握的知识

### 5.1 坐标系

能够解释：

- world frame、base frame 和 end-effector frame；
- 旋转矩阵的列表示局部坐标轴在世界坐标系中的方向；
- 点和方向向量的坐标变换区别；
- 世界轴旋转与局部轴旋转为什么使用不同的乘法顺序。

### 5.2 Jacobian

能够解释：

- 每一列对应一个关节；
- `Jp` 的三行对应世界 x、y、z 线速度；
- `Jr` 的三行对应世界 x、y、z 角速度；
- Jacobian 是当前姿态下的局部线性关系；
- 为什么大幅度运动不能只计算一次 Jacobian；
- 为什么有限差分可以检查解析 Jacobian。

### 5.3 逆运动学

能够解释：

- 7 自由度机械臂为什么不能直接使用普通矩阵逆；
- 伪逆为什么选择最小范数解；
- 零空间运动为什么可以不改变一阶任务结果；
- DLS 中阻尼过小和过大的权衡；
- 单步 DLS 与迭代 IK 的区别；
- 关节单步限幅和关节限位分别解决什么问题。

### 5.4 动力学执行

能够解释：

- 直接写 `qpos` 是设置状态，不是执行器控制；
- `ctrl` 是目标命令，实际状态需要通过 `mj_step` 演化；
- `qfrc_bias` 包含什么类型的偏置项；
- 为什么补偿要在每一个物理步更新；
- 为什么规划场景与控制场景需要分离；
- 为什么运动误差和稳定后误差不同。

### 5.5 工程评测

能够解释：

- 固定随机种子的意义；
- 单案例成功和批量成功率的区别；
- 为什么要保存 CSV 原始数据；
- 为什么要记录失败原因而不是只记录 False；
- 为什么 100% 仿真成功率不能直接外推到真实机器人。

---

## 6. 当前局限性

第二周已经完成可工作的末端控制基线，但仍有明确局限：

1. IK 没有显式优化与关节限位的距离；
2. 没有零空间姿态或 manipulability 优化；
3. 没有碰撞检测和避障规划；
4. 路点使用简单线性插值，没有速度和加速度连续规划；
5. 使用理想 `qfrc_bias`，规划与控制模型完全一致；
6. 批量位置评测不约束末端姿态；
7. 所有案例都从同一个 home 姿态开始；
8. 采样范围只是 home 附近的一个长方体，不是完整工作空间；
9. 尚未加入噪声、延迟、摩擦扰动和模型失配；
10. 尚未接入 ROS 2 或真实机器人。

这些限制不否定当前结果，而是定义了结果适用的边界，并为后续改进提供方向。

---

## 7. 与 Week3 抓取环境的接口

Week3 不需要重新实现末端控制，而是组合已有模块形成任务：

```text
RESET
  -> MOVE_ABOVE
  -> APPROACH
  -> CLOSE_GRIPPER
  -> VERIFY_CONTACT
  -> LIFT
  -> CHECK_SUCCESS
```

可直接复用：

- `get_ee_pose()`：读取末端状态；
- `solve_position_ik()`：移动到方块上方；
- `solve_pose_ik()`：保持合适的夹爪朝向；
- 笛卡尔路点：生成接近和抬升路径；
- `command_arm_joint_positions()`：执行关节目标；
- `apply_arm_bias_compensation()`：维持跟踪精度；
- Day5 夹爪接口：张开与闭合；
- Day5 接触检测：判断左右手指是否接触方块。

Week3 的评测将沿用 Day14 思路：固定随机种子改变方块位置，统计抓取成功率、失败状态、接触情况和抬升高度，而不是只保存一次成功演示。

---

## 8. 面试表述建议

可以将第二周工作概括为：

> 我在 MuJoCo 中为 7 自由度 Panda 实现了基于 Jacobian 和阻尼最小二乘的位置/6D 位姿 IK，并将 IK 关节目标接入位置执行器、关节轨迹与模型偏置力补偿。为了避免只展示单一成功案例，我使用固定随机种子建立了 50 点动力学到达评测，记录逐案例 CSV、失败原因、数值稳定性和关节限位，当前限定采样范围内成功率为 100%，平均末端误差约 0.027 mm。该结果是在理想仿真模型和固定初始状态下获得的，尚未包含碰撞规避和 sim-to-real 误差。

这段表述包含：

- 做了什么；
- 使用了什么方法；
- 如何验证；
- 得到了什么量化结果；
- 对结果边界是否有清醒认识。

---

## 9. Week2 交付物

```text
src/panda_mujoco/kinematics.py
src/panda_mujoco/ik.py
src/panda_mujoco/rotations.py
src/panda_mujoco/arm_control.py
src/panda_mujoco/cartesian_trajectory.py

examples/day8/
examples/day9/
examples/day10/
examples/day11/
examples/day12/
examples/day13/
examples/day14/

tests/test_kinematics.py
tests/test_ik.py
tests/test_rotations.py
tests/test_arm_control.py
tests/test_cartesian_trajectory.py

results/day14/reach_evaluation.csv
docs/day8_log.md
docs/day9_log.md
docs/day10_log.md
docs/day11_log.md
docs/day12_log.md
docs/day13_log.md
docs/day14_log.md
docs/week2_report.md
```

Week2 状态：**PASSED**。
