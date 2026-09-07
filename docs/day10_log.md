# Day 10：伪逆、阻尼最小二乘与迭代位置 IK

## 1. 今日目标

Day 9 已经能够读取和验证 Panda 末端 Jacobian。Day 10 在此基础上解决逆问题：已知希望末端移动的方向和距离，计算七个机械臂关节应当如何变化。

今天完成了以下任务：

- 理解冗余机械臂中伪逆的最小范数解；
- 理解奇异值较小时普通伪逆可能产生很大关节更新；
- 实现阻尼最小二乘（Damped Least Squares，DLS）；
- 在 Panda 的真实位置 Jacobian 上验证一次 DLS 更新；
- 通过反复重算 Jacobian 实现迭代位置 IK；
- 将演示脚本中的算法重构为可复用接口；
- 为 DLS 和位置 IK 增加自动化测试。

## 2. 从正向关系到逆向问题

Day 9 使用的局部运动关系为：

```text
delta_p ≈ Jp(q) @ delta_q
```

其中：

- `delta_p` 是末端位置变化，形状为 `(3,)`，单位为米；
- `Jp(q)` 是当前姿态下的位置 Jacobian，形状为 `(3, 7)`；
- `delta_q` 是七个关节的角度变化，形状为 `(7,)`，单位为弧度。

Day 10 要解决的是相反方向的问题：给定末端位置误差 `e`，求一个合适的 `delta_q`：

```text
Jp(q) @ delta_q ≈ e
```

Panda 有 7 个手臂关节，而位置任务只有 x、y、z 三个维度。因此这是一个冗余问题，通常不存在唯一的关节解。

## 3. 伪逆直觉实验

文件：`examples/day10/pseudoinverse_intuition.py`

使用简单 Jacobian：

```text
J = [[1, 1]]
```

希望任务空间移动：

```text
e = [0.1]
```

所有满足下式的关节更新都能完成任务：

```text
delta_q1 + delta_q2 = 0.1
```

例如：

```text
[0.10,  0.00]
[0.05,  0.05]
[0.20, -0.10]
```

伪逆得到：

```text
delta_q = [0.05, 0.05]
```

它是所有精确完成任务的候选解中二范数最小的解。

实验也验证了零空间方向：

```text
n = [1, -1]
J @ n = 0
```

因此在一个解上加上零空间分量，不会改变当前线性模型中的任务结果：

```text
delta_q_alternative = delta_q + t * n
```

## 4. 为什么使用阻尼最小二乘

文件：`examples/day10/damped_least_squares_intuition.py`

实验 Jacobian 为：

```text
J = [[1.00, 0.00],
     [0.00, 0.02]]
```

第二个奇异值只有 `0.02`。为了产生 `0.01` 的第二方向位移，普通伪逆会给出：

```text
delta_q2 = 0.01 / 0.02 = 0.5
```

这说明当某个方向的运动能力很弱时，伪逆可能放大关节更新。

DLS 使用：

```text
delta_q = J.T @ inv(J @ J.T + lambda^2 I) @ e
```

代码没有显式计算矩阵逆，而是先求解线性方程：

```text
(J @ J.T + lambda^2 I) @ y = e
delta_q = J.T @ y
```

这样数值上更稳定。

实验结果：

| damping | 关节更新范数 | 达到的第二方向位移 | 剩余误差 |
|---:|---:|---:|---:|
| 0.01 | 0.400000 | 0.008000 | 0.002000 |
| 0.05 | 0.068966 | 0.001379 | 0.008621 |
| 0.10 | 0.019231 | 0.000385 | 0.009615 |

阻尼越大，关节更新越保守，但单次更新留下的任务误差也越大。因此阻尼是在稳定性和收敛速度之间进行权衡。

如果奇异值严格等于零，NumPy 的伪逆不会真的计算无穷大，而是将低于阈值的奇异值方向截断。DLS 在该方向上的增益也为零，因此无法创造机械臂本身不具备的瞬时运动能力。

## 5. Panda 单次 DLS 更新

文件：`examples/day10/panda_dls_step_probe.py`

目标偏移设置为：

```text
[+0.020, -0.010, +0.015] m
```

初始误差范数：

```text
2.692582e-02 m
```

在 home 姿态计算 DLS 更新后：

```text
delta_q = [
    -0.007215152,
     0.073852117,
    -0.007215152,
     0.095867035,
     0.000599191,
     0.031468953,
     0.000000000,
]
```

Jacobian 预测的末端位移：

```text
[ 0.018461531, -0.009973812, 0.014250200] m
```

正运动学得到的实际位移：

```text
[ 0.017734908, -0.009971019, 0.013287533] m
```

一次更新后的最终误差范数：

```text
2.839723e-03 m
```

误差减少 `89.45%`，但没有一次完全到达目标。这是因为 Jacobian 只描述当前姿态附近的一阶局部关系。

## 6. 迭代位置 IK

文件：

- `src/panda_mujoco/ik.py`
- `examples/day10/panda_iterative_ik_probe.py`

迭代流程为：

```text
读取当前末端位置
→ 计算 target - current
→ 计算当前姿态下的 Jacobian
→ 使用 DLS 求 delta_q
→ 限制单次关节变化
→ 限制最终关节角范围
→ 写入 qpos
→ 调用 mj_forward
→ 重新检查误差
```

每次更新后必须重新计算 Jacobian，因为 Jacobian 是 `J(q)`，它取决于当前关节姿态。

实验收敛过程：

```text
iteration=0 | error=2.692582e-02 m
iteration=1 | error=2.839723e-03 m
iteration=2 | error=2.463747e-04 m
iteration=3 | error=2.087908e-05 m
```

最终结果：

```text
updates used:    3
final error:     2.087908e-05 m
error reduction: 99.92%
success:         True
```

最终误差约为 `0.0209 mm`，小于 `0.1 mm` 的收敛容差。

## 7. 两种限制的区别

### 7.1 单步限制

```python
delta_q = np.clip(
    raw_delta_q,
    -max_joint_step,
    max_joint_step,
)
```

`max_joint_step` 限制每一轮中单个关节最多改变多少，防止一次移动过大，使局部线性近似失效。

### 7.2 关节限位

```python
limited_joint_positions = np.clip(
    candidate_joint_positions,
    lower_limits,
    upper_limits,
)
```

关节限位保证更新后的绝对关节角符合 MJCF 模型中的物理范围。

两者分别解决：

- 单步限制：这一轮是否走得太远；
- 关节限位：最终姿态是否合法。

## 8. 软件结构重构

正式接口为：

```python
solve_position_ik(
    scene,
    target_position,
    damping=0.05,
    tolerance=1e-4,
    max_iterations=50,
    max_joint_step=0.1,
)
```

职责划分：

```text
src/panda_mujoco/ik.py
├── damped_least_squares()   计算一次 DLS 关节更新
└── solve_position_ik()      管理完整迭代过程

examples/day10/
├── pseudoinverse_intuition.py
├── damped_least_squares_intuition.py
├── panda_dls_step_probe.py
└── panda_iterative_ik_probe.py
```

`PositionIKResult` 用于返回：

- 是否成功；
- 更新次数；
- 最终末端位置；
- 最终误差向量；
- 最终误差范数。

## 9. 当前 IK 不等于电机控制

当前求解器执行：

```python
scene.data.qpos[...] = new_joint_positions
mujoco.mj_forward(scene.model, scene.data)
```

这是直接修改状态并重新计算正运动学：

- 没有推进仿真时间；
- 没有通过执行器产生力；
- 没有模拟速度、惯性和动力学过程。

真正的仿真控制需要将目标关节角写入 `data.ctrl`，然后反复调用 `mujoco.mj_step()`，使 `qpos` 和 `qvel` 随时间演化。

因此当前模块属于“目标末端位置到目标关节角”的 IK 层。后续还需要连接关节控制器和 MuJoCo 动力学层。

## 10. 自动化测试

文件：`tests/test_ik.py`

测试覆盖：

- DLS 与可手算答案一致；
- 零任务误差产生零关节更新；
- 非法阻尼被拒绝；
- 错误的数组形状被拒绝；
- Panda 单次 DLS 更新能够减小位置误差；
- 迭代 IK 能够到达可达目标；
- 初始位置已经等于目标时使用零次更新；
- IK 接口拒绝非法目标和配置参数。

Day 10 完成时的全项目测试结果：

```text
32 passed
```

## 11. 今日结论

1. 位置 Jacobian 将关节微小变化映射为末端位置微小变化。
2. Panda 对三维位置任务是冗余机械臂，逆解不唯一。
3. 伪逆选择最小二范数解，但在弱运动方向附近可能产生较大关节更新。
4. DLS 通过阻尼换取更好的数值稳定性。
5. Jacobian 是当前姿态下的局部模型，因此迭代过程中必须重新计算。
6. 单步限制与关节限位分别保证局部近似稳定和关节姿态合法。
7. 当前 IK 直接修改 `qpos`，尚未模拟执行器驱动的真实运动。

## 12. 与后续工作的关系

下一步将把 IK 求出的目标关节角交给关节位置控制器，通过 `data.ctrl` 和 `mujoco.mj_step()` 让 Panda 在仿真时间中实际运动。这会形成完整链路：

```text
目标末端位置
→ 位置 IK
→ 目标关节角
→ 关节控制器
→ MuJoCo 动力学
→ 实际末端运动
```
