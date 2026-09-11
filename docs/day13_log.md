---
exp_id: PANDA-SIM-260911-001
date: 2026-09-11
system: Panda MuJoCo Manipulation
exp_type: 6D pose inverse kinematics and dynamics tracking
scene: assets/robots/panda/scene_with_cube.xml
python: 3.10.20
mujoco: 3.12.0
numpy: 2.2.6
scipy: 1.15.3
status: passed
test_count: 56
anomaly: false
tags: [Panda, MuJoCo, rotation error, angular Jacobian, 6D IK, DLS, pose tracking]
---

# Day 13：旋转误差与6D末端位姿控制

## 1. 本日目标

Day 10 实现了只控制末端位置的迭代 DLS IK，Day 11 将位置 IK 的关节目标接入 MuJoCo 动力学，Day 12 将单个位置目标扩展为连续笛卡尔路点。Day 13 在此基础上加入末端姿态控制。

本日完成以下任务：

1. 使用旋转矩阵和旋转向量定义姿态误差；
2. 区分世界坐标轴旋转和末端局部坐标轴旋转；
3. 验证只使用旋转 Jacobian 的姿态 DLS 更新；
4. 将位置和姿态组成6维任务误差；
5. 实现迭代 6D 位姿 IK；
6. 将 6D IK 整理为正式的 `solve_pose_ik()` 接口；
7. 通过位置执行器和偏置力补偿执行目标位姿；
8. 对世界坐标系三个旋转轴进行动力学验收；
9. 添加旋转工具测试和 6D IK 测试。

---

## 2. 姿态误差的定义

### 2.1 为什么不能直接相减旋转矩阵

末端当前姿态和目标姿态分别记为：

```text
R_current
R_target
```

旋转矩阵的每个元素描述坐标轴方向。直接计算：

```python
R_target - R_current
```

得到的只是矩阵元素差，不是一个具有明确旋转轴和旋转角意义的姿态误差。

本项目使用相对旋转：

```text
R_error = R_target @ R_current.T
```

旋转矩阵满足：

```text
R^-1 = R.T
```

因此该公式先撤销当前姿态，再得到从当前姿态旋转到目标姿态所需的相对旋转。

### 2.2 旋转向量

将 `R_error` 转换成旋转向量：

```python
Rotation.from_matrix(R_error).as_rotvec()
```

旋转向量可写为：

```text
e_r = theta * u
```

其中：

- `u` 是单位旋转轴；
- `theta` 是旋转角，单位为弧度；
- 向量方向表示旋转方向；
- 向量模长表示还需旋转的角度。

例如绕世界 `+z` 轴旋转15度：

```text
rotation_error = [0, 0, 0.26179939]
```

因为：

```text
15 deg = 0.26179939 rad
```

---

## 3. 世界轴与局部轴

在 Panda home 姿态下，末端局部 `z` 轴在世界坐标系中的方向约为：

```text
[0.989993, 0, -0.141114]
```

左乘增量旋转：

```python
target_rotation = delta_rotation @ current_rotation
```

表示绕固定的世界坐标轴旋转。

右乘增量旋转：

```python
target_rotation = current_rotation @ delta_rotation
```

表示绕末端局部坐标轴旋转。

绕世界 `z` 轴15度时，误差向量约为：

```text
[0, 0, 0.261799]
```

绕末端局部 `z` 轴15度时，误差向量用世界坐标系表达为：

```text
[0.259180, 0, -0.036943]
```

两者模长都为15度，但旋转轴不同。

---

## 4. 正式旋转误差接口

新增：

```text
src/panda_mujoco/rotations.py
```

核心接口：

```python
rotation_error(target_rotation, current_rotation)
```

返回世界坐标系下的三维旋转误差向量。

模块还检查旋转矩阵是否满足：

1. 形状为 `(3, 3)`；
2. 所有元素有限；
3. `R.T @ R` 约等于单位矩阵；
4. 行列式约等于 `+1`。

其中第4项可以排除行列式为 `-1` 的镜像矩阵。

新增10项旋转测试：

- 相同姿态产生零误差；
- 世界 `x/y/z` 轴正负小角度；
- 局部轴误差正确转换到世界坐标系；
- 错误形状、NaN、非正交矩阵和镜像矩阵被拒绝。

---

## 5. 只控制姿态的对照实验

首先只使用旋转 Jacobian：

```text
J_r * delta_q = e_r
```

实验目标为保持初始位置，同时让末端绕世界 `+z` 轴旋转15度。但是求解任务中并未真正加入位置约束。

单次 DLS 更新结果：

```text
Initial orientation error: 15.000000 degrees
Final orientation error:    0.018683 degrees
Orientation linearization error: 8.996986e-06 rad
End-effector position drift: 0.178175 m
```

姿态误差几乎被消除，但末端位置漂移约：

```text
178.175 mm
```

原因是 `J_r` 只约束末端角运动。DLS 求解器没有收到保持位置的任务，因此不会自动保持末端位置。

这证明：

> 一个解在已定义任务上最优，不代表它会自动满足没有写入任务的要求。

---

## 6. 6D位姿任务

完整任务使用位置 Jacobian 和旋转 Jacobian：

```text
J_pose = [J_p]
         [J_r]
```

其形状为：

```text
(6, 7)
```

任务误差为：

```text
e_pose = [e_p]
         [e_r]
```

其中：

- `e_p` 是三维世界坐标位置误差，单位为米；
- `e_r` 是三维世界坐标旋转误差，单位为弧度。

一次 DLS 更新仍使用：

```text
delta_q = J.T @ inv(J @ J.T + lambda^2 I) @ e
```

### 6.1 单步结果

保持目标位置不变，姿态绕世界 `+z` 轴旋转15度。原始 DLS 解中部分腕部关节更新约为 `1.15 rad`，超过单步限制，因此对整个 `delta_q` 统一缩放：

```text
step_scale = 0.086357846
maximum joint update = 0.1 rad
```

单步结果：

```text
Final position error:    0.312954 mm
Initial orientation error: 15.000000 degrees
Final orientation error:   13.711099 degrees
```

和只使用 `J_r` 相比，位置漂移从 `178.175 mm` 降到 `0.313 mm`。由于单步被限制，姿态需要多次迭代才能到达。

### 6.2 为什么统一缩放

本实验使用：

```python
delta_q = raw_delta_q * step_scale
```

所有关节按同一比例缩小，因此 DLS 解的方向保持不变。

如果分别使用 `np.clip()` 截断不同关节，超过限值的分量会被单独改变，原更新向量的方向以及位置与姿态之间的协调关系都会发生变化。

---

## 7. 迭代6D位姿IK

每次迭代都执行：

```text
读取当前位姿
→ 计算位置误差和旋转误差
→ 重新计算当前6D Jacobian
→ DLS求关节更新
→ 统一缩放到最大单步0.1 rad
→ 检查关节限位
→ 更新qpos并调用mj_forward
```

绕世界 `+z` 轴15度的迭代记录：

| 迭代 | 位置误差 (mm) | 姿态误差 (deg) |
|---:|---:|---:|
| 0 | 0.000000 | 15.000000 |
| 1 | 0.312954 | 13.711099 |
| 2 | 0.629805 | 12.396366 |
| 3 | 0.950774 | 11.023527 |
| 4 | 1.273292 | 9.556979 |
| 5 | 1.589989 | 7.954173 |
| 6 | 1.881457 | 6.159773 |
| 7 | 2.091170 | 4.094300 |
| 8 | 2.014321 | 1.628913 |
| 9 | 0.523151 | 0.113292 |
| 10 | 0.003842 | 0.000122 |

位置误差在前几步暂时增大，但完整任务最终同时收敛。这说明单个误差分量不一定每一步单调下降，应以联合任务的最终收敛和安全约束判断结果。

---

## 8. 正式6D IK接口

在以下文件中新增：

```text
src/panda_mujoco/ik.py
```

正式接口：

```python
solve_pose_ik(
    scene,
    target_position,
    target_rotation,
)
```

返回 `PoseIKResult`：

```text
success
iterations
final_position
final_rotation
final_position_error
final_orientation_error
final_position_error_norm
final_orientation_error_norm
```

默认成功条件为：

```text
position error < 1e-4 m
orientation error < 1e-3 rad
```

两个条件必须同时满足。

正式接口在世界 `+z 15 deg` 目标上的结果：

```text
success: True
iterations: 10
position error: 0.003842 mm
orientation error: 0.000122 deg
```

---

## 9. 动力学执行

运动学求解和动力学执行使用两个独立场景：

```text
planning_scene
    → solve_pose_ik()
    → 得到目标关节角

control_scene
    → JointPath平滑插值
    → 写入data.ctrl
    → 每步更新qfrc_bias补偿
    → mujoco.mj_step()
```

这样可以确认机械臂不是通过直接改写 `qpos` 瞬移，而是通过执行器真正到达目标。

单个世界 `+z 15 deg` 动力学结果：

```text
IK updates:              10
Simulation time:         5.000 s
Maximum joint error:     9.532904e-07 rad
Position error:          3.868388e-06 m
Orientation error:       1.346963e-04 degrees
Joint velocity norm:     1.383679e-05 rad/s
```

---

## 10. 多轴动力学验收

三个案例均从独立的 home 场景开始：

| 目标 | IK更新 | 位置误差 (mm) | 姿态误差 (deg) | 最大关节误差 (rad) | 速度范数 (rad/s) |
|---|---:|---:|---:|---:|---:|
| world +x 15 deg | 3 | 0.000173 | 0.000019 | 1.834e-07 | 2.181e-06 |
| world -y 15 deg | 5 | 0.003419 | 0.000012 | 4.407e-07 | 4.570e-06 |
| world +z 15 deg | 10 | 0.003868 | 0.000135 | 9.533e-07 | 1.384e-05 |

汇总：

```text
maximum position error:    0.003868 mm
maximum orientation error: 0.000135 degrees
maximum joint error:       9.532904e-07 rad
maximum velocity norm:     1.383679e-05 rad/s
```

三个方向需要的迭代数不同，说明机械臂在当前姿态下不同任务方向的可操作性不同。世界 `+z` 旋转需要更明显的腕部关节协调，因此受到最大单步限制的影响更大。

路线表要求：

```text
姿态误差 <= 5 deg
位置误差 <= 20 mm
```

当前自动验收采用更严格的门槛：

```text
姿态误差 < 1e-3 rad，约0.057 deg
位置误差 < 1e-4 m，即0.1 mm
```

全部案例通过。

---

## 11. 自动测试

`tests/test_ik.py` 新增5项 6D IK 测试：

1. 当前位姿就是目标时使用零次更新；
2. 保持位置并完成世界 `+z 15 deg` 旋转；
3. 位置和姿态同时改变时共同收敛；
4. 迭代次数不足时返回 `success=False`；
5. 非法目标和非法参数被拒绝。

测试结果：

```text
tests/test_ik.py: 13 passed
full project: 56 passed
```

---

## 12. 本日文件

正式模块：

```text
src/panda_mujoco/rotations.py
src/panda_mujoco/ik.py
```

自动测试：

```text
tests/test_rotations.py
tests/test_ik.py
```

实验脚本：

```text
examples/day13/rotation_error_intuition.py
examples/day13/panda_rotation_error_probe.py
examples/day13/orientation_only_dls_probe.py
examples/day13/pose_dls_step_probe.py
examples/day13/panda_iterative_pose_ik_probe.py
examples/day13/pose_ik_dynamics_probe.py
examples/day13/pose_ik_multiaxis_acceptance.py
```

---

## 13. 当前局限

1. 当前只验证了15度小角度目标；
2. 位置误差使用米，姿态误差使用弧度，目前采用相同数值权重；
3. DLS 阻尼固定为常数，没有根据奇异值自适应调整；
4. 关节限位使用最终裁剪，没有零空间避限位策略；
5. 规划阶段仍直接修改独立场景的 `qpos`；
6. 动力学阶段执行预先计算的关节目标，不是实时笛卡尔反馈控制；
7. 偏置力补偿依赖 MuJoCo 精确模型；
8. 尚未加入碰撞约束、姿态轨迹插值和角速度限制；
9. 没有测试接近180度时旋转向量的表示不连续问题。

---

## 14. 需要掌握的内容

完成 Day13 后，应能解释：

1. 为什么旋转矩阵不能直接相减作为姿态误差；
2. `R_target @ R_current.T` 的含义；
3. 旋转向量的方向和模长分别表示什么；
4. 左乘和右乘增量旋转的坐标系区别；
5. `Jr` 的形状为什么是 `(3, 7)`；
6. 完整位姿 Jacobian 为什么是 `(6, 7)`；
7. 为什么只控制姿态会造成位置漂移；
8. 为什么位置和姿态必须分别设置成功容差；
9. 为什么每轮 IK 都要重新计算 Jacobian；
10. 为什么规划场景和动力学控制场景需要分开。

---

## 15. 可复现实验命令

```powershell
& "C:\Users\29391\.conda\envs\mujoco\python.exe" examples\day13\rotation_error_intuition.py
& "C:\Users\29391\.conda\envs\mujoco\python.exe" examples\day13\panda_rotation_error_probe.py
& "C:\Users\29391\.conda\envs\mujoco\python.exe" examples\day13\orientation_only_dls_probe.py
& "C:\Users\29391\.conda\envs\mujoco\python.exe" examples\day13\pose_dls_step_probe.py
& "C:\Users\29391\.conda\envs\mujoco\python.exe" examples\day13\panda_iterative_pose_ik_probe.py
& "C:\Users\29391\.conda\envs\mujoco\python.exe" examples\day13\pose_ik_dynamics_probe.py
& "C:\Users\29391\.conda\envs\mujoco\python.exe" examples\day13\pose_ik_multiaxis_acceptance.py
& "C:\Users\29391\.conda\envs\mujoco\python.exe" -m pytest -q
```

---

## 16. 本日结论

Day13 完成了从3D位置 IK 到6D位姿 IK 的升级。旋转误差和旋转 Jacobian 已接入 DLS 求解，位置与姿态能够同时收敛，并且目标关节角能够通过 MuJoCo 执行器稳定执行。

最关键的实验对照为：

```text
只控制姿态：位置漂移178.175 mm
完整6D控制：最大稳态位置误差0.003868 mm
```

这说明机器人控制器只会优化显式写入任务的量。要在旋转末端时保持工具中心点的位置，必须把位置误差和姿态误差共同放入任务空间。

该接口已经能够为后续抓取任务提供固定末端方向的接近、下降和抬升目标。
