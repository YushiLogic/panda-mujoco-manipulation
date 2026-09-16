---
exp_id: PM-B-260916-001
date: 2026-09-16
project: panda-mujoco-manipulation
day: 16
exp_type: 方块复位与可复现随机化
platform: Windows
python: 3.10.20
mujoco: 3.12.0
random_seed_examples: [42, 43, 44]
batch_seed_range: "0-19"
batch_cases: 20
full_test_result: "89 passed in 5.48s"
anomaly: false
tags: [MuJoCo, Panda, free-joint, quaternion, reset, seed, randomization]
---

# Day 16 日志：方块复位与可复现随机化

## 1. 实验目的

Day15 已经完成固定方块位置下的抓取几何设计，并验证了 `pregrasp`、`grasp` 和 `lift` 三个目标位姿都存在可行的 6D IK 解。

Day16 的目标是让方块不再只能使用固定位置，而是建立一套可复现、可检查的随机复位机制：

1. 根据名称查询方块 free joint 的状态地址；
2. 理解 free joint 在 `qpos` 和 `qvel` 中的表示；
3. 将随机 yaw 转换为 MuJoCo 使用的 `wxyz` 四元数；
4. 使用 seed 生成可复现的方块位置和朝向；
5. 将目标位姿写入 MuJoCo，并清除上一轮速度；
6. 推进动力学，使悬空方块自然落地并稳定；
7. 通过动画、单次探针、20-seed 批量探针和 pytest 验证行为。

Day16 构成后续多次抓取实验的 reset 基础。如果 reset 不可靠，后续抓取成功率就无法公平比较，也无法复现失败案例。

---

## 2. 与 Day15 的关系

Day15 的输入是一个方块位置：

```text
cube_position
    ↓
generate_grasp_targets()
    ↓
pregrasp / grasp / lift
```

Day16 解决的是这个方块位置从哪里来：

```text
seed
    ↓
sample_cube_pose()
    ↓
reset_cube()
    ↓
settle_cube()
    ↓
稳定的cube_position
    ↓
Day15的抓取目标生成
```

因此，Day15 定义“抓取目标”，Day16 定义“每一轮实验开始时方块在哪里”。

---

## 3. MuJoCo 中的 joint ID 与状态地址

### 3.1 当前关节顺序

当前模型中的关节顺序为：

```text
joint id 0：joint1
joint id 1：joint2
joint id 2：joint3
joint id 3：joint4
joint id 4：joint5
joint id 5：joint6
joint id 6：joint7
joint id 7：finger_joint1
joint id 8：finger_joint2
joint id 9：cube_joint
```

`cube_joint` 不是第8个关节，因为七个机械臂关节后还有两个手指关节。MuJoCo 的 ID 从0开始，所以 ID 9 表示第10个关节。

### 3.2 三个数值为9并不代表同一概念

模型查询结果为：

```text
joint id:     9
qpos address: 9
qvel address: 9
```

它们的含义分别是：

- `joint id`：关节在模型关节表中的编号；
- `qpos address`：该关节状态在 `data.qpos` 中的起始下标；
- `qvel address`：该关节速度在 `data.qvel` 中的起始下标。

当前三者恰好都是9，是因为 `cube_joint` 之前的9个关节全部是单自由度关节，每个关节各占一个 `qpos` 和一个 `qvel`。

如果前面出现 ball joint 或 free joint，一个 joint 会占用多个状态槽，joint ID 和状态地址就不再相同。因此不能把 joint ID 当作数组地址使用。

### 3.3 为什么通过名称查询

Day16 使用：

```python
joint_id = model.joint("cube_joint").id
qpos_address = model.jnt_qposadr[joint_id]
qvel_address = model.jnt_dofadr[joint_id]
```

而不是直接写：

```python
data.qpos[9:16]
data.qvel[9:15]
```

通过名称查询可以降低模型结构变化后代码静默写错位置的风险。

---

## 4. free joint 的状态表示

方块可以在三维空间中平移和旋转，因此使用 MuJoCo 的 free joint。

### 4.1 qpos：7维

```text
[px, py, pz, qw, qx, qy, qz]
```

其中：

- `px, py, pz`：方块中心的世界位置，单位为米；
- `qw, qx, qy, qz`：方块姿态的单位四元数。

当前 home 位姿为：

```text
[0.45, 0.00, 0.05, 1.0, 0.0, 0.0, 0.0]
```

含义是：方块中心位于 `[0.45, 0.00, 0.05] m`，局部坐标轴与世界坐标轴对齐。

### 4.2 qvel：6维

```text
[vx, vy, vz, wx, wy, wz]
```

其中：

- 前三项是线速度，单位为 `m/s`；
- 后三项是角速度，单位为 `rad/s`。

reset 时必须将这六项全部置零，否则上一轮实验留下的移动或旋转会污染下一轮。

`qpos` 使用四个数表示旋转，但角速度仍然只有三个自由度，因此 free joint 的 `qvel` 是6维而不是7维。

---

## 5. 四元数理解

### 5.1 MuJoCo 的顺序

MuJoCo 使用：

```text
[w, x, y, z]
```

为了避免和位置坐标混淆，更明确地写作：

```text
[qw, qx, qy, qz]
```

完整 free-joint 状态中的前一个 `z` 是位置高度 `pz`，最后一个 `z` 是四元数分量 `qz`，二者没有直接关系。

### 5.2 w 与向量部分表示什么

单位四元数可以写成：

$$
q=w+q_xi+q_yj+q_zk
$$

绕单位轴

$$
\mathbf a=[a_x,a_y,a_z]
$$

旋转角度 $\theta$ 时：

$$
q=
\left[
\cos\frac{\theta}{2},
a_x\sin\frac{\theta}{2},
a_y\sin\frac{\theta}{2},
a_z\sin\frac{\theta}{2}
\right]
$$

其中：

- `w` 是标量部分，编码旋转角的半角余弦；
- `[qx,qy,qz]` 是向量部分，方向与旋转轴一致，长度为半角正弦。

### 5.3 为什么绕 z 轴时 w 和 qz 有值

绕世界 $z$ 轴旋转时：

$$
\mathbf a=[0,0,1]
$$

因此：

$$
q=
\left[
\cos\frac{\theta}{2},
0,
0,
\sin\frac{\theta}{2}
\right]
$$

`qz` 有值，正是因为旋转轴是 $z$ 轴。`qz` 不表示方块沿世界 $z$ 方向移动，也不表示高度发生变化。

例如绕 $z$ 轴旋转90°：

```text
qw = cos(45°) = 0.70710678
qx = 0
qy = 0
qz = sin(45°) = 0.70710678
```

对应四元数：

```text
[0.70710678, 0, 0, 0.70710678]
```

它对应的旋转矩阵为：

$$
R_z(90^\circ)=
\begin{bmatrix}
0&-1&0\\
1&0&0\\
0&0&1
\end{bmatrix}
$$

此时局部 $x$ 轴转到世界 $+y$，局部 $y$ 轴转到世界 $-x$，局部 $z$ 轴仍与世界 $z$ 轴重合。

### 5.4 本项目中的起始姿态

本项目将方块保持直立，只随机改变 yaw：

```text
yaw = 0°  → [1, 0, 0, 0]
yaw = 90° → [0.7071, 0, 0, 0.7071]
```

这表示相对于世界单位姿态的绝对 yaw，而不是在任意已有姿态上继续旋转。如果以后需要表达“在当前姿态上再旋转”，需要进行四元数乘法。

还需注意：SciPy 的 `Rotation.as_quat()` 默认使用 `xyzw` 顺序，不能未经转换直接写入 MuJoCo。

---

## 6. 可复现随机采样

### 6.1 采样范围

Day16 使用较保守的抓取区域：

```text
x：   0.42 ～ 0.48 m
y：  -0.06 ～ 0.06 m
yaw： -30° ～ +30°
z：   固定为0.05 m
```

这个范围足以验证随机化与 reset，同时避免在抓取状态机尚未完成前引入过大的可达性和接触难度。

### 6.2 seed 的作用

采样器使用：

```python
rng = np.random.default_rng(seed)
```

实验结果验证：

```text
same seed identical: True
different seed different: True
```

这意味着：

- 相同 seed 会生成逐位相同的位置和四元数；
- 不同 seed 会生成不同实验条件；
- 如果未来某个抓取 episode 失败，只要保存 seed，就可以重新生成相同方块位姿。

这里使用局部随机数生成器，不修改 NumPy 的全局随机状态，避免对其他模块产生隐式影响。

---

## 7. reset 与 settle 的职责分离

### 7.1 reset_cube()

`reset_cube()` 负责瞬时状态设置：

```text
验证输入
    ↓
查询cube_joint地址
    ↓
写入xyz和wxyz
    ↓
清零六维速度
    ↓
mj_forward更新派生数据
```

它不会：

- 推进仿真时间；
- 让方块产生自由落体；
- 修改机械臂关节位置；
- 修改执行器 `ctrl`。

### 7.2 settle_cube()

`settle_cube()` 负责动力学落稳：

```text
对机械臂应用bias补偿
    ↓
mj_step推进一个物理步
    ↓
重复到固定仿真时长
    ↓
读取线速度与角速度
    ↓
判断是否满足稳定阈值
```

默认阈值为：

```text
线速度 ≤ 1e-4 m/s
角速度 ≤ 1e-3 rad/s
```

固定落稳时间为1.5秒。

### 7.3 为什么是750个 mj_step

当前物理时间步为：

```text
0.002 s
```

所以1.5秒对应：

$$
1.5/0.002=750
$$

750不是方块落地所必需的固定步数。它只是“1.5秒固定观察窗口”在当前时间步下对应的步数。

方块底面初始离地高度约为：

$$
0.05-0.02=0.03\text{ m}
$$

理想自由落体时间约为：

$$
t=\sqrt{\frac{2h}{g}}
=\sqrt{\frac{2\times0.03}{9.81}}
\approx0.078\text{ s}
$$

即大约39个物理步就会首次接触地面。剩余时间用于等待碰撞、轻微反弹和转动衰减。

当前采用固定时长策略是为了保持 Day16 实现简单、确定。更高效的策略需要同时判断“已经接触地面”和“速度连续一段时间低于阈值”，否则方块刚生成时虽然速度为零，却会被误判为已经稳定。

---

## 8. mj_forward、mj_step、viewer.sync 与 sleep

四个操作的职责不同：

### `mujoco.mj_forward()`

- 根据当前 `qpos`、`qvel` 重新计算 body、site、contact 等派生量；
- 不推进仿真时间；
- 用于 reset 写入状态之后立即刷新读数。

### `mujoco.mj_step()`

- 推进一个动力学时间步；
- 计算重力、碰撞、摩擦和执行器作用；
- 会改变时间、位置和速度。

### `viewer.sync()`

- 将最新仿真状态刷新到可视化窗口；
- 不产生物理运动；
- 不代替 `mj_step()`。

### `time.sleep()`

- 只限制电脑的墙钟执行速度；
- 让动画大致按照真实时间播放；
- 不改变 MuJoCo 的物理规律。

无窗口探针没有 `sleep`，因此可以在不到现实30秒的时间内计算20个各1.5秒的物理实验。

---

## 9. 可视化实验

`examples/day16/cube_reset_visual_demo.py` 依次演示 seed 42、43、44：

```text
显示悬空初始状态1秒
    ↓
开始推进动力学
    ↓
观察方块下落和碰撞1.5秒
    ↓
显示落稳结果1秒
```

已完整观察 seed 42 和43，结果为：

| seed | 初始位置 | yaw | 最终高度 | 最终6D速度范数 |
|---:|---|---:|---:|---:|
| 42 | `[0.46643736, -0.00733459, 0.05]` | 21.516° | 0.0199276 m | `1.82e-08` |
| 43 | `[0.45913796, -0.05474696, 0.05]` | -28.798° | 0.0199276 m | `5.03e-08` |

seed 44 动画开始后由用户主动关闭 viewer，因此没有最终打印；这是预期的安全退出行为，不记为实验异常。

可视化确认：

- 方块出现在合理的桌面区域；
- 不同 seed 产生不同位置和 yaw；
- 方块从 `z=0.05 m` 自然下落；
- 与地面发生正常碰撞；
- 最终没有弹飞或穿透地面；
- 机械臂在 bias 补偿下保持基本静止。

---

## 10. 单次 reset 与落稳探针

seed 42 的 reset 瞬间结果为：

```text
position:   [ 0.46643736, -0.00733459, 0.05 ]
yaw:         21.515875°
quaternion: [ 0.98242455,  0, 0, 0.18666014 ]
qvel:       [ 0, 0, 0, 0, 0, 0 ]
time:        0.0 s
```

落稳结果为：

```text
settled:             True
steps:               750
simulation duration: 1.500000000000001 s
final position:      [0.46643737, -0.00733459, 0.01992760]
linear speed:        5.10e-16 m/s
angular speed:       1.82e-08 rad/s
xy drift:            9.81e-09 m
```

`1.500000000000001` 与理论1.5秒的差异是二进制浮点累加误差，不代表有意义的额外仿真时间。

探针中的 `.copy()` 用于保存 reset 瞬间的快照。如果不复制，NumPy 切片可能只是 `data.qpos` 的视图，后续 `mj_step()` 会让所谓的“旧快照”跟着改变。

---

## 11. 20-seed 批量实验

批量探针复用同一个 `PandaScene`，对 seed 0～19 依次执行：

```text
reset_to_home()
    ↓
sample_cube_pose(seed)
    ↓
reset_cube()
    ↓
检查初始速度与接触
    ↓
settle_cube(1.5 s)
    ↓
检查高度、速度、漂移和地面接触
```

故意复用同一个 scene，可以更容易发现上一轮速度、姿态或接触状态污染下一轮的问题。

所有20个案例均满足：

- 初始方块—地面接触数为0；
- 落稳后方块—地面接触数为4；
- 最终高度约为 `0.0199276 m`；
- `settled=True`；
- 无 NaN 或 Inf；
- 采样位置与 yaw 均在规定范围内。

汇总结果：

```text
cases:                     20
maximum height error:      7.240217e-05 m
maximum xy drift:          1.584601e-07 m
maximum linear speed:      1.471220e-15 m/s
maximum angular speed:     2.103200e-07 rad/s
```

单位换算：

```text
最大高度误差 = 0.0724 mm
最大水平漂移 = 0.000158 mm
```

落稳后的4个接触不是接触了4个物体，而是同一个方块底部与同一地面之间形成的4个接触点。

结果：

```text
Twenty-seed reset and settling: PASSED
```

---

## 12. 自动化测试

`tests/test_cube_reset.py` 共生成21个 pytest 案例，覆盖：

| 测试组 | 数量 | 内容 |
|---|---:|---|
| 模型地址 | 1 | `cube_joint` 的 qpos/qvel 地址 |
| 正常四元数 | 2 | 0°与90° yaw |
| 非法 yaw | 3 | NaN、正无穷、负无穷 |
| 随机采样 | 3 | 同 seed、不同 seed、20个范围检查 |
| 正常 reset | 1 | 写入、速度清零、无副作用 |
| 非法位姿 | 6 | 形状、有限性、单位四元数、yaw |
| 正常落稳 | 1 | 高度、速度和有限性 |
| 非法落稳参数 | 4 | 时间和速度阈值检查 |

Day16 测试结果：

```text
21 passed in 1.08s
```

其中非法输入测试显示 `PASSED`，表示函数按照预期拒绝输入并抛出 `ValueError`，不是表示非法输入被接受。

完整项目回归测试结果：

```text
89 passed in 5.48s
```

无失败、跳过或警告。新增模块没有破坏已有的机械臂控制、夹爪、Jacobian、IK、笛卡尔轨迹、旋转计算和抓取目标生成。

---

## 13. 今日实现的公共接口

### `get_cube_joint_addresses(model)`

根据名称查询 free joint 的 qpos 和 qvel 地址，避免在新代码中硬编码状态下标。

### `yaw_to_quaternion(yaw)`

将绕世界 $z$ 轴的 yaw 转换成 MuJoCo `wxyz` 单位四元数，并拒绝 NaN/Inf。

### `sample_cube_pose(seed)`

使用局部 NumPy 随机数生成器采样 `x、y、yaw`，返回 `CubePose`。

### `reset_cube(scene, pose)`

验证位姿、写入 free-joint qpos、清零六维速度并调用 `mj_forward()`，不推进时间。

### `settle_cube(scene, ...)`

推进固定时长动力学，计算最终位置、线速度和角速度，返回 `CubeSettleResult`。

---

## 14. 文件清单

```text
src/panda_mujoco/cube_reset.py
    方块地址、四元数、采样、reset与settle公共接口

examples/day16/cube_reset_visual_demo.py
    可视化随机复位、自由落体、碰撞与落稳

examples/day16/cube_reset_probe.py
    单次检查、seed复现检查和20-seed批量验收

tests/test_cube_reset.py
    21项自动化回归测试

docs/day16_log.md
    Day16理论、实现、实验结果与验收记录
```

---

## 15. 当前限制

1. `settle_cube()` 使用固定1.5秒窗口，没有根据连续稳定时间提前停止；
2. 当前只随机化方块 `x、y、yaw`，保持 roll、pitch 为0；
3. 当前采样范围是为早期抓取实验选择的保守范围；
4. seed 保证采样结果可复现，但完整抓取轨迹的严格复现还需要状态机和完整环境 reset；
5. 当前只验证方块稳定落地，没有执行机械臂接近、闭合夹爪或抬升；
6. 当前地面接触检查在探针中实现，后续会整理为可复用的接触状态接口。

这些限制与第三周计划一致，将在后续运动原语、接触判定和抓取状态机中逐步处理。

---

## 16. Day16 需要掌握的内容

完成 Day16 后，应能解释：

1. joint ID、qpos address 和 qvel address 的区别；
2. 为什么 `cube_joint` 的 ID 是9；
3. free joint 为什么使用7维 qpos 和6维 qvel；
4. MuJoCo 四元数为什么使用 `wxyz`；
5. 为什么绕 $z$ 轴旋转时 `w` 和 `qz` 有值；
6. 为什么四元数的 `qz` 与方块高度无关；
7. 相同 seed 为什么可以复现相同随机位姿；
8. 为什么 reset 必须清除线速度和角速度；
9. `mj_forward()` 与 `mj_step()` 的区别；
10. `viewer.sync()` 与 `time.sleep()` 各自解决什么问题；
11. 为什么首次接触地面不等于已经稳定；
12. 为什么750步只是固定1.5秒窗口，而不是落地所需步数；
13. 为什么测试需要先故意污染速度再执行 reset；
14. 为什么保存状态快照时需要 `.copy()`；
15. 为什么探针和 pytest 都有价值，但承担不同职责。

---

## 17. Day16 验收结论

- [x] free joint 地址按名称查询；
- [x] free-joint qpos/qvel 结构已验证；
- [x] yaw 到 `wxyz` 四元数转换正确；
- [x] 相同 seed 逐位一致；
- [x] 不同 seed 产生不同位姿；
- [x] 20个 seed 全部位于规定范围；
- [x] reset 后方块六维速度严格清零；
- [x] reset 不修改机械臂状态和控制命令；
- [x] reset 不推进仿真时间；
- [x] reset 后没有提前接触地面；
- [x] 落稳后形成稳定地面接触；
- [x] 20个案例全部达到稳定速度阈值；
- [x] 最大高度误差为0.0724 mm；
- [x] 最大水平漂移为0.000158 mm；
- [x] 可视化演示正常；
- [x] Day16 的21项测试全部通过；
- [x] 项目完整89项测试全部通过。

**Day16：PASSED**

---

## 18. 下一步

Day17 将实现抓取运动原语，把已经具备的能力组合起来：

```text
稳定方块位置
    ↓
生成pregrasp和grasp目标
    ↓
规划6D IK目标
    ↓
通过动力学轨迹移动到方块上方
    ↓
沿接近方向下降
```

重点将从“方块如何开始一轮实验”转向“机械臂如何可靠地执行一个抓取阶段”。
