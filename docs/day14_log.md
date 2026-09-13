---
exp_id: PANDA-SIM-260913-001
date: 2026-09-13
system: Panda MuJoCo Manipulation
exp_type: deterministic batch reach evaluation
scene: assets/robots/panda/scene_with_cube.xml
python: 3.10.20
mujoco: 3.12.0
numpy: 2.2.6
scipy: 1.15.3
random_seed: 20260913
target_count: 50
status: passed
test_count: 56
anomaly: false
tags: [Panda, MuJoCo, IK, dynamics, evaluation, reproducibility, CSV]
---

# Day 14：固定随机种子的批量末端到达评测

## 1. 本日目标

Day 10 至 Day 13 已经完成位置 IK、动力学关节跟踪、笛卡尔路点和 6D 位姿 IK。此前实验主要验证一个或少数几个手工选择的目标，能够说明功能可用，但不足以回答以下工程问题：

1. 算法面对一批不同目标时是否仍然稳定；
2. 成功率是多少，而不是只展示一个成功案例；
3. 是否出现 NaN、Inf 或关节越界；
4. IK 的数学解能否通过执行器和动力学真正到达；
5. 实验结果能否被其他人重复得到；
6. 单个案例的原始数据能否保存下来供后续分析。

因此，Day 14 的核心不是再增加一种控制算法，而是建立一套可复现、可统计、可追踪的批量评测流程。

本日完成以下任务：

1. 使用固定随机种子生成 50 个末端目标位置；
2. 验证相同种子产生完全相同的目标集合；
3. 使用前 5 个目标执行纯运动学预检；
4. 使用前 5 个目标执行动力学预检；
5. 对全部 50 个目标执行 IK 与动力学跟踪；
6. 对每个案例记录结构化指标和失败原因；
7. 将逐案例结果写入 CSV；
8. 计算成功率、误差、迭代次数、耗时和安全性指标；
9. 运行完整项目回归测试。

---

## 2. 为什么使用固定随机种子

目标由 NumPy 随机数生成器产生：

```python
rng = np.random.default_rng(20260913)
```

随机种子固定后，同一个生成算法会得到同一组目标。这意味着：

- 修改 IK 后可以在完全相同的 50 个目标上重新测试；
- 两次实验的结果具有可比性；
- 其他开发者能够复现目标集合；
- 某个案例失败时，可以用 `case_id` 精确定位并重复运行。

如果每次运行都生成不同目标，那么成功率的变化可能来自目标集合变化，而不一定来自算法变化。

本实验使用：

```text
random seed = 20260913
target count = 50
```

目标相对于 home 末端位置的采样范围为：

```text
x offset: [-0.10, +0.10] m
y offset: [-0.15, +0.15] m
z offset: [-0.20, +0.05] m
```

目标计算公式为：

```text
p_target = p_home + offset
```

采样程序验证了：

```text
Same-seed targets identical: True
Different-seed targets different: True
Deterministic target sampling: PASSED
```

这里的最小、最大样本偏移没有恰好等于采样边界是正常现象。均匀随机采样只保证样本位于范围内部，不保证一定抽中边界。

---

## 3. 三层评测结构

### 3.1 第一层：目标生成检查

文件：

```text
examples/day14/target_sampling_probe.py
```

该程序只验证目标数据本身：

- 数组形状为 `(50, 3)`；
- 所有数值有限；
- 所有偏移位于设定范围；
- 相同种子结果完全相同；
- 不同种子结果发生变化。

这一层不运行 IK，也不推进动力学。

### 3.2 第二层：运动学预检

文件：

```text
examples/day14/kinematic_reach_preflight.py
```

前 5 个目标分别使用一个新的 `PandaScene`，并从相同 home 姿态开始运行位置 IK。

每个案例检查：

- IK 是否收敛；
- 最终位置误差；
- 迭代次数；
- 是否出现 NaN 或 Inf；
- 是否违反关节限位；
- 失败原因；
- 求解耗时。

预检结果：

```text
success count: 5/5
maximum error: 0.030058 mm
Kinematic reach preflight: PASSED
```

这一层直接修改规划场景的 `qpos` 并调用正运动学，因此验证的是“数学求解结果”，不代表执行器已经让机器人真实运动到该姿态。

### 3.3 第三层：动力学预检与正式评测

文件：

```text
examples/day14/dynamic_reach_preflight.py
examples/day14/evaluate_reach.py
```

每个目标使用两个独立场景：

```text
planning_scene
    -> 运行位置 IK
    -> 得到 7 维目标关节角

control_scene
    -> 从 home 开始
    -> 通过位置执行器跟踪关节轨迹
    -> 每步施加 qfrc_bias 补偿
    -> 测量实际末端位置
```

这样可以明确区分：

- `IK error`：规划场景中的纯运动学误差；
- `tracking error`：控制场景经过 MuJoCo 动力学后的实际误差。

如果 IK 误差很小而跟踪误差很大，问题通常位于控制器、轨迹时间、执行器参数或动力学补偿，而不是 IK 数学求解本身。

---

## 4. 动力学执行流程

每个目标都从 home 姿态独立开始，避免前一个案例的终点影响下一个案例。

一次动力学执行包括：

1. 在 `planning_scene` 中求解目标关节角；
2. 在 `control_scene` 中建立 home 到目标关节角的插值轨迹；
3. 使用 50 Hz 控制频率更新位置执行器目标；
4. 使用 500 Hz 物理频率推进 MuJoCo；
5. 运动 2 秒；
6. 在最终目标保持 1 秒；
7. 测量最终关节误差、速度和末端误差。

对应参数为：

```text
CONTROL_EVERY = 10
MOVE_TICKS = 100
SETTLE_STEPS = 500
simulation time per case = 3.0 s
```

每个物理步之前都会调用：

```python
apply_arm_bias_compensation(scene)
```

原因是 `qfrc_bias` 随当前 `qpos` 和 `qvel` 改变，不能只在运动开始前计算一次。

---

## 5. 成功定义与失败分类

### 5.1 单案例成功条件

一个案例通过需要同时满足：

```text
IK 收敛
and 所有状态均为有限数值
and 没有违反关节限位
and 动力学跟踪误差不超过 0.02 m
```

不能只根据 `ik_result.success` 判断整个机器人任务成功，因为 IK 成功后仍可能出现执行器跟踪误差或动力学异常。

### 5.2 末端误差

目标位置和实际位置分别为：

```text
p_target
p_actual
```

末端位置误差使用欧氏距离：

```text
e = p_target - p_actual
tracking_error = ||e||_2
```

CSV 中的 `tracking_error_m` 单位是米，控制台显示时乘以 1000 转换成毫米。

例如：

```text
tracking_error_m = 2.5613e-05 m
```

等价于：

```text
tracking error = 0.025613 mm
```

### 5.3 失败原因

评测程序使用以下失败原因：

| 失败原因 | 含义 |
|---|---|
| `non_finite_state` | 状态出现 NaN 或 Inf |
| `joint_limit_violation` | 至少一个关节超出限位 |
| `ik_not_converged` | IK 在最大迭代次数内未达到容差 |
| `tracking_error_too_large` | IK 成功，但动力学末端误差超过 2 cm |
| `none` | 案例通过，无失败原因 |

本次输出：

```text
failure reasons: {}
```

空字典表示 50 个案例均未失败，不是漏记。

---

## 6. CSV 数据结构

输出文件：

```text
results/day14/reach_evaluation.csv
```

CSV 的每一行对应一个固定目标。主要字段如下：

| 字段 | 含义 | 单位 |
|---|---|---|
| `case_id` | 案例编号 | 无 |
| `target_x_m` | 目标世界坐标 x | m |
| `target_y_m` | 目标世界坐标 y | m |
| `target_z_m` | 目标世界坐标 z | m |
| `ik_success` | 位置 IK 是否收敛 | bool |
| `success` | 综合验收是否通过 | bool |
| `failure_reason` | 失败原因 | 字符串 |
| `ik_iterations` | IK 更新次数 | 次 |
| `ik_error_m` | 纯运动学末端误差 | m |
| `tracking_error_m` | 动力学末端误差 | m |
| `max_joint_error_rad` | 最大最终关节跟踪误差 | rad |
| `joint_velocity_norm_rad_s` | 最终关节速度范数 | rad/s |
| `all_finite` | 状态是否全部有限 | bool |
| `joint_limit_violation` | 是否发生关节越界 | bool |
| `simulation_time_s` | 单案例仿真时长 | s |
| `wall_time_s` | 单案例真实计算耗时 | s |

保存逐案例数据比只打印一个平均值更有价值，因为后续可以：

- 查找误差最大的目标；
- 分析目标距离和迭代次数的关系；
- 对比不同阻尼系数；
- 画成功率和误差分布图；
- 复现某个失败案例。

---

## 7. 正式 50 点评测结果

运行命令：

```powershell
python examples/day14/evaluate_reach.py
```

结果：

```text
IK success:                 50/50
overall success:            50/50
success rate:               100.00%
mean successful error:      0.027230 mm
maximum successful error:   0.098611 mm
mean IK iterations:         4.440
mean wall time per target:  0.199663 s
total wall time:            9.983144 s
non-finite cases:           0
joint-limit violations:     0
failure reasons:            {}
```

验收标准和结果：

| 指标 | 标准 | 实测 | 结论 |
|---|---:|---:|---|
| 综合成功率 | `>= 90%` | `100%` | 通过 |
| 成功案例平均误差 | `<= 20 mm` | `0.027230 mm` | 通过 |
| 非有限数值案例 | `0` | `0` | 通过 |
| 关节限位违规案例 | `0` | `0` | 通过 |

最大跟踪误差约为 `0.098611 mm`，接近位置 IK 使用的 `0.1 mm` 收敛容差。IK 误差和动力学误差总体接近，说明当前位置执行器、插值轨迹和理想偏置力补偿能够较准确地执行 IK 给出的关节目标。

真实耗时 `wall_time_s` 会受到 CPU、系统负载和首次运行初始化影响，不要求不同计算机逐位一致。固定种子保证的是目标数据一致，而不是运行时间一致。

---

## 8. 回归测试

完整测试命令：

```powershell
python -m pytest -v
```

结果：

```text
56 passed in 3.03s
```

说明 Day14 新增评测脚本没有破坏此前的重置、夹爪、运动学、Jacobian、IK、姿态误差、轨迹和关节控制功能。

---

## 9. 本次结果不能说明什么

本次 `100%` 成功率只适用于当前明确限定的实验条件：

- 固定的 50 个采样目标；
- 目标位于 home 附近的长方体范围；
- 每个案例都从同一个 home 姿态开始；
- 只约束末端位置，没有同时约束姿态；
- 场景中没有进行障碍物规避；
- 规划模型和控制模型完全一致；
- 没有传感器噪声、执行器误差和模型失配；
- 使用理想模型提供的 `qfrc_bias` 补偿；
- 没有验证真实 Panda 机器人。

因此不能将该结果表述为：

```text
Panda 在整个工作空间中具有 100% 到达率。
```

更准确的表述是：

```text
在固定随机种子、指定 home 附近采样范围和当前 MuJoCo 模型条件下，
50 个位置目标均通过 IK 与动力学跟踪验收。
```

---

## 10. 与 Week3 抓取任务的关系

Week3 的抓取状态机可以复用本周形成的完整运动链：

```text
读取方块位置
    -> 生成方块上方目标
    -> 位置或位姿 IK
    -> 生成路点/关节轨迹
    -> 动力学执行
    -> 夹爪闭合
    -> 接触检测
    -> 抬升并判断抓取成功
```

Day14 的批量评测方法也会继续用于抓取任务：不能只展示一次成功抓取，而应固定随机种子生成多组方块初始位置，统计抓取成功率、失败原因和最终高度。

---

## 11. 本日掌握检查

完成 Day14 后，应能够解释：

1. 为什么批量实验要固定随机种子；
2. 为什么每个案例要从相同初始状态开始；
3. IK error 和 tracking error 的区别；
4. 为什么 `ik_success=True` 不等于整个任务成功；
5. 为什么要同时检查 NaN、Inf 和关节限位；
6. 为什么平均误差不能替代逐案例原始数据；
7. CSV 每一行表示什么；
8. `failure reasons: {}` 为什么表示没有失败；
9. 为什么本次 100% 成功率不能外推到整个工作空间和真实机器人；
10. 如何将同一套评测思路迁移到 Week3 抓取成功率测试。

---

## 12. 本日交付物

```text
examples/day14/target_sampling_probe.py
examples/day14/kinematic_reach_preflight.py
examples/day14/dynamic_reach_preflight.py
examples/day14/evaluate_reach.py
results/day14/reach_evaluation.csv
docs/day14_log.md
```

Day14 状态：**PASSED**。
