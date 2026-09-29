---
experiment_id: PANDA-MJ-SIM-260922-001
date: 2026-09-22
project: panda-mujoco-manipulation
day: 23
topic: 7D incremental Cartesian action and continuous Gymnasium environment
platform: Windows
python: 3.10.20
mujoco: 3.12.0
gymnasium: 1.3.0
seed: 42
action_space: "Box(-1.0, 1.0, (7,), float32)"
observation_space: "Box(-inf, inf, (29,), float64)"
frame_skip: 10
simulation_timestep_s: 0.002
control_period_s: 0.02
action_mapping_tests: "16 passed"
continuous_env_tests: "4 passed"
full_tests: "187 passed"
env_checker: passed_with_observation_bound_warnings
result: passed_with_known_limitations
---

# Day 23 日志：7 维增量动作与连续控制环境

## 1. 今天解决了什么

Day 22 的 `PandaScriptedPickEnv` 使用 `Discrete(2)`：动作 `0` 等待，动作 `1` 运行一整套脚本抓取。它适合检查 Gymnasium 接口与已有抓取任务的衔接，但不能在每个时间步细粒度地指定末端动作。

Day 23 **保留 Day 22 环境**，另建 `PandaContinuousEnv`。新环境接受七维归一化增量动作：末端位置 3 维、末端姿态 3 维、夹爪开口 1 维。动作先改变**目标**，经 IK 得到关节目标，再由 MuJoCo 动力学推进实际机器人；这不是直接修改 `qpos`，也不是让智能体或强化学习算法已经学会抓取。

今天的验收范围是动作含义、坐标系、缩放与裁剪、环境接口、控制周期、短时跟踪以及可视化。当前奖励和 episode 结束条件尚未设计完成，不能把新环境视为完整的抓取训练任务。

## 2. 文件与职责

| 文件 | 本日职责 |
| --- | --- |
| `src/panda_mujoco/action_mapping.py` | 校验、裁剪七维动作，并转换为位置、旋转和夹爪宽度目标。 |
| `src/panda_mujoco/continuous_env.py` | 实现连续动作环境的 `reset()`、`step()`、空间声明和跟踪误差 `info`。 |
| `tests/test_action_mapping.py` | 检验零动作、缩放裁剪、非法输入及六轴正负方向。 |
| `tests/test_continuous_env.py` | 检验 100 次零动作稳定性、目标保持、姿态/夹爪目标与误差收敛。 |
| `examples/day23/action_ik_probe.py` | 区分“规划得到目标”和“物理运动到目标”。 |
| `examples/day23/action_axis_sign_probe.py` | 打印世界坐标系六个运动分量的正负方向。 |
| `examples/day23/zero_action_stability_probe.py` | 观察从初始状态执行 100 次零动作后的漂移。 |
| `examples/day23/continuous_action_visual_demo.py` | 在 MuJoCo Viewer 中依次演示平移、保持、旋转、保持、夹爪闭合、保持。 |

## 3. 七维动作究竟表示什么

动作空间是 `Box(-1, 1, (7,), float32)`。输入若有 NaN、Inf 或维度不是 7，会被拒绝；有限但超出 `[-1, 1]` 的分量会被裁剪。动作不是关节角、力矩或速度，而是**对上一次目标值的增量**。

| 下标 | 动作分量 | `+1` 的目标变化 | 坐标系／单位 |
| --- | --- | --- | --- |
| 0、1、2 | `dx, dy, dz` | 对应轴增加 0.002 m | 世界坐标系的位置，m |
| 3、4、5 | `dRx, dRy, dRz` | 对应轴增加 2° | 世界坐标系的旋转向量，rad 在代码内计算 |
| 6 | `gripper` | 总开口增加 0.002 m | 夹爪总开口，m；`-1` 表示闭合 2 mm |

位置更新可写为 `p_target_next = p_target + 0.002 × action[:3]`。姿态使用旋转向量生成 `R_delta`，再计算 `R_target_next = R_delta @ R_target`；增量矩阵放在左侧，表明旋转轴在**世界坐标系**中解释。夹爪开口更新后限制在 `[0, 0.08] m`。

这里的 `p_target` 是上一次**目标位置**，不是每次都重新读取实际位置。因此连续 20 次 `dy=+1` 会把目标累计移动 40 mm，即使实际机器人仍在追赶。`dy=0` 则表示**不再改变目标**，不是断电、停止物理仿真或把速度设为零。夹爪自身的局部 Y 轴、相机画面的左右方向、Panda 的 `joint2` 均不能与世界 +Y 混为一谈。

`ee_center_site` 是固定在手部的末端参考点；我们控制的是它的目标位姿。给它一个世界 +Y 位置目标，不保证实际夹爪在一个控制周期内精确平移 2 mm，也不保证两个控制时刻之间的真实轨迹严格是一条直线。若姿态和开口保持不变且位姿完美跟踪，夹爪刚体才会近似作世界 +Y 平移。

## 4. 环境的一次 `reset()` 与 `step()`

### `reset(seed)`

1. 调用 Gymnasium 的 `super().reset(seed=seed)`，让随机数生成器受到 seed 管理。
2. 复用 `PandaPickTask.reset()`，包括已有的随机方块复位与落地稳定过程。
3. 从实际场景读取末端位置、姿态和夹爪宽度，作为新的目标起点；并把 `episode_steps` 清零。
4. 返回长度为 29 的 `float64` 观测及 `info`。`seed=42` 时，复位后读取的夹爪总开口约为 `0.079998211 m`，与理想 0.08 m 的细小差异是物理状态，不应强行改写为 0.08。

本日沿用的 29 维观测由 7 维机械臂关节位置、7 维关节速度、1 维实际夹爪开口、3 维末端世界位置、3 维方块世界位置、2 维方块 yaw 的正余弦、3 维方块线速度、3 维方块角速度组成。`reset()` 和 `step()` 返回的都是**实际仿真状态**，不等于内部目标。

### `step(action)`

1. 校验并裁剪动作，把它叠加到当前目标位姿和开口上。
2. 调用 `plan_pose_target()` 求解该目标位姿的 IK。规划不改变控制场景的 `qpos`、`ctrl` 或仿真时间。
3. 若 IK 成功，发送关节位置命令和夹爪目标，并保存新目标；若 IK 失败，保留上一条控制目标，但仍推进仿真，并通过 `info` 报告失败。
4. 每个动作执行 `frame_skip=10` 个物理步；每步先应用手臂 bias compensation，再调用 `mujoco.mj_step()`。
5. 读取新观测，统计 `episode_steps`，并报告本次 IK 状态、位置误差、姿态误差、夹爪宽度误差等。

模型时间步为 0.002 s，所以一个环境动作推进 `10 × 0.002 = 0.02 s` **仿真时间**，相当于每秒 50 次环境动作的模拟控制频率。这不保证计算机墙钟恰好每 0.02 s 完成一次调用；可视化脚本中的 `time.sleep()` 只负责放慢画面，不推进动力学。

目前 `step()` 返回的奖励固定为 `0.0`，`terminated=False`、`truncated=False` 也只是 Day 23 的占位行为。它们尚未定义抓取成功、失败或时间上限。

## 5. 手动探针与数据

### 5.1 映射、方向与 IK/动力学分离

- 裁剪例子：原始动作 `[1.5, -0.5, 0, 0, 0, 0, -2]` 变成 `[1, -0.5, 0, 0, 0, 0, -1]`；原数组不应被就地修改。
- 位置分量 `[1, -0.5, 0]` 对应世界位移 `[2, -1, 0] mm`；旋转 Z 分量 `0.5` 对应 1°；夹爪 `-1` 使 40 mm 目标开口变成 38 mm。
- 六轴正负方向探针显示：`+x` 为 `[2, 0, 0] mm`，`-y` 为 `[0, -2, 0] mm`，`+Rz` 的世界旋转向量为 `[0, 0, 2]°`，反向动作符号相反；未操作的夹爪分量保持原宽度。
- 单个 +x 动作要求目标移动 2 mm。IK 在 1 次更新后找到关节目标，规划位置误差约 `0.013406 mm`；规划前后的控制场景 `qpos`、`ctrl`、仿真时间均不变。随后执行 10 个物理步（0.02 s），实际末端仅移动约 `[0.260666, 0.005914, 0.113563] mm`，目标位置误差约 `1.743047 mm`。保持同一控制命令再执行 100 个物理步，误差约为 `0.216791 mm`，总仿真时间 0.22 s。这证明 `IK success=True` 不等于当下的物理跟踪完成。

### 5.2 零动作不是“什么都不做”

从初始位姿连续调用 100 次全零动作：

```text
episode_steps:                100
target unchanged:            True
end-effector drift:          8.296056388398787e-11 mm
observation finite:          True
action duration:             2.0 s simulated
```

漂移远低于 0.1 mm 的测试阈值，可视为浮点数值误差。这里测量的 2.0 s 从 `reset()` 完成后开始，不包含复位阶段方块落地的时间。

另一实验先给一次 +x 动作，目标增加 2 mm；随后连续给 10 次零动作。目标保持不变，但位置跟踪误差从约 `1.743047 mm` 降至约 `0.210539 mm`。原因是 `step(0向量)` 仍继续给控制器维持目标，并执行物理步；实际机械臂有时间追赶目标。

### 5.3 Viewer 中的平移、旋转、夹爪响应

以下数值来自 `seed=42` 的 `continuous_action_visual_demo.py`。表中“位置误差”是末端参考点到目标位置的欧氏距离；姿态误差用目标姿态相对实际姿态的旋转角计算；开口误差是目标总开口与实际总开口之差的绝对值。

| 阶段 | 目标与实际的关键状态 | 位置误差 | 姿态误差 | 夹爪开口误差 |
| --- | --- | ---: | ---: | ---: |
| `MOVE +Y`，20 次 | 目标 +Y 40 mm；实际 +Y 31.016 mm | 8.984134 mm | 0.030996° | 0.002778 mm |
| `HOLD position`，50 次 | 目标仍 +Y 40 mm；实际 +Y 39.914 mm | 0.085843 mm | 0.003930° | 0.000068 mm |
| `ROTATE +WORLD Z`，10 次 | 目标绕世界 Z 累计 +20°；实际相对初始姿态的旋转向量 Z 分量约 +10.907° | 4.811094 mm | 9.257176° | 0.000686 mm |
| `HOLD rotation`，50 次 | 目标仍 +20°；实际旋转向量 Z 分量约 +19.982° | 0.001123 mm | 0.122604° | 0.000180 mm |
| `CLOSE GRIPPER`，20 次 | 目标总开口 39.998 mm；实际总开口 50.725 mm | 0.001116 mm | 0.122130° | 10.726864 mm |
| `HOLD gripper width`，50 次 | 目标总开口不变；实际总开口 39.999 mm | 0.001119 mm | 0.122033° | 0.000473 mm |

旋转阶段虽然**位置目标不变**，实际末端仍暂时偏离约 4.8 mm：关节为追赶新的姿态目标而运动，位置与姿态在动力学过程中存在瞬态耦合。目前只给一系列离散位姿目标和关节位置命令，并未强制真实运动路径始终贴着某条笛卡尔直线。保持目标后误差显著下降。

打印的“target/actual rotation vector”均是**相对初始姿态**的表示，既不是欧拉角，也不能简单逐分量相减当作姿态误差。日志表里的姿态误差来自 `rotation_error(target_rotation, actual_rotation)` 的向量模长；内部单位 rad，展示时转换为度。

Viewer 每阶段打印的 `IK success` 来自**该阶段最后一次** `env.step()` 的 `info`，不能仅凭这一行断言该阶段所有动作都成功。它更不能代表抓取成功；本演示中夹爪在空中开合，没有抓取方块。

## 6. 自动化验收

```powershell
python -m pytest tests/test_action_mapping.py -v
python -m pytest tests/test_continuous_env.py -v
python -m pytest -q
python examples/day23/continuous_action_visual_demo.py
```

在激活项目 `mujoco` 环境、从仓库根目录执行的本地结果：

| 检查 | 结果 | 验证要点 |
| --- | --- | --- |
| `tests/test_action_mapping.py` | 16 passed | 零动作、裁剪与拒绝非法值、六轴各正负方向；非单位初始姿态用于验证世界坐标系旋转约定。 |
| `tests/test_continuous_env.py` | 4 passed | 100 次零动作不漂移、一次 +x 后保持目标并减小误差、姿态/夹爪目标更新、保持后误差减小。 |
| 完整测试集 | 187 passed | 本日新增代码未使已有测试失败。 |
| Gymnasium `check_env(..., skip_render_check=True)` | PASSED | 有上下界为 `-inf`/`inf` 的观测空间建议性警告；不是接口失败。 |
| MuJoCo Viewer 演示 | 已完成 | 观察了位置、姿态和夹爪目标变化与实际响应的时间差。 |

普通的“运行 Python 文件”只会定义 `tests/test_continuous_env.py` 中的测试函数，不会自行调用它们；要用 `python -m pytest ...`，或 VS Code 的“Run Test”。pytest 测试本身是无窗口的；动画来自 `examples/day23/continuous_action_visual_demo.py`。

## 7. 遇到的问题与定位方式

1. 初次导入连续环境时出现 `ModuleNotFoundError`，后来又出现继承自 `gymnasium.Env.step()` 的 `NotImplementedError`。原因是项目根目录与 `src/panda_mujoco/` 曾同时存在名为 `continuous_env.py` 的文件；编辑了前者，Python 实际导入了后者。可用 `inspect.getfile(PandaContinuousEnv)` 核对实际加载路径。
2. 给 Viewer 增加 `orientation_error_rad` 输出后出现 `KeyError`，也是同一原因：新 `info` 字段写入根目录同名文件，但包内文件仍是旧版。现已把变更同步到 `src/panda_mujoco/continuous_env.py`，并通过测试。以后只维护包内文件，避免再引入根目录同名文件。
3. 视觉上难以判断 +Y 对应屏幕哪边：Viewer 相机从斜角观察，屏幕方向不等于世界坐标方向。判断动作坐标系应读 `ee_center_site` 的世界位置、代码中的动作映射及打印的 `[x, y, z]`，不能只凭画面方向。

## 8. 当前结论与边界

**已完成**：一个可复位、可调用 `step()` 的七维增量末端控制原型；动作单位与世界坐标系约定明确；位置、姿态、夹爪目标能经 IK/控制器进入 MuJoCo 动力学；零动作与短时跟踪、三类误差监测、动画和自动化测试均有实测证据。

**尚未完成**：

- 奖励、成功/失败终止、时间上限等任务语义仍是占位值；暂不能用于严肃的 episode 评估或训练。
- 连续环境没有自己的 `render()` 实现；Day 23 通过独立的 Viewer 演示脚本可视化。
- 观测空间仍声明无穷上下界，未做逐维物理范围约束或归一化。
- `step()` 没有保证末端严格走直线，也没有约束动态跟踪误差始终小于指定阈值；大一些或更快的动作在旋转阶段已显示明显瞬态偏差。
- IK 失败分支虽在代码中保留上一控制目标并报告原因，但本日测试未覆盖多样化的不可达目标与长期失败序列。
- 尚无智能体策略或学习过程；本环境只是为后续控制或学习实验准备可调用的接口。

下一步继续按第四周计划补全连续环境的任务判定与评估，并在涉及抓取时把“规划成功”“跟踪到位”“接触成立”“抬升保持”分开检查。日志本身暂不代表已经提交或上传 GitHub；按本周统一上传的约定处理。
