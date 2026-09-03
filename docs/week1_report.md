# Week 1 Report：Panda MuJoCo 项目基础能力

## 1. 本周目标

第一周的目标不是直接实现抓取算法，而是搭建一个可靠、可复现、可继续扩展的 Panda MuJoCo 项目基础。

本周主要完成：

1. 建立可安装的 Python 项目；
2. 集成 Franka Emika Panda MuJoCo 模型；
3. 实现可靠的 home 状态和复位功能；
4. 实现关节空间轨迹控制；
5. 实现夹爪宽度控制；
6. 理解并验证方块、地面和夹爪接触；
7. 将模型资产放入仓库，使项目不再依赖外部路径；
8. 编写自动测试和第一周集成验收脚本。

## 2. 当前环境

- Operating system：Windows
- Python：3.10.20
- MuJoCo：3.12.0
- NumPy：2.2.6
- SciPy：1.15.3
- Test framework：pytest 9.1.1
- Robot：Franka Emika Panda
- Simulation scene：Panda + floor + free cube

项目采用 editable install：

```powershell
python -m pip install -e .
```

因此可以在示例、测试和后续控制程序中统一导入：

```python
from panda_mujoco.simulation import PandaScene
```

## 3. 本周完成内容

### Day 1：项目与环境初始化

完成 Python 项目基本结构：

```text
assets/
configs/
docs/
examples/
src/panda_mujoco/
tests/
```

建立：

- `pyproject.toml`
- `environment.yml`
- `.gitignore`
- `README.md`
- Git 仓库

完成 MuJoCo 最小环境测试，验证模型能够加载、仿真能够推进、物体能够在重力作用下落到地面。

### Day 2：Panda 模型审计

检查 Panda XML 模型中的：

- body
- joint
- actuator
- site
- tendon
- equality constraint
- joint range
- qpos 和 ctrl 维度

确认末端参考点为：

```text
ee_center_site
```

理解了 `MjModel` 保存静态模型结构，`MjData` 保存运行时状态。

### Day 3：可靠复位与 home 状态

实现：

```python
PandaScene.reset_to_home()
```

统一管理：

- Panda 机械臂初始关节角；
- 两根手指的初始位置；
- 方块的初始位置和姿态；
- 关节速度清零；
- 执行器控制目标；
- 状态合法性检查。

特别处理了 joint4：

```text
joint4 home = -1.57079 rad
```

避免使用超出 joint4 限位的默认零位。

### Day 4：关节空间轨迹

实现 `JointPath`，使用线性插值生成连续关节目标：

```text
q_target = q_start + (q_end - q_start) × s
```

理解了：

- 物理仿真步；
- 控制拍；
- `data.ctrl` 是控制目标；
- `data.qpos` 是实际状态；
- 控制命令不会瞬间改变实际位置；
- 必须通过 `mujoco.mj_step()` 推进物理系统。

### Day 5：夹爪与接触

实现夹爪宽度到控制量的映射：

```text
0.00 m → ctrl 0
0.04 m → ctrl 127.5
0.08 m → ctrl 255
```

实现：

- `width_to_ctrl()`
- `command_gripper()`
- `open_gripper()`
- `close_gripper()`
- `get_gripper_width()`

理解了 Panda 夹爪中的：

- tendon；
- split tendon；
- equality constraint；
- affine bias；
- 双指同步运动。

完成接触实验：

- 方块与地面接触；
- 左手指单侧接触；
- 右手指单侧接触；
- 双指同时接触；
- 开放状态无接触。

认识到“发生接触”不等于“成功抓取”。成功抓取还需要持续接触、闭合约束和物体随夹爪稳定移动。

### Day 6：项目自包含与可复现性

将 Panda 模型、场景、网格和许可证复制到仓库内部：

```text
assets/robots/panda/
```

默认场景路径改为从项目根目录推导：

```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE = (
    PROJECT_ROOT
    / "assets"
    / "robots"
    / "panda"
    / "scene_with_cube.xml"
)
```

项目不再依赖本机外部模型目录。

### Day 7：第一周集成验收

新增：

```text
examples/week1_acceptance.py
```

该脚本依次验证：

1. 场景加载；
2. 连续 10 次复位；
3. 三个关节目标；
4. 三个夹爪宽度目标；
5. 300 秒长时间仿真稳定性。

## 4. 验收结果

### 自动测试

运行：

```powershell
python -m pytest -v
```

结果：

```text
16 passed
```

测试覆盖：

- 夹爪宽度映射；
- 非法宽度输入；
- 夹爪控制量写入；
- 夹爪实际运动；
- 夹爪开合函数；
- 连续复位一致性；
- 污染状态恢复；
- joint4 初始位置合法性；
- qpos 与 ctrl 对齐；
- 单自由度关节限位。

### 重复复位

```text
10/10 PASSED
```

每次复位后：

- `qpos` 回到标准状态；
- `qvel` 清零；
- `ctrl` 回到 home 控制量。

### 关节目标

```text
target 1：最大误差 0.006527 rad
target 2：最大误差 0.006530 rad
target 3：最大误差 0.006528 rad
```

验收标准：

```text
最大关节误差 < 0.02 rad
```

三组目标全部通过。

### 夹爪目标

```text
0.00 m：误差 3.10e-06 m
0.04 m：误差 1.55e-06 m
0.08 m：误差 1.69e-06 m
```

验收标准：

```text
夹爪宽度误差 < 0.0005 m
```

三组目标全部通过。

### 长时间稳定性

```text
仿真时间：300.0 s
实际计算时间：2.76 s
方块漂移：1.140e-16 m
最大关节误差：0.006527 rad
最终夹爪宽度：0.079998 m
```

长时间仿真中没有出现：

- NaN；
- Inf；
- 明显方块漂移；
- 关节发散；
- 夹爪异常闭合。

最终结果：

```text
Week 1 acceptance: PASSED
```

## 5. 当前代码结构

```text
src/panda_mujoco/
├── simulation.py
├── joint_trajectory.py
├── gripper.py
└── contact.py
```

各模块职责：

- `simulation.py`：模型加载、home 状态和复位；
- `joint_trajectory.py`：关节空间轨迹；
- `gripper.py`：夹爪宽度控制；
- `contact.py`：接触信息识别。

## 6. 本周理解的核心关系

### 控制目标与实际状态

```text
data.ctrl
    ↓
执行器产生力
    ↓
mj_step 推进动力学
    ↓
data.qvel 变化
    ↓
data.qpos 变化
```

因此，修改 `ctrl` 不会瞬间修改 `qpos`。

### 模型与数据

```text
MjModel：模型结构、质量、关节、限位、执行器
MjData：位置、速度、控制量、接触、时间
```

### XML 依赖关系

```text
scene_with_cube.xml
    ↓ include
panda.xml
    ↓ mesh file
assets/*.obj 和 assets/*.stl
```

## 7. 当前局限

目前项目仍有以下局限：

1. 只能直接控制关节目标，尚未实现末端笛卡尔空间控制；
2. 尚未实现 Jacobian 和逆运动学；
3. 接触检测尚未升级为完整的抓取成功判定；
4. 尚未实现自动抓取状态机；
5. `reset_to_home()` 当前没有把仿真时间重置为零；
6. 第一周关节验收主要改变 joint1，尚未覆盖复杂多关节轨迹；
7. 尚未接入持续集成和自动发布流程。

## 8. 第二周计划

第二周进入末端控制，计划依次完成：

1. 读取末端位置和姿态；
2. 理解 world frame、base frame 和 end-effector frame；
3. 使用 `mujoco.mj_jacSite()` 计算末端 Jacobian；
4. 验证 Jacobian 与有限差分结果；
5. 实现阻尼最小二乘逆运动学；
6. 实现末端位置控制；
7. 添加目标点和误差记录；
8. 为第三周抓取任务提供末端移动接口。

计划形成的接口：

```python
get_ee_pose(scene)
compute_ee_jacobian(scene)
solve_damped_least_squares(jacobian, error)
move_ee_to(scene, target_position)
```

## 9. 第一周总结

第一周完成了一个可安装、可测试、可复位、可进行关节和夹爪控制的 Panda MuJoCo 项目基础。

这一阶段最重要的成果不是某一个单独的运动效果，而是建立了后续算法可以依赖的统一入口、明确状态和可重复验收标准。

第二周将在该基础上从关节空间控制过渡到末端笛卡尔空间控制。
