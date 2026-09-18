---
exp_id: PM-B-260918-001
date: 2026-09-18
project: panda-mujoco-manipulation
day: 18
exp_type: 接触分类、持续夹持与抓取状态监测
platform: Windows
python: 3.10.20
mujoco: 3.12.0
contact_duration_s: 0.1
required_contact_steps: 50
minimum_lift_height_m: 0.05
contact_test_result: "28 passed"
full_test_result: "131 passed in 3.89s"
anomaly: true
anomaly_status: resolved
anomaly_summary: 初版实验辅助函数缺少offset接口，且监测器重复读取同一仿真时刻可能重复累计接触
result: passed
tags: [MuJoCo, Panda, contact, grasp, monitor, state-estimation, pytest]
---

# Day 18 日志：接触分类、持续夹持与抓取状态监测

## 1. 实验目的

Day17 已经实现了两个抓取运动原语：

```text
MOVE_ABOVE
    ↓
APPROACH
```

机械臂可以从 home 姿态运动到方块上方，再下降到抓取位姿。但仅仅到达
抓取位姿，还不能证明方块已经被夹住。

Day18 的目标是建立一个可靠的抓取监测层，将 MuJoCo 的底层接触、位置
和速度数据转换成状态机能够直接使用的任务语义：

```text
当前是否双侧接触？
双侧接触是否持续足够长？
夹爪宽度是否像夹住了物体？
方块是否抬离初始高度？
方块是否已经稳定？
当前为什么还不能判定成功？
```

今天不实现完整抓取状态机，也不执行真实抬升。Day18 交付的是 Day19
状态机所依赖的“裁判”。

---

## 2. Day5 与 Day18 的关系

Day5 已经验证了以下四种单帧接触：

```text
OPEN CENTERED    左=0，右=0
LEFT ONLY        左>0，右=0
RIGHT ONLY       左=0，右>0
CLOSED BILATERAL 左>0，右>0
```

当时的简化判断为：

```python
left_contacts > 0 and right_contacts > 0
```

它只能证明当前物理步中两根手指都碰到了方块。`data.contact` 会在每个
物理步重新计算，不会自动保存过去的接触历史。

因此，下面两种情况在 Day5 的单帧判断中都可能得到 `True`：

```text
情况A：两根手指只在一帧中擦过方块
情况B：两根手指已经连续稳定夹住方块0.2秒
```

Day18 在 Day5 基础上增加时间记忆、夹爪宽度、抬升高度和速度判据。

---

## 3. 监测系统的职责分层

今天形成四层结构：

```text
ContactSensor
    ↓ 当前帧分类
ContactSnapshot
    ↓ 输入历史计数器
BilateralContactTracker
    ↓ 与夹爪/方块测量组合
GraspMonitor
    ↓
GraspStatus
```

### 3.1 `ContactSensor`

负责读取 `data.contact`，识别接触双方的 geom，并把接触分为：

- 左手指—方块；
- 右手指—方块；
- 方块—地面。

它没有历史，只解释当前物理步。

### 3.2 `ContactSnapshot`

保存当前物理步的分类结果：

```python
ContactSnapshot(
    left_cube_contacts=16,
    right_cube_contacts=16,
    cube_floor_contacts=0,
    total_contacts=32,
)
```

它提供两个派生属性：

```python
snapshot.bilateral_contact
snapshot.cube_on_floor
```

`ContactSnapshot` 使用 `@dataclass(frozen=True)`，创建后不可修改。下一
物理步应创建一个新快照，而不是改写旧快照。

### 3.3 `BilateralContactTracker`

保存连续双侧接触步数：

```python
if snapshot.bilateral_contact:
    consecutive_steps += 1
else:
    consecutive_steps = 0
```

任何一侧接触丢失，连续计数都会清零。它要求的是连续性，而不是历史
中累计出现了多少次双侧接触。

### 3.4 `GraspMonitor`

它组合接触分类、历史计数和物理测量，生成完整的 `GraspStatus`。

监测器只读取仿真状态，不会：

- 写入 `ctrl`；
- 修改 `qpos`；
- 修改 `qvel`；
- 调用 `mj_step()`；
- 移动机械臂或夹爪；
- 决定状态机进入哪个阶段。

它只修改自己的内部接触计数和上次更新时间。

### 3.5 `GraspStatus`

保存一次监测结果，并提供三个上层语义：

```python
status.grasp_candidate
status.grasp_success
status.reason
```

---

## 4. 接触对象如何识别

本模型中：

```text
left_finger  是body，直接拥有8个geom
right_finger 是body，直接拥有8个geom
cube         是body，直接拥有cube_geom
floor        是直接属于world的具名geom
```

因此采用两种查找方式。

### 4.1 手指和方块

通过 body 名称得到 body ID，再扫描：

```python
model.geom_bodyid
```

收集该 body 直接拥有的全部 geom ID。

### 4.2 地面

`floor` 不是独立 body，因此直接按 geom 名称查询：

```python
mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_GEOM,
    "floor",
)
```

实际审计结果为：

```text
left geom count: 8
right geom count: 8
cube geom IDs: {82}
floor geom IDs: {0}
floor geom name: floor
```

这些 ID 由模型查询得到，没有硬编码在程序中。

---

## 5. 通用接触计数

MuJoCo 不保证接触双方的存储顺序。一次左指—方块接触可能表示为：

```text
geom1=left_finger_geom, geom2=cube_geom
```

也可能是：

```text
geom1=cube_geom, geom2=left_finger_geom
```

因此 `_count_between()` 同时检查两个方向：

```python
first_to_second = (
    geom1 in first_geoms
    and geom2 in second_geoms
)

second_to_first = (
    geom1 in second_geoms
    and geom2 in first_geoms
)
```

只有接触双方分别属于指定的两个集合时才计数。

方块与地面的四个接触条目不会命中任何手指集合，因此不会被误判为
抓取接触。即使方块—地面接触持续很久，也仍然不是手指—方块接触。

---

## 6. 接触条目数不等于力或时间

实验中夹住方块时经常出现：

```text
left-cube contacts: 16
right-cube contacts: 16
```

这里的 16 表示当前物理步中，MuJoCo 为多个碰撞 geom 和接触点生成了
16 条接触约束记录。它不表示：

- 16 N 接触力；
- 已经持续16步；
- 抓取质量为16；
- 存在16个相互独立的传感器。

如需接触力，需要针对具体 contact 调用 `mujoco.mj_contactForce()`。
Day18 暂时使用几何接触与运动状态，不引入力反馈。

---

## 7. 持续接触时间

默认要求双侧接触持续：

```text
contact_duration = 0.1 s
```

当前模型时间步为：

```text
timestep = 0.002 s
```

因此：

```python
required_steps = ceil(0.1 / 0.002)
               = 50
```

使用 `ceil()` 可以保证实际要求不会短于给定持续时间。

真实探针输出：

```text
time=0.002 s | counter=0  | bilateral=False | persistent=False
time=0.094 s | counter=1  | bilateral=True  | persistent=False
time=0.192 s | counter=50 | bilateral=True  | persistent=True
time=1.000 s | counter=454| bilateral=True  | persistent=True
```

夹爪在 `0.094 s` 首次建立双侧接触。总仿真为500步，但前46步没有双侧
接触，所以最终连续计数为：

```text
500 - 46 = 454
```

---

## 8. 监测器读取的物理量

`GraspMonitor.update(scene)` 每次读取：

| 物理量 | 来源 | 任务含义 |
|---|---|---|
| 左右接触数 | `data.contact` | 当前是否双侧接触 |
| 连续双侧步数 | Tracker | 接触是否持续 |
| 实际夹爪宽度 | 两个手指关节 `qpos` | 排除完全张开或空夹 |
| 方块中心高度 | `body("cube").xpos[2]` | 计算相对抬升高度 |
| 方块线速度 | free joint `qvel[:3]` | 判断平移是否稳定 |
| 方块角速度 | free joint `qvel[3:]` | 判断旋转是否稳定 |
| 地面接触 | `data.contact` | 判断是否真正离地 |

方块 free joint 的 qvel 起始地址通过 `get_cube_joint_addresses()` 查询，
没有直接假设固定数组下标。

---

## 9. 默认阈值

今天使用的默认值为：

```text
持续接触时间       ≥ 0.1 s
有效夹爪宽度       0.005～0.075 m
最小相对抬升高度   ≥ 0.05 m
最大稳定线速度     ≤ 0.02 m/s
最大稳定角速度     ≤ 0.2 rad/s
```

### 9.1 夹爪宽度

```text
完全张开：约0.08 m
空夹到底：约0 m
夹住4 cm方块：约0.04 m
```

因此有效宽度区间排除了明显的完全张开和完全空夹状态。

宽度本身不能证明抓取，因为手指也可能被其他物体阻挡，所以必须和
持续双侧方块接触共同使用。

### 9.2 相对抬升高度

方块落在地面时，其中心高度约为：

```text
0.0199276 m
```

“抬升5 cm”不能写成 `cube_height >= 0.05`，而应计算：

```python
lift_height = cube_height - reference_cube_height
```

如果初始高度为 `0.01993 m`，抬升5 cm后的中心高度约为：

```text
0.01993 + 0.05 = 0.06993 m
```

### 9.3 参考高度的记录时机

正确顺序是：

```python
reset_cube(scene, pose)
settle_cube(scene)
monitor.reset(scene)
```

必须在方块落稳后记录参考高度。若在生成高度 `z=0.05 m` 时记录，方块
落到地面后会先产生负的相对高度，后续抬升标准也会发生偏移。

### 9.4 速度判据

线速度与角速度分别计算六维 free-joint qvel 的两个范数：

```python
linear_speed = norm(qvel[:3])
angular_speed = norm(qvel[3:])
```

方块在 LIFT 阶段运动是正常现象。因此速度稳定性不参与
`grasp_candidate`，只参与抬升后的最终 `grasp_success`。

---

## 10. 抓取候选与抓取成功

### 10.1 `grasp_candidate`

```python
grasp_candidate = (
    persistent_contact
    and width_valid
)
```

它回答：

> 夹爪是否已经可靠夹住方块，可以尝试进入 LIFT？

此时方块仍可与地面接触。如果候选条件也要求方块已经抬起，就会出现：

```text
没有抬起 → 不算抓住
不算抓住 → 不允许抬起
```

因此必须把预抬升判定和最终成功判定分开。

### 10.2 `grasp_success`

```python
grasp_success = (
    grasp_candidate
    and cube_lifted
    and cube_stable
    and not cube_on_floor
)
```

它回答：

> 方块是否已经抬离地面，并在保持阶段处于稳定状态？

---

## 11. 失败原因

`GraspStatus.reason` 按任务流程返回首要失败原因：

```text
no_bilateral_contact
contact_not_persistent
gripper_width_invalid
cube_on_floor
cube_not_lifted
cube_not_stable
none
```

例如，张开的夹爪面对地面上的方块时，同时存在：

- 没有双侧接触；
- 夹爪宽度无效；
- 方块仍在地面；
- 方块没有抬升。

程序返回 `no_bilateral_contact`，因为这是流程中首先失败的条件。这使
Day19 状态机能够定位当前最直接的阻塞原因。

---

## 12. 六类场景验收

`grasp_status_probe.py` 构造并检查以下场景。

### 12.1 无接触

```text
left=0, right=0
bilateral=False
candidate=False
reason=no_bilateral_contact
```

### 12.2 只有左侧接触

```text
left=16, right=0
bilateral=False
candidate=False
```

### 12.3 只有右侧接触

```text
left=0, right=16
bilateral=False
candidate=False
```

左右单侧实验沿手指的实际世界开合轴移动方块。由于 home 姿态下开合
轴包含 z 分量，左右偏移时方块世界高度也会相应变化。这不是异常，
而是局部轴在世界坐标系中的正确表现。

### 12.4 短暂双侧接触

```text
contact history: 49/50 steps
bilateral=True
persistent=False
width=0.039690 m
candidate=False
reason=contact_not_persistent
```

即使当前已经双侧接触，只要持续时间还差一步，就不能进入抬升阶段。

### 12.5 持续双侧接触

```text
contact history: 50/50 steps
bilateral=True
persistent=True
width=0.039707 m
width_valid=True
candidate=True
success=False
reason=cube_not_lifted
```

实际夹爪宽度接近方块的4 cm尺寸，且接触持续达到门槛，因此可以进入
LIFT。但实验没有执行抬升，所以不能判定最终成功。

### 12.6 方块—地面接触

```text
left=0, right=0, floor=4
cube_on_floor=True
bilateral=False
candidate=False
success=False
```

地面接触没有被误判为抓取。

最终结果：

```text
Day 18 grasp status probe: PASSED
```

---

## 13. 可视化演示

`contact_persistence_visual_demo.py` 展示：

```text
Stage 1：方块悬停在张开的两指之间
Stage 2：夹爪通过动力学逐渐闭合
Stage 3：双侧接触建立并持续到门槛
```

该演示关闭重力，只用于隔离接触建立过程。它不是完整抓取，因为：

- 方块没有从桌面抓取；
- 机械臂没有执行 LIFT；
- 没有完成最终成功验收。

关闭重力不是为了让结果更好看，而是为了控制变量，使接触计数逻辑
能够独立观察和解释。

---

## 14. 监测器与控制器的分工

```text
控制器写入动作
    ↓
MuJoCo执行mj_step
    ↓
监测器读取新状态
    ↓
GraspStatus返回任务语义
    ↓
控制器决定是否切换状态
```

推荐调用顺序：

```python
apply_control(scene)
mujoco.mj_step(scene.model, scene.data)
status = monitor.update(scene)
controller.update(status)
```

职责分别为：

```text
控制器：接下来做什么？
MuJoCo：动作执行后世界变成什么样？
监测器：当前到底发生了什么？
GraspStatus：把观察翻译成任务结论。
```

---

## 15. 防止同一物理帧重复计数

初版 `GraspMonitor.update()` 每被调用一次，就会更新一次接触计数。

如果调用者在没有执行 `mj_step()` 的情况下重复调用：

```python
monitor.update(scene)
monitor.update(scene)
monitor.update(scene)
```

同一个物理时刻可能被错误当成三帧。

修复后，监测器保存：

```python
self._last_update_time
```

规则为：

```text
第一次读取：计数一次
data.time增加：计数一次
data.time相同：只刷新测量，不重复计数
data.time倒退：报错，提示在scene reset后调用monitor.reset()
```

对应测试连续读取同一个 `scene.data.time` 两次，确认计数仍为1，而不是2。

这不能代替“每个物理步调用一次”的正常接口约定，但可以防止重复读取
同一帧造成明显的假阳性。

---

## 16. 今日异常与修复

### 16.1 `offset` 参数接口不一致

首次运行六场景探针时出现：

```text
TypeError: place_cube_between_fingers()
got an unexpected keyword argument 'offset'
```

原因是 Day18 初版辅助函数只支持把方块放在夹爪中心，而六场景探针
需要沿开合轴构造左右单侧接触。

修复方式：

1. 为函数增加 `offset: float = 0.0`；
2. 读取 `finger_joint1` 当前世界开合轴；
3. 使用 `ee_position + offset * opening_axis` 计算方块位置；
4. 验证 offset 有限。

修复后成功构造左侧和右侧接触。

### 16.2 同一时刻重复累计

在梳理监测器与控制器调用图时发现，单纯按 `update()` 调用次数计数存在
重复读取同一帧的风险。

通过记录 `data.time` 和增加专项测试解决，测试结果通过。

---

## 17. 自动测试

`tests/test_contacts.py` 共28项测试，覆盖：

- 无接触、左侧、右侧和双侧接触；
- 地面接触不等于双侧抓取；
- 持续计数在门槛处准确成立；
- 任意一侧丢失后计数清零；
- 非法持续步数被拒绝；
- 抓取候选可以在抬升前成立；
- 最终成功要求抬升、稳定且离地；
- 七种 `reason` 分支；
- 真实 MuJoCo 方块落地场景；
- `GraspMonitor.update()` 不修改物理状态；
- 同一仿真时刻不会重复累计；
- `reset()` 清除接触历史并更新参考高度；
- 非法监测阈值被拒绝。

专项测试结果：

```text
28 passed in 0.82s
```

完整测试结果：

```text
131 passed in 3.89s
```

`git diff --check` 无输出，说明本次修改没有行尾空格等补丁格式问题。

---

## 18. 当前局限

### 18.1 没有接触力判据

当前只判断接触是否存在，没有读取法向力和切向力。持续双侧接触加上
抬升与稳定性足以支持当前脚本抓取任务，但不等于完整的抓取质量评估。

### 18.2 阈值与当前物体相关

默认夹爪宽度范围针对当前4 cm方块设计。更换不同尺寸物体时，应重新
配置或根据物体尺寸计算宽度阈值。

### 18.3 速度稳定只适合保持阶段

方块在 LIFT 过程中本来就应有非零速度。不能在运动阶段因为
`cube_stable=False` 就判定抓取失败。最终稳定判据应在抬升完成并保持
一段时间后使用。

### 18.4 今天没有产生真实成功抓取

Day18 的 `grasp_success` 逻辑已定义并测试，但实际实验没有执行完整
抬升。真实的成功状态将在 Day19 状态机中产生和验证。

---

## 19. 与 Day19 的关系

Day19 将实现：

```text
RESET
MOVE_ABOVE
APPROACH
CLOSE_GRIPPER
VERIFY_CONTACT
LIFT
CHECK_SUCCESS
DONE / FAILED
```

其中：

```text
VERIFY_CONTACT
    使用 status.grasp_candidate

CHECK_SUCCESS
    使用 status.grasp_success

FAILED
    记录 status.reason 和失败阶段
```

Day18 将底层 MuJoCo 数据转换成任务语义，使 Day19 的状态机不需要理解
geom ID、free-joint 地址、速度数组布局或持续步数换算。

---

## 20. 交付文件

```text
src/panda_mujoco/contact.py
examples/day18/contact_persistence_probe.py
examples/day18/contact_persistence_visual_demo.py
examples/day18/grasp_status_probe.py
tests/test_contacts.py
docs/day18_log.md
```

---

## 21. 常用复现命令

持续接触探针：

```powershell
& "C:\Users\29391\.conda\envs\mujoco\python.exe" `
  "examples\day18\contact_persistence_probe.py"
```

持续接触动画：

```powershell
& "C:\Users\29391\.conda\envs\mujoco\python.exe" `
  "examples\day18\contact_persistence_visual_demo.py"
```

六类状态探针：

```powershell
& "C:\Users\29391\.conda\envs\mujoco\python.exe" `
  "examples\day18\grasp_status_probe.py"
```

Day18专项测试：

```powershell
& "C:\Users\29391\.conda\envs\mujoco\python.exe" `
  -m pytest "tests\test_contacts.py" -v
```

完整测试：

```powershell
& "C:\Users\29391\.conda\envs\mujoco\python.exe" -m pytest -q
```

---

## 22. Day18 验收结论

已完成：

- 当前帧左右手指—方块和方块—地面接触分类；
- 持续双侧接触计数；
- 同一仿真时刻重复读取保护；
- 夹爪宽度、相对抬升高度和方块速度读取；
- `grasp_candidate` 与 `grasp_success` 分层；
- 可追踪失败原因；
- 六类场景探针；
- MuJoCo 可视化演示；
- 28项专项测试和131项完整测试。

Day18 验收通过，可以进入 Day19 脚本抓取状态机。
