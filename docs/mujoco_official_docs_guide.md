# MuJoCo 官方文档精读笔记

> 官方入口：<https://mujoco.readthedocs.io/en/stable/overview.html>
>
> 当前用途：服务于 Panda 末端控制与抓取项目，而不是逐句翻译整套手册。
>
> 阅读原则：先解释概念，再对应项目代码，最后用小实验验证。

## 0. 这份笔记如何使用

官方文档信息密度很高，而且底层接口沿用 C API 的术语。第一次阅读不需要试图记住所有结构体、函数和求解器名称。

以后每个主题统一按照以下格式整理：

1. 官方内容在回答什么问题；
2. 用直白语言解释核心概念；
3. 对应到当前 Panda 项目的 XML 和 Python；
4. 标记“现在必须掌握”与“以后再学”；
5. 给出一个可以实际验证的小实验；
6. 记录仍未理解的问题。

难度标记：

- **A：现在掌握**——会直接影响当前项目；
- **B：知道用途**——遇到时能查到即可；
- **C：以后再学**——当前跳过不会阻碍末端控制和抓取。

---

## 1. 推荐阅读顺序

不要完全按照网站侧边栏从头读到尾。针对当前项目，推荐顺序如下：

| 阶段 | 官方章节 | 当前目标 | 优先级 |
|---|---|---|---|
| 1 | Overview | 建立 MuJoCo 的整体心智模型 | A |
| 2 | Modeling | 理解 body、joint、geom、site、actuator、constraint | A |
| 3 | XML Reference | 写或查 MJCF 属性；把它当字典，不要求通读 | A/B |
| 4 | Programming → Simulation | 理解状态、控制、`mj_step`、`mj_forward`、回调 | A |
| 5 | Computation | 学习正动力学、约束、接触、Jacobian | B；用到哪读到哪 |
| 6 | Visualization | viewer、相机、离屏渲染 | B |
| 7 | API Reference | 查函数和数据字段 | B；不通读 |
| 8 | MJX / Warp / Plugin / OpenUSD | 大规模并行或扩展功能 | C |

当前先完成 Overview 中与 Panda 项目直接相关的内容，然后进入 Modeling。

---

# 第一章：Overview 解读

## 2. MuJoCo 到底是什么？

MuJoCo 的名字来自 **Multi-Joint dynamics with Contact**，核心任务是快速、准确地模拟“多关节机构与环境发生接触”的动力学。

它可以用于机器人、人体生物力学、控制、状态估计、机构设计和机器学习。对本项目来说，它主要承担三件事：

1. 根据 Panda 的质量、惯量、关节和执行器计算机械臂运动；
2. 计算夹爪、方块和地面之间的碰撞与接触力；
3. 向控制算法提供状态、Jacobian 和动力学量。

MuJoCo 本身不是完整的机器人学习框架。它不会自动替你完成：

- 逆运动学策略设计；
- 抓取状态机；
- Gymnasium 环境接口；
- 强化学习算法；
- 任务奖励和成功判定。

这些属于我们在 MuJoCo 物理引擎之上编写的项目代码。

### 与当前项目对应

```text
MuJoCo 提供                         我们编写
────────────────────────────────────────────────
物理规则与积分                      reset 和控制流程
关节、执行器、接触计算              目标宽度与关节轨迹
qpos/qvel/ctrl/contact              抓取成功判定
site 世界位姿与 Jacobian            末端控制和 IK
viewer                              演示程序
```

**掌握级别：A。**

---

## 3. MJCF 不是仿真本身，而是模型源文件

用户编写的 XML 使用 MJCF 描述语言。MuJoCo 先解析并编译它，然后生成适合高速计算的低层模型。

```text
panda.xml + scene_with_cube.xml + mesh assets
                     │
                     │ 解析与编译
                     ▼
                  MjModel
                     │
                     │ 创建运行状态
                     ▼
                   MjData
```

Python 中对应：

```python
model = mujoco.MjModel.from_xml_path(scene_path)
data = mujoco.MjData(model)
```

第一行不只是“读取文本”。它会完成很多编译工作，例如：

- 展开 `<include>`；
- 应用 `<default>` 继承；
- 加载和处理 mesh；
- 建立 body、joint、geom、site、actuator 的 ID；
- 生成 `qpos`、`qvel`、`ctrl` 地址表；
- 检查模型是否合法；
- 为运行时计算预分配结构。

所以 XML 中短短的：

```xml
<joint name="joint1"/>
```

经过默认类继承和编译后，会在 `MjModel` 中得到类型、轴线、限位、阻尼、惯量等完整数据。

### 为什么项目应该保存 XML，而不只保存 MJB？

官方说明，编译后的 MJB 二进制文件与 MuJoCo 版本相关，而且不能反向还原成原始模型。因此项目应把 XML 和资源文件作为长期维护的源文件；MJB 更适合特定版本下的快速加载或部署。

**掌握级别：A。**

---

## 4. `MjModel` 与 `MjData` 为什么分开？

这是官方 Overview 最重要的概念之一。

### `MjModel`：规则和结构

`MjModel` 保存相对稳定的模型描述，例如：

- body、joint、geom、site 的数量和名称；
- 质量、惯量和几何参数；
- 关节范围；
- actuator 参数；
- 仿真时间步和求解器选项；
- 各对象在数组中的地址关系。

当前代码示例：

```python
scene.model.nq
scene.model.jnt_range
scene.model.opt.timestep
scene.model.actuator("actuator8").id
```

### `MjData`：此刻的状态与计算结果

`MjData` 保存会随仿真变化的内容，例如：

- 当前时间 `time`；
- 当前状态 `qpos`、`qvel`；
- 当前命令 `ctrl`；
- 当前末端位置 `site_xpos`；
- 当前接触 `contact`；
- 动力学计算中的中间结果。

当前代码示例：

```python
scene.data.qpos
scene.data.qvel
scene.data.ctrl
scene.data.site("ee_center_site").xpos
scene.data.contact
```

### 一个模型可以配多个状态

这种分离意味着可以让多个 `MjData` 共用同一个 `MjModel`：

```text
                    ┌── MjData A：home 状态
一个 MjModel ───────┼── MjData B：正在抓取
                    └── MjData C：另一组控制参数
```

这对批量采样、有限差分和并行强化学习很重要。当前项目暂时使用一个 model 和一个 data，但要理解二者不是“一整个仿真对象”。

### 当前项目中的一句话版本

```text
MjModel = 世界服从什么规则
MjData  = 世界此刻是什么状态
```

**掌握级别：A。**

---

## 5. 什么是广义坐标？

MuJoCo 使用 joint/generalized coordinates 表示系统状态，而不是给每个 body 都保存一套独立的世界坐标。

对于 Panda：

```text
joint1...joint7 的角度
        ↓ 正运动学递推
link1...hand 的世界位姿
        ↓
ee_center_site 的世界位姿
```

我们保存七个关节角，就能通过 body 树推导整台机械臂的姿态。没有必要把每根连杆的 3D 位置和姿态都当成独立未知量，再用约束把它们拼起来。

这解释了：

- `data.qpos[:7]` 是 Panda 七轴角度；
- `data.qpos` 中没有为每个 link 单独存储 7 个自由位姿数；
- `site_xpos` 是由 qpos 推导出的结果；
- 修改 qpos 后，需要 `mj_forward` 更新派生的世界坐标。

### 为什么方块不同？

方块通过 `freejoint` 相对 world 自由运动，因此必须保存完整位姿：

```text
qpos：xyz + 四元数 wxyz = 7 个数
qvel：3 个线速度 + 3 个角速度 = 6 个数
```

所以当前场景：

```text
nq = 7 个手臂关节 + 2 个手指关节 + 7 个方块位姿数 = 16
nv = 7 + 2 + 6 = 15
```

### 与传统游戏引擎思路的差异

官方特别强调：在 MuJoCo 中，joint 是给 body **增加允许运动的自由度**；没有 joint 的 body 焊接在父 body 上。某些使用冗余笛卡尔坐标的引擎则从“所有 body 都自由”开始，再用关节约束移除自由度。

**掌握级别：A。**

---

## 6. 接触为什么是 MuJoCo 的核心？

机器人抓取不是只有运动学。手指碰到方块后，需要同时处理：

- 防止几何体相互穿透的法向约束；
- 阻止滑动的切向摩擦；
- 扭转和滚动摩擦；
- 多个接触点之间的共同作用；
- 接触与关节驱动力之间的耦合。

MuJoCo 把接触力问题写成优化问题，并允许软接触。这里的“软”不是指方块使用了橡胶有限元模型，而是接触约束可以存在很小的形变或穿透量，以获得稳定、可调的数值求解。

Day 5 中方块最终中心高度约为：

```text
0.019928 m
```

而几何半边长是：

```text
0.020000 m
```

两者约 `0.000072 m` 的差值，就是当前接触参数和重力平衡下的微小软约束形变。

### 当前只需掌握

- `data.contact` 是当前时间步的接触几何信息；
- 一对物体可能产生多个 contact；
- 有 contact 不等于抓取成功；
- 接触参数会影响稳定性；
- 接触力由求解器结合整个系统状态计算。

LCP、NCP、凸优化推导和求解器收敛理论暂时只需知道用途，不要求现在推公式。

**掌握级别：概念 A，数学细节 C。**

---

## 7. 官方 Overview 中的 tendon，与 Panda 的 tendon 有何关系？

官方介绍的 tendon 有两类常见用途：

1. 具有空间路径、绕点或滑轮的绳索；
2. 耦合多个自由度的抽象传动关系。

Panda 使用第二类：

```xml
<fixed name="split">
  <joint joint="finger_joint1" coef="0.5"/>
  <joint joint="finger_joint2" coef="0.5"/>
</fixed>
```

它把左右两个手指关节组合为一个 tendon 长度。`actuator8` 控制 tendon，而 equality 进一步保证双指同步。

因此 tendon 不是必须在画面中可见的实体绳索。它也可以是用来表达关节耦合的数学传动结构。

**掌握级别：A。**

---

## 8. 通用 actuator 模型意味着什么？

MuJoCo 不把 actuator 限制为一种“电机力矩输入”。它把执行器拆成三个方面：

1. **Transmission**：作用传给哪个 joint、tendon 或 site；
2. **Activation dynamics**：输入是否先经过内部动态状态；
3. **Force generation**：控制量和当前状态怎样产生力。

当前 Panda 模型中：

- actuator1–7 传动到七个 joint；
- actuator8 传动到 tendon `split`；
- affine gain/bias 组合成近似 PD 位置伺服；
- `data.ctrl` 表示目标型命令，而不是直接力矩。

因此看到 `data.ctrl` 时不能脱离 XML 猜单位。对另一个模型，同一个数组可能表示力矩、速度、位置目标、肌肉激活量或其他控制输入。

**掌握级别：A。**

---

## 9. `mj_step` 做了什么？

官方把 `mj_step(model, data)` 作为顶层仿真步进函数。它根据当前状态和控制输入运行正向动力学，并把状态推进一个时间步。

项目中的最小循环：

```python
while scene.data.time < end_time:
    mujoco.mj_step(scene.model, scene.data)
```

概念上可以先理解为：

```text
读取当前 qpos、qvel、ctrl
           ↓
计算运动学、惯性、重力、执行器力、碰撞和约束
           ↓
求得加速度和约束力
           ↓
数值积分
           ↓
写回下一时刻的 qpos、qvel、time
```

这解释了夹爪练习中的现象：

```python
command_gripper(scene, 0.04)
```

只改变 `data.ctrl`；只有随后执行 `mj_step`，手指 `qpos` 才会逐渐改变。

官方还允许调用计算管线中的部分函数。当前最重要的例子是：

```python
mujoco.mj_forward(model, data)
```

它根据当前状态重算派生量，但不推进仿真时间。Day 3 的 reset 在直接写入 qpos 后使用它更新末端位置。

**掌握级别：A。**

---

## 10. 官方最小方块示例与我们的场景

Overview 给出的第一个 MJCF 示例包含：

- worldbody；
- 光源；
- 固定平面；
- 带 free joint 的 box。

这与 `scene_with_cube.xml` 中的方块实验几乎是同一个最小结构：

```xml
<geom name="floor" type="plane" .../>

<body name="cube" pos="0.45 0 0.05">
  <freejoint name="cube_joint"/>
  <geom name="cube_geom" type="box" .../>
</body>
```

因果关系是：

```text
body 有 freejoint
        ↓
方块具有 6 个运动自由度
        ↓
重力使它下落
        ↓
box geom 与 plane geom 检测到碰撞
        ↓
接触约束产生支撑力
        ↓
方块在地面稳定下来
```

如果删掉 freejoint，方块 body 会焊接到 world，即使保留 geom 和质量也不会自由下落。

**掌握级别：A。**

---

## 11. Overview 中暂时不深入的内容

| 内容 | 现在知道什么即可 | 何时再学 |
|---|---|---|
| Newton / CG / PGS 求解器 | 它们是约束优化的不同数值算法 | 调接触稳定性或性能时 |
| Constraint islands / sleeping | 独立约束子系统可分别求解，静止部分可休眠 | 大场景性能优化时 |
| Skin / height field / flexible objects | 皮肤用于显示，高度场用于地形，软体需要大量约束 | 项目扩展到布料、绳索、地形时 |
| Plugins | 可添加自定义传感器、执行器和被动力 | 标准功能不够用时 |
| MJX / MuJoCo Warp | GPU 上的大规模并行仿真后端 | 开始强化学习批量采样时 |
| MJB / MJZ / mjSpec | 其他模型存储和程序化编辑形式 | 部署或动态生成模型时 |

跳过这些不是遗漏，而是控制当前学习范围。

---

## 12. Overview 阶段自测

不看答案，尝试用自己的话回答：

1. 为什么 `panda.xml` 不能直接参与每一步物理计算，而要先编译成 `MjModel`？
2. `MjModel` 和 `MjData` 的根本区别是什么？
3. 为什么修改 `data.ctrl` 后，`data.qpos` 不会立刻变化？
4. 为什么 Panda 七个 link 的世界位姿不需要全部存进 qpos？
5. 为什么方块 freejoint 占 7 个 qpos，却只有 6 个速度自由度？
6. 为什么一个 actuator 可以控制两个手指关节？
7. 为什么 `data.contact` 中存在接触仍不能直接判定抓取成功？
8. XML 中没有 joint 的 body 会怎样运动？

### 参考答案要点

1. MJCF 是便于人编辑的高层描述；编译器需要展开继承、加载资源、检查模型并生成适合运行时计算的交叉索引结构。
2. `MjModel` 保存结构和规则，`MjData` 保存某一时刻的状态及中间计算结果。
3. ctrl 是输入命令；必须经过动力学计算和积分，状态才会演化。
4. MuJoCo 使用广义关节坐标，通过运动学树递推所有 body 的世界位姿。
5. 四元数用 4 个数表示 3 个旋转自由度，并带单位长度约束。
6. actuator8 传动到 tendon `split`，tendon 耦合两个 finger joint，equality 保证同步。
7. 接触只描述当前步的接触状态；稳定抓取还需要持续接触、离地、跟随和不掉落等条件。
8. 它焊接在父 body 上，只能跟随父 body 一起运动。

---

## 13. 下一章

下一步阅读官方 **Modeling** 章节，重点回答：

- body frame、joint frame 和 inertial frame 是什么关系？
- MuJoCo 如何从 geom 推断质量和惯量？
- default class 的继承规则究竟怎样工作？
- `qpos0`、joint `ref` 和我们自己写的 home 有什么区别？
- 接触参数 `solref`、`solimp` 应该怎样理解？

官方资料：

- Overview：<https://mujoco.readthedocs.io/en/stable/overview.html>
- Modeling：<https://mujoco.readthedocs.io/en/stable/modeling.html>
- XML Reference：<https://mujoco.readthedocs.io/en/stable/XMLreference.html>
- Programming / Simulation：<https://mujoco.readthedocs.io/en/stable/programming/simulation.html>
