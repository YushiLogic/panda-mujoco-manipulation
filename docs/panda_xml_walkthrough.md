# Panda 与抓取场景 XML 导读

这份文档不是要求你背诵 XML 标签，而是带你回答三个问题：

1. `scene_with_cube.xml` 和 `panda.xml` 各自负责什么？
2. XML 中的结构怎样变成 Python 里的 `model`、`data.qpos`、`data.qvel`、`data.ctrl` 和 `data.contact`？
3. 前五天写的 Python 代码为什么能找到关节、控制夹爪、读取方块和判断接触？

本文分析的源文件是：

- `D:\Mujoco Project\model\franka_emika_panda\scene_with_cube.xml`
- `D:\Mujoco Project\model\franka_emika_panda\panda.xml`

建议把本文和两个 XML 并排打开。第一次阅读只看每一节的“先记住”句子，不必一次理解所有参数。

---

## 1. 先建立全局地图

这两个 XML 是分层组织的：

```text
scene_with_cube.xml                         场景层
├── include panda.xml ─────────────────┐
├── 灯光、相机显示参数                  │
├── 地面                               │
└── 可自由运动的红色方块               │
                                         ▼
panda.xml                                机器人层
├── 默认参数和网格资源
├── Panda 连杆、关节和碰撞体
├── 末端参考点 ee_center_site
├── 双指夹爪的 tendon 和 equality
└── 8 个 actuator

                MuJoCo 编译二者
                         ▼
             一个 MjModel + 一个 MjData
```

Python 加载的是场景入口：

```python
model = mujoco.MjModel.from_xml_path("scene_with_cube.xml")
data = mujoco.MjData(model)
```

`<include file="panda.xml"/>` 不是 Python 那种运行时导入。MuJoCo 在**编译模型时**把 `panda.xml` 的内容合并进场景，最后得到一个模型。因此，Python 可以同时查询 `joint1`、`ee_center_site`、`floor` 和 `cube_joint`。

> 先记住：`panda.xml` 定义“机器人本体”，`scene_with_cube.xml` 定义“机器人在哪个实验环境中工作”。

---

## 2. 读 MJCF 时最重要的五种对象

MuJoCo 使用的 XML 格式通常叫 MJCF。读文件前，先把下面五个对象分清：

| 对象 | 可以把它想成 | 主要作用 | 是否产生状态量 |
|---|---|---|---|
| `body` | 刚体或零件的坐标架 | 建立父子层级，承载其他元素 | 本身不直接产生 `qpos` |
| `joint` | 运动许可 | 规定子 body 相对父 body 怎样运动 | 会产生 `qpos`、`qvel` |
| `geom` | 外形和碰撞壳 | 显示、碰撞、摩擦、质量推断 | 不直接产生 `qpos` |
| `site` | 贴在 body 上的标记点 | 读取末端位置、算 Jacobian、显示目标点 | 不产生质量和碰撞 |
| `actuator` | 电机或控制旋钮 | 把 `data.ctrl` 转换为力或伺服作用 | 每个 actuator 贡献一个 `ctrl` |

一个很常见的误解是“一个 body 就等于一个关节”。实际上：

- body 是零件及其局部坐标系；
- joint 是这个零件相对父零件的运动方式；
- geom 是挂在这个零件上的外形；
- 一个 body 可以没有 joint，也可以挂多个 geom。

例如 `link4` 是 body，`joint4` 是它的转动自由度，`link4_0` 等网格 geom 负责显示，`link4_c` 负责碰撞。

---

## 3. 从场景入口 `scene_with_cube.xml` 开始

这个文件只有约 30 行，适合先读。它回答的是：“Panda 周围有什么？”

### 3.1 根节点与模型合并

```xml
<mujoco model="panda scene">
  <include file="panda.xml"/>
```

- `<mujoco>` 是整个模型的根节点；
- `model="panda scene"` 是模型名称，不是文件路径；
- `<include>` 要求 `panda.xml` 能从当前 XML 所在位置找到。

如果只加载 `panda.xml`，会有机器人，但没有这里定义的红色方块和地面。加载 `scene_with_cube.xml`，则两部分都存在。

### 3.2 `statistic` 与 `visual`：只影响观察，不是控制器

```xml
<statistic center="0.3 0 0.4" extent="1"/>

<visual>
  <headlight .../>
  <rgba haze="0.15 0.25 0.35 1"/>
  <global azimuth="120" elevation="-20"/>
</visual>
```

这些参数用于画面取景、光照、雾色和默认视角。它们不会增加自由度，也不会出现在 `qpos` 或 `ctrl` 中。

排查问题时可以这样区分：

- 机器人运动不对：优先查 joint、actuator、控制代码；
- 只是看不清或默认镜头不合适：查 visual、light、viewer 设置。

### 3.3 `asset`：先定义可复用的视觉资源

```xml
<asset>
  <texture type="skybox" .../>
  <texture type="2d" name="groundplane" .../>
  <material name="groundplane" texture="groundplane" .../>
</asset>
```

`asset` 像一个资源仓库：这里先创建天空纹理、棋盘格纹理和地面材质，后面的 geom 再通过名称引用它们。

资源定义本身不会自动出现在场景中。只有下面这行把材质挂到地面 geom 后，地面才真正使用它：

```xml
<geom name="floor" ... material="groundplane"/>
```

### 3.4 `worldbody`：真正放置场景物体

```xml
<worldbody>
  <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
  <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  ...
</worldbody>
```

`worldbody` 是世界根节点。所有机器人连杆、方块、地面和灯光最终都位于这棵树里。

地面直接挂在 worldbody 上，并且没有 joint，因此它固定在世界中。`type="plane"` 表示无限平面；此时 `size` 的前两个数不表示一块有限地板的长宽。

### 3.5 为什么被注释的方块不能掉落

文件中保留了一个旧版本：

```xml
<!--
<body name="cube" pos="0.45 0 0.05">
  <geom name="cube_geom" type="box" size="0.02 0.02 0.02" .../>
</body>
-->
```

因为这个 body 没有 joint，它相对 world 是固定的。即使它有质量和碰撞外形，重力也不能让它改变位姿。

当前生效的版本增加了自由关节：

```xml
<body name="cube" pos="0.45 0 0.05">
  <freejoint name="cube_joint"/>
  <geom name="cube_geom"
        type="box"
        size="0.02 0.02 0.02"
        rgba="1 0 0 1"
        condim="4"
        friction="1.0 0.115 0.0001"/>
</body>
```

逐项拆开：

- `body name="cube"`：建立方块刚体及局部坐标系；
- `pos="0.45 0 0.05"`：初始时 body 原点位于世界坐标 `(0.45, 0, 0.05)` 米；
- `freejoint`：允许方块三维平移和三维转动，所以它能掉落、被推走、被抓起；
- `geom type="box"`：碰撞和显示形状是盒子；
- `size="0.02 0.02 0.02"`：MuJoCo 的 box `size` 是**半边长**，所以真实尺寸为 `0.04 × 0.04 × 0.04 m`；
- `rgba="1 0 0 1"`：红、绿、蓝、透明度，即不透明红色；
- `friction`：依次是滑动、扭转、滚动摩擦系数；
- `condim="4"`：接触约束考虑法向、两个切向以及一个扭转摩擦方向。

方块中心初始高度是 `0.05 m`，半边长是 `0.02 m`，所以底面起初在 `z=0.03 m`，比地面高 `0.03 m`。仿真开始后它会落下，最终中心高度约为 `0.02 m`。这正是 Day 5 稳定性测试中 `z≈0.019928 m` 的来源。

### 3.6 为什么自由关节让 `nq` 增加 7、`nv` 只增加 6

`cube_joint` 的位置状态是：

```text
[x, y, z, qw, qx, qy, qz]
```

所以它占 7 个 `qpos`。但速度是：

```text
[vx, vy, vz, wx, wy, wz]
```

只占 6 个 `qvel`。四元数用 4 个数字表达 3 个旋转自由度，并且要满足单位长度约束，因此 `nq` 比 `nv` 多 1。

> 先记住：使方块能动的不是 `geom`，而是 `freejoint`；方块的碰撞外形和摩擦来自 `geom`。

---

## 4. 再读机器人文件 `panda.xml`

这个文件更长，但不用从第一行硬读到最后一行。按以下顺序看最容易理解：

```text
compiler / option
        ↓
default（模板）
        ↓
asset（网格和材质）
        ↓
worldbody（机器人运动链）
        ↓
tendon + equality（双指耦合）
        ↓
actuator（ctrl 的入口）
        ↓
contact（碰撞例外）
```

### 4.1 `compiler` 与 `option`：编译和仿真规则

```xml
<compiler angle="radian" meshdir="assets" autolimits="true"/>
<option integrator="implicitfast"/>
```

- `angle="radian"`：XML 内所有角度按弧度解释；
- `meshdir="assets"`：诸如 `link0.stl` 的网格文件从 `assets` 子目录寻找；
- `autolimits="true"`：只要元素写了 `range`，MuJoCo 自动启用对应限位；
- `integrator="implicitfast"`：选择隐式快速积分器，通常比简单显式积分对刚性系统更稳定。

这也说明为什么不能只复制两个 XML、却漏掉 `assets` 目录：模型编译时找不到 STL/OBJ 网格就会失败。

### 4.2 `default`：减少重复书写的参数模板

核心模板如下：

```xml
<default class="panda">
  <material specular="0.5" shininess="0.25"/>
  <joint armature="0.1" damping="1" axis="0 0 1"
         range="-2.8973 2.8973"/>
  <general dyntype="none" biastype="affine"
           ctrlrange="-2.8973 2.8973" forcerange="-87 87"/>
  ...
</default>
```

它的意思不是“现在创建一个关节或电机”，而是规定：以后继承 `panda` 类的 joint/general 如果没单独写某个属性，就使用这里的默认值。

例如：

```xml
<body name="link1" ...>
  <joint name="joint1"/>
</body>
```

`link0` 写了 `childclass="panda"`，后代元素会继承该类，因此短短的 `<joint name="joint1"/>` 实际还具有：

- `type="hinge"`（joint 的默认类型）；
- `axis="0 0 1"`；
- `range="-2.8973 2.8973"`；
- `armature="0.1"`；
- `damping="1"`。

而 `joint2` 单独写了 `range="-1.7628 1.7628"`，这个局部值会覆盖模板范围。

#### 手指模板

```xml
<default class="finger">
  <joint axis="0 1 0" type="slide" range="0 0.04"/>
</default>
```

两个 `finger_joint` 都引用 `class="finger"`，因此它们不是转动关节，而是沿各自局部 y 轴平移的滑动关节，行程为 0 到 0.04 米。

#### 显示与碰撞模板

```xml
<default class="visual">
  <geom type="mesh" contype="0" conaffinity="0" group="2"/>
</default>

<default class="collision">
  <geom type="mesh" group="3"/>
</default>
```

同一连杆通常挂两套 geom：

- `visual`：形状精细、负责好看，但关闭碰撞；
- `collision`：形状可更简单、负责接触计算。

夹爪指垫又用多个小 box 近似接触面。这解释了 Day 5 为什么一次夹持可能出现十几个甚至几十个 contact：MuJoCo 报告的是多个 geom 之间的多个接触点，不是“一个手指只对应一条接触记录”。

> 阅读默认类的技巧：看到一个元素属性很少，不要马上认为参数缺失；先沿 `class` 和 `childclass` 找它继承了什么。

### 4.3 `asset`：机器人外观和碰撞网格仓库

`asset` 中先定义材质，然后加载 STL/OBJ：

```xml
<mesh name="link0_c" file="link0.stl"/>
<mesh file="link1.obj"/>
```

- 名称带 `_c` 的通常是 collision mesh；
- 其他大量 OBJ 通常是拆分后的 visual mesh；
- 未显式写 `name` 时，MuJoCo 可从文件名推断网格名，例如 `link1.obj` 可由 `mesh="link1"` 引用。

网格 asset 只是资源。下面这种 geom 才把网格真正挂到 body 上：

```xml
<geom material="white" mesh="link1" class="visual"/>
<geom mesh="link1_c" class="collision"/>
```

### 4.4 `worldbody`：嵌套 body 就是串联运动链

Panda 的主体结构可压缩成：

```text
world
└── link0（固定基座）
    └── joint1 → link1
        └── joint2 → link2
            └── joint3 → link3
                └── joint4 → link4
                    └── joint5 → link5
                        └── joint6 → link6
                            └── joint7 → link7
                                └── hand
                                    ├── ee_center_body
                                    │   └── ee_center_site
                                    ├── finger_joint1 → left_finger
                                    └── finger_joint2 → right_finger
```

这里要抓住一个规则：**子 body 的位姿先相对父 body 定义，父 body 一动，整棵子树都跟着动。**

例如 `joint4` 转动时，`link4` 以及它下面的 `link5 → link6 → link7 → hand → fingers → ee_center_site` 都会随之改变世界位姿。这就是正运动学由关节角逐级传递到末端的结构基础。

#### `pos` 和 `quat` 是局部变换

```xml
<body name="link4" pos="0.0825 0 0" quat="1 1 0 0">
```

这里的 `pos` 不是 `link4` 的固定世界坐标，而是相对父 body `link3` 的局部平移；`quat` 也是相对父坐标系的局部旋转。MuJoCo 会把父子变换连乘，计算每个 body 的世界位姿。

文件里部分四元数看起来不是单位长度，例如 `quat="1 1 0 0"`。MuJoCo 编译时会归一化它，因此表示绕 x 轴旋转 90°，不是一个非法姿态。

#### `inertial` 是动力学参数

```xml
<inertial mass="3.587895" pos="..." fullinertia="..."/>
```

它定义质量、质心位置和惯量张量。运动学只关心关节和几何变换；动力学仿真还需要这些量，才能算重力、惯性和加速度。

机械工程视角可以这样对应：

- `mass`：质量；
- `pos`：质心相对 body 坐标系的位置；
- `fullinertia`：惯量矩阵的 6 个独立分量。

#### 七个手臂关节的限位

| qpos 下标 | joint | 类型 | 范围（rad） |
|---:|---|---|---|
| 0 | `joint1` | hinge | `[-2.8973, 2.8973]` |
| 1 | `joint2` | hinge | `[-1.7628, 1.7628]` |
| 2 | `joint3` | hinge | `[-2.8973, 2.8973]` |
| 3 | `joint4` | hinge | `[-3.0718, -0.0698]` |
| 4 | `joint5` | hinge | `[-2.8973, 2.8973]` |
| 5 | `joint6` | hinge | `[-0.0175, 3.7525]` |
| 6 | `joint7` | hinge | `[-2.8973, 2.8973]` |

注意 `joint4` 的范围不包含 0。这就是直接创建 `MjData` 后把所有 `qpos` 留为 0 会产生非法初态的原因，也是 Day 3 要实现 `reset_to_home()` 的直接动机。

### 4.5 `ee_center_site`：末端控制的测量点

```xml
<body name="ee_center_body" pos="0 0 0.105">
  <site name="ee_center_site" size="0.01" group="3"/>
</body>
```

这个 site 位于 hand 坐标系 z 方向 0.105 米处，接近两指之间的抓取中心。它没有质量，也不参与碰撞；它只是一个随 hand 运动的命名参考点。

Python 中：

```python
site_id = model.site("ee_center_site").id
ee_position = data.site_xpos[site_id].copy()
ee_rotation = data.site_xmat[site_id].reshape(3, 3).copy()
```

这两项是由当前 `qpos` 推导出的世界坐标结果。手动修改 `data.qpos` 后，要先调用 `mujoco.mj_forward(model, data)`，再读取它们。

后续末端控制和 IK 也会以这个 site 为对象：

```python
mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
```

其中 `jacp` 描述关节速度如何产生末端线速度，`jacr` 描述如何产生末端角速度。

> 先记住：我们控制的“末端点”不是凭空选的 Python 数组，而是 XML 中随手掌运动的 `ee_center_site`。

### 4.6 两根手指为什么只有一个控制量

两根手指各有一个独立状态：

```xml
<joint name="finger_joint1" class="finger"/>
<joint name="finger_joint2" class="finger"/>
```

因此它们分别占 `qpos[7]` 和 `qpos[8]`。但是模型没有给每根手指各装一个 actuator，而是先用 tendon 打包：

```xml
<tendon>
  <fixed name="split">
    <joint joint="finger_joint1" coef="0.5"/>
    <joint joint="finger_joint2" coef="0.5"/>
  </fixed>
</tendon>
```

可以先把 `split` 理解为一根虚拟传动腱，它对两根手指使用相同的 0.5 系数。然后 equality 约束两指同步：

```xml
<equality>
  <joint joint1="finger_joint1" joint2="finger_joint2" .../>
</equality>
```

最后 actuator8 不直接指向某一个 finger joint，而是指向 tendon：

```xml
<general name="actuator8"
         tendon="split"
         ctrlrange="0 255"
         gainprm="0.01568627451 0 0"
         biasprm="0 -100 -10" .../>
```

所以结构是：

```text
data.ctrl[7]
      │
      ▼
 actuator8
      │
      ▼
 tendon split
   ┌──┴──┐
   ▼     ▼
finger1 finger2  ← equality 继续约束二者同步
```

在这个经过修改的模型中，夹爪命令被映射为 0–255：

- `data.ctrl[7] = 0`：闭合目标；
- `data.ctrl[7] = 255`：张开目标；
- 约 128：半开目标。

这与你 Day 5 的实测一致：255 时两指各约 0.04 m，0 时接近 0 m。

这里还要区分三个量：

- `finger_joint1/2` 的 `qpos`：实际手指位置，单位米；
- `data.ctrl[7]`：给 actuator8 的命令，范围 0–255，无量纲；
- 两指间总宽度：近似 `qpos_left + qpos_right`，最大约 0.08 m。

因此，“把 qpos 映射到 ctrl”并不准确。更准确的说法是：**执行器把 0–255 的命令按其增益和偏置转成对 tendon 的伺服作用，系统运动后，两个 finger qpos 才逐渐达到相应位置。** `ctrl` 是命令，`qpos` 是物理状态，它们不是同一个数组。

### 4.7 七个手臂 actuator 在做什么

```xml
<general name="actuator1" joint="joint1"
         gainprm="4500" biasprm="0 -4500 -450"/>
```

七个 actuator 分别指向七个手臂关节。它们使用 affine bias，形成近似 PD 位置伺服：

```text
执行器力 ≈ kp × (ctrl - qpos) - kd × qvel
```

以 actuator1 为例，`kp≈4500`、`kd≈450`。因此把 `data.ctrl[0]` 设置为某个弧度值，含义接近“让 joint1 去这个目标角度”，而不是“直接施加这么多牛米的力”。

这就是 Day 4 关节空间轨迹使用以下逻辑的原因：

```python
q_desired = (1 - alpha) * q_start + alpha * q_goal
data.ctrl[:7] = q_desired
```

插值代码产生连续变化的目标角度，XML 中的执行器负责把实际关节拉向这些目标。

`forcerange` 是执行器输出力/力矩限制，`ctrlrange` 是你允许写入命令的范围，两者不要混淆。

### 4.8 为什么注释掉的 `keyframe` 不应直接依赖

文件末尾有：

```xml
<!--
<keyframe>
  <key name="home" qpos="..." ctrl="..."/>
</keyframe>
-->
```

因为被 `<!-- -->` 包住，MuJoCo 完全看不到它，所以当前模型中没有可用的 `home` keyframe。

而且本地这段注释里的 `qpos` 与 `ctrl` 在 joint6、joint7 上并不一致。如果启用后直接重置，位置状态与执行器目标之间会突然出现较大误差，可能造成开机甩动。

因此本项目在 Python 的 `reset_to_home()` 中显式设置并对齐：

1. 七轴 home qpos；
2. 两指 qpos；
3. 方块自由关节 qpos；
4. 手臂与夹爪 ctrl；
5. qvel 清零；
6. 调用 `mj_forward` 重算派生量。

这不是绕开 XML，而是为实验建立一个明确、可测试、可重复的初始化过程。

### 4.9 `contact/exclude`：排除不需要的自碰撞

```xml
<contact>
  <exclude body1="link0" body2="link1"/>
</contact>
```

相邻连杆 link0/link1 在结构上紧邻，碰撞网格可能天然重合或无需相互碰撞，所以这里明确排除这对 body 的接触计算。

它不会关闭所有机器人碰撞，只排除指定 body 对。

---

## 5. 两个 XML 合并后，为什么是 `nq=16, nv=15, nu=8`

MuJoCo 3.12 编译当前场景得到：

| 来源 | 元素 | `qpos` 数量 | `qvel` 数量 | `ctrl` 数量 |
|---|---|---:|---:|---:|
| `panda.xml` | 7 个 hinge 关节 | 7 | 7 | 7 个手臂 actuator |
| `panda.xml` | 2 个 slide 手指关节 | 2 | 2 | 1 个 tendon actuator |
| `scene_with_cube.xml` | 1 个 freejoint | 7 | 6 | 0 |
| **合计** |  | **16** | **15** | **8** |

所以数组布局是：

```text
data.qpos，长度 16
┌───────────────┬───────────────┬────────────────────────────┐
│ 0 ... 6       │ 7 ... 8       │ 9 ... 15                   │
│ Panda 七轴角度 │ 左右指位置     │ cube: xyz + quaternion     │
└───────────────┴───────────────┴────────────────────────────┘

data.qvel，长度 15
┌───────────────┬───────────────┬────────────────────────────┐
│ 0 ... 6       │ 7 ... 8       │ 9 ... 14                   │
│ 七轴角速度     │ 左右指速度     │ cube: 3 线速度 + 3 角速度  │
└───────────────┴───────────────┴────────────────────────────┘

data.ctrl，长度 8
┌───────────────────────────────┬────────────────────────────┐
│ 0 ... 6                       │ 7                          │
│ joint1...joint7 的目标角度     │ 夹爪命令 0...255           │
└───────────────────────────────┴────────────────────────────┘
```

这不是我们随意规定的下标，而是 MuJoCo 编译 XML 后按模型元素建立的地址表。

不过正式代码不要把所有下标都“写死”。更稳妥的做法是按名称查询地址：

```python
joint_id = model.joint("cube_joint").id
cube_qpos_start = model.jnt_qposadr[joint_id]

actuator_id = model.actuator("actuator8").id
site_id = model.site("ee_center_site").id
```

如果以后在 XML 前面插入新关节，裸下标可能整体移动；名称查询仍然能找到正确对象。

---

## 6. XML 如何对应到你前五天的代码

### 6.1 Day 2：模型审计不是“打印一堆数字”

审计代码做的是验证 XML 编译结果：

| Python 查询 | 追溯到 XML | 想确认什么 |
|---|---|---|
| `model.nq/nv/nu` | joint、freejoint、actuator | 状态和控制维度是否符合预期 |
| `model.joint(name)` | `<joint name="...">` | 名称是否存在、类型和地址是否正确 |
| `model.site(name)` | `<site name="...">` | 末端参考点是否存在 |
| `model.actuator_ctrlrange` | actuator 的 `ctrlrange` | 写入控制量是否合法 |
| `model.geom(name)` | `<geom name="...">` | 接触检测能否按名称识别对象 |

审计的意义是：先确认“模型接口是什么”，之后才写控制器。否则一旦把夹爪误当成 0–0.04 控制、把 freejoint 误当成 6 个 qpos，后面的代码会从根上错掉。

### 6.2 Day 3：为什么需要统一 reset

XML 默认状态通常是 `qpos=0`，但本模型的 joint4 不允许 0；方块和控制器也需要明确初态。因此 `reset_to_home()` 在 Python 中负责把编译模型变成一个可靠的实验起点。

`mj_forward` 则根据你写入的 `qpos` 重新计算：

- 每个 body 的世界位姿；
- `ee_center_site` 的位置和姿态；
- geom 的位置；
- 潜在接触等派生量。

### 6.3 Day 4：关节轨迹为什么写入 `ctrl[:7]`

XML 中 actuator1–7 分别绑定 joint1–7，并使用位置伺服形式。因此插值得到的七维 `q_desired` 可以作为七个 actuator 的目标。

如果 XML 中使用的是纯 torque actuator，同一段代码的物理含义会完全不同。所以控制代码不能脱离 XML 阅读。

### 6.4 Day 5：夹爪探针在验证哪段 XML

`gripper_probe.py` 实际验证了三件事：

1. actuator8 的合法命令范围确实是 `[0, 255]`；
2. tendon 与 equality 能让左右指同步；
3. 0/128/255 命令对应闭合/半开/张开的实际 qpos。

你测得同步误差约 `1.85e-06 m`，说明约束不是数学上逐位完全相等，但在数值求解误差范围内高度同步。

### 6.5 Day 5：方块稳定性探针在验证哪段 XML

它验证的是：

- `freejoint` 让方块受重力自由下落；
- `cube_geom` 与 `floor` 能发生碰撞；
- 尺寸、接触参数和积分器能让方块稳定落在桌面；
- 仿真长时间推进没有 NaN/Inf 或持续抖动。

### 6.6 Day 5：为什么接触清单里地面与方块有 4 个 contact

盒子底面落在平面上时，求解器可生成多个接触点来表达稳定支撑。你看到的 4 条记录是当前时间步的接触几何信息，不等于“四次碰撞”，也不直接等于总接触力。

`data.contact` 的每条记录通过 geom id 指向 XML 中的 geom。代码再把 id 转回名字，才能得到：

```text
floor ↔ cube_geom
left finger collision geoms ↔ cube_geom
right finger collision geoms ↔ cube_geom
```

因为多数手指碰撞 geom 没有显式名字，实际代码通常还需要通过 geom 所属的 body id 判断它属于 `left_finger` 还是 `right_finger`。

这也是为什么 Day 5 的双侧接触判断不是简单检查字符串，而是沿着：

```text
contact → geom id → body id → left_finger/right_finger/cube
```

来识别接触双方。

---

## 7. 以后读 XML，可以固定使用这套步骤

面对一个陌生机器人模型，不要逐行硬啃。按下面七步做：

1. 找场景入口：哪个 XML 被 Python 加载？
2. 查 `<include>`：机器人、夹爪、场景是否拆在其他文件？
3. 看 `<worldbody>` 的 body 树：画出父子运动链。
4. 数 joint：类型、名称、range、qpos 地址分别是什么？
5. 找关键 site：末端控制和传感参考点在哪里？
6. 看 actuator：它指向 joint、tendon 还是其他对象？`ctrl` 代表位置、速度还是力？
7. 看 geom/contact：哪些对象参与碰撞，摩擦和碰撞排除如何设置？

可以先用纸写出下面这张“接口卡”，再开始写 Python：

| 问题 | 当前 Panda 场景答案 |
|---|---|
| 场景入口 | `scene_with_cube.xml` |
| 机器人模型 | `panda.xml` |
| 手臂关节 | `joint1`–`joint7` |
| 夹爪关节 | `finger_joint1/2` |
| 末端 site | `ee_center_site` |
| 可抓物体 | body `cube` / geom `cube_geom` / joint `cube_joint` |
| 手臂控制 | actuator1–7，目标近似为关节角 |
| 夹爪控制 | actuator8，`0=闭合`、`255=张开` |
| 模型维度 | `nq=16, nv=15, nu=8` |

---

## 8. 四个阅读练习

先尝试自己回答，再看后面的答案。练习目标不是记忆参数，而是练习从需求追到 XML。

### 练习 1：如果删掉 `cube_joint`，会发生什么？

思考：方块 body 和 geom 仍在，是否还能被抓起？`nq/nv` 怎样变化？

### 练习 2：为什么改变 joint4 后，末端 site 会动？

思考：在 body 树中，从 link4 到 ee_center_site 的父子路径是什么？

### 练习 3：为什么有两个 finger qpos，却只有一个 gripper ctrl？

思考：在 joint 和 actuator 之间多了哪两个耦合结构？

### 练习 4：要增加一个可视化目标点，应该放在哪一层？

假设目标点不属于 Panda 本体，只用于当前抓取任务。它应该放进 `panda.xml`，还是场景 XML？

### 参考答案

1. 没有 joint 的 cube 固定在 world，不能因重力或抓取改变位姿；模型会少 7 个 qpos 和 6 个 qvel，变成 `nq=9, nv=9`。
2. `joint4 → link4 → link5 → link6 → link7 → hand → ee_center_body → ee_center_site` 是同一子树，父关节运动会逐级改变后代世界位姿。
3. 两个 slide joint 保存实际双指状态；`tendon split` 把它们作为一条传动控制，`equality` 约束二者同步，actuator8 只需控制 tendon。
4. 放在场景层更合理。机器人本体 XML 应尽量保持通用；任务目标、桌子、方块属于实验环境。后续可以在项目自己的场景 XML 中增加 `target_site`。

---

## 9. 它与后续末端控制和抓取的关系

接下来的代码不是突然出现的，而是沿 XML 接口向上搭建：

```text
XML body/joint 树
    ↓ 正运动学
ee_center_site 当前位姿
    ↓ 与目标位姿比较
末端位置/姿态误差
    ↓ Jacobian + 阻尼最小二乘
关节角增量或关节速度
    ↓ 关节轨迹/位置伺服
data.ctrl[:7]
    ↓
机械臂靠近方块
    ↓ actuator8 + tendon + equality
夹爪闭合
    ↓ contact + 方块随末端持续移动
判定抓取是否成功
```

因此前五天不是在做零散探针：

- Day 2 确认模型提供了哪些接口；
- Day 3 建立可重复的合法初态；
- Day 4 学会通过 actuator 驱动关节；
- Day 5 验证夹爪、方块动力学和接触识别；
- 后续把这些部件串成“末端到目标 → 闭合 → 抬升 → 稳定抓取”的完整流程。

---

## 10. 一页速查：从需求反查 XML 和 Python

| 我想知道/完成什么 | 先查 XML | 再看 Python |
|---|---|---|
| 某关节能转多大 | joint `range` 和 default | `model.jnt_range` |
| 某个 qpos 在哪 | joint 的编译顺序/类型 | `model.jnt_qposadr` |
| 末端在哪里 | `ee_center_site` 所在 body 树 | `data.site_xpos/site_xmat` |
| 手臂 ctrl 是什么含义 | actuator 的 joint、gain、bias | `data.ctrl[:7]` |
| 夹爪为何同步 | tendon + equality | `data.ctrl[actuator8_id]` 和双指 qpos |
| 方块为何会掉落 | `freejoint` | cube 的 qpos/qvel |
| 方块多大 | box 的半边长 `size` | 中心落稳高度约 0.02 m |
| 谁和谁接触 | geom、body、contact exclude | `data.contact` → geom id → body id |
| 为什么改 qpos 后 site 没更新 | site 是派生量 | 调用 `mj_forward` |
| 为什么不能随便写 ctrl | actuator `ctrlrange` | 写入前 clip 或断言 |

最后只需要牢牢记住这四句话：

1. `body` 建树，`joint` 给运动自由度，`geom` 给外形和碰撞，`site` 给参考点。
2. `qpos/qvel` 是当前物理状态，`ctrl` 是执行器命令，它们不是一回事。
3. `scene_with_cube.xml` 把任务环境与通用的 `panda.xml` 合并成一个模型。
4. 后续每段控制代码，都应该能追溯到 XML 中一个明确的 joint、site、actuator、tendon 或 geom。
