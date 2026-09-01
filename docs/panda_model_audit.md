# Panda 模型审计表（Day 2）

- 审计对象：`model/franka_emika_panda/scene_with_cube.xml`（来自 mujoco_menagerie，保留原 LICENSE）
- 复现命令：`python examples/inspect_panda.py`
- 审计环境：MuJoCo 3.12.0 / Python 3.10.20
- 结论：**AUDIT PASSED**，所有名称查询有效，无 -1 ID。

## 名词速查（读表前先看这里）

| 术语 | 含义 |
|---|---|
| qpos (nq) | 关节位置型状态数组：铰链=角度(弧度)，滑轨=位移(米)，自由关节=7 个数（xyz+四元数）|
| dof (nv) | 速度自由度：自由关节的旋转只算 3 维，因此 nv < nq |
| ctrlrange | 执行器控制量 `data.ctrl` 的合法写入区间 |
| site | 挂在刚体上的参考点/坐标系；本项目末端控制一律以 `ee_center_site` 为基准 |
| tendon | 腱：把多个关节耦合成一条传动路径；`split` 以 0.5/0.5 把力分给两指 |
| HINGE / SLIDE / FREE | 转动关节 / 滑动关节 / 自由关节（6 自由度浮空刚体）|

## 维度解释

| 维度 | 值 | 来源拆解 |
|---|---|---|
| nq | 16 | 9 个铰链/滑动关节（7 臂 + 2 指）×1 qpos + cube 自由关节 ×7（位置 xyz + 四元数 wxyz）|
| nv | 15 | 同上 9 个关节 ×1 dof + cube 自由关节 ×6（平移 + 旋转）|
| nu | 8 | 7 个手臂位置执行器 + 1 个腱（tendon）执行器同时驱动两根手指 |

nq 与 nv 差 1 的原因：四元数用 4 个数表示 3 个旋转自由度（带归一化约束）。

## 关节清单

| id | 名称 | 类型 | qposadr | 范围 (rad / m) |
|---|---|---|---|---|
| 0–6 | joint1–joint7 | HINGE | 0–6 | ±2.8973 / ±1.7628 / ±2.8973 / **[-3.0718, -0.0698]** / ±2.8973 / [-0.0175, 3.7525] / ±2.8973 |
| 7–8 | finger_joint1/2 | SLIDE | 7–8 | [0, 0.04]（米，指根行程）|
| 9 | cube_joint | FREE | 9 | unlimited |

## 执行器清单

| id | 名称 | ctrlrange | 驱动目标 |
|---|---|---|---|
| 0–6 | actuator1–7 | 与对应关节限位一致（rad）| joint1–7 |
| 7 | actuator8 | **[0, 255]**（无量纲）| tendon `split`（0.5/0.5 分力到两指）|

两根手指另由 equality 约束同步，控制 actuator8 即可同步开合。方向为实测结论（测左右指 body 间距）：**0 = 闭合**（指尖并拢，间距 0），**255 = 张开**（指间距 8 cm）；home keyframe 中双指 0.04 + ctrl 255 即"张开待命"。

## 关键位姿（qpos0，未设 home）

- `ee_center_site`：(0.0880, 0.0000, 0.8210)
- cube：(0.4500, 0.0000, 0.0500)，姿态 (w,x,y,z)=(1,0,0,0)；半边长 0.02 m，静置后 z≈0.02
- 场景中唯一的 site 是 `ee_center_site`，没有 target/attachment site，后续任务需自行添加

## 发现与风险

1. **joint4 默认 qpos0=0 超出上限 -0.0698**（illegal start）。`panda.xml` 内有现成但被注释的 home keyframe（qpos≈`0 0 0 -1.57079 0 3.0 -1.7853 0.04 0.04`）。Day 3 的 `reset_to_home` 必须显式设置 7 轴 home + 手指 + cube 初态。
   - **出处考证（Day 4 追加）**：官方 mujoco_menagerie 原版的 keyframe 是生效且自洽的（joint6/7 的 qpos=ctrl=1.57079/-0.7853）；本地文件被人改为 joint6=3.0/joint7=-1.7853 但未同步 ctrl，随后整段被注释。即"qpos/ctrl 不一致"是本地编辑痕迹而非上游问题。本项目 home 采用本地姿态（已验证合法），不依赖文件内 keyframe。
2. **夹爪控制量是 0–255** 而非弧度/米（由 0–0.04 m 行程重映射而来），与手臂执行器单位不同，写夹爪代码时勿混用。
3. cube 的 `condim=4`（含扭转摩擦），摩擦 1.0/0.115/0.0001，抓取稳定性依赖这组参数；Day 3 接触检测要以 geom 名称（`cube_geom` ↔ 两侧 finger geom）配对判断。
4. 自由关节的 qpos 是 7 维（位置+四元数），读取 cube 姿态时注意与 6 维速度 dof 的错位。
