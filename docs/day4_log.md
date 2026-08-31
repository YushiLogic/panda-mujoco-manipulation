# Day 4 日志：Panda 关节空间控制

- 日期：2026-08-31
- 路线图任务：Day 4「Panda关节空间控制」
- 状态：**已完成，验收全部通过**

## 今日完成事项

1. **交付 `src/panda_mujoco/joint_trajectory.py`**：`JointPath` 线性插值规划器（MoveJ 核心），按控制拍推进，与物理步频解耦（500Hz 物理 / 50Hz 控制）
2. **交付 `examples/joint_trajectory_demo.py`**（学习者亲手敲写，非复制）：home→左偏→右偏→回 home 三段闭环挥手，viewer 实时动画 + 误差记录 + 自动存误差曲线并退出
3. **关节误差曲线**：`outputs/day4_joint_error_curve.png`（outputs/ 被 gitignore，仅本地保留；Day 7 决定是否将代表图收入 docs/）
4. **根治一个环境级疑难杂症**（详见下节）
5. 学习者独立完成的手写调试：list.append 打包语法、模块/子模块导入、qpos 广播长度、qpos-ctrl 对齐复现实验

## 实测结果

| 段 | 目标 | 到达误差（段末）| 说明 |
|---|---|---|---|
| 1 | LEFT (joint1=+0.6) | 0.0259 rad | 3.0 s |
| 2 | RIGHT (joint1=-0.6) | 0.0518 rad | 6.0 s（大角度+换向，滞后最大）|
| 3 | HOME | 0.0259 rad | 9.0 s，静止 3s 后收敛至 ≤0.0065 rad |

**误差曲线两个物理发现**：
1. joint1 三段呈现恒定平台误差（匀速跟踪特征），段边界归零（规划器重置效应），静止后收敛；
2. **joint2/joint4 存在恒定 ~0.006 rad 重力稳态误差**（绕竖直轴的 joint1 无此现象）——比例控制在重力力矩下的必然结果，为第 2 周重力补偿/积分项提供动机与数据。

**追加实验（学习者自主扩展）：底座扇形扫掠**。用 `turn(a)` 辅助函数（复制 HOME 只改 joint1）构造 9 目标 8 段序列（±0.3/±0.6 rad 扇形扫掠，24 仿真秒）。结果：8 段到达误差**全部恒等于 0.0129 rad**——因为每段都是 joint1 单关节 0.3 rad / 3s 的匀速运动，跟踪误差与速度成正比（此前 0.6 rad/3s 的段误差 0.0259，恰为 2 倍），完全符合"比例伺服匀速跟踪误差 ∝ 速度"的推论，也是系统确定性的直接证据。

## 环境级 Bug 排查实录（本次最有价值沉淀）

**现象**：demo 加入 matplotlib 绘图后，进程在渲染阶段原生崩溃（exit 127、无 Traceback、缓冲区输出丢失）。

**排障链**（每步工具都值得复用）：
1. `python -u` 关闭缓冲 → 输出不再丢失，确认三段运动本身正常；
2. 探针 print 夹逼 → 定位崩溃点在首帧渲染；
3. 剥离 mujoco 纯 matplotlib 复现 → 排除本项目代码；
4. **Windows 事件日志** `Get-WinEvent Id=1000` → 异常码 `0xC06D007F` = 延迟加载 DLL 缺函数；
5. `where` 搜索 PATH 上流氓 DLL → Git(MinGW)、"Video Monitor"、base Anaconda 三处老版本 zlib/freetype/libjpeg；
6. 根因：**conda 环境 `Library\bin` 仅在 `conda activate` 后进入 PATH**，直接调用 python.exe 时 Windows 搜到不兼容的同名 DLL。

**修复**：脚本顶部将 `<解释器所在目录>/Library/bin` 前置到 PATH（从 `sys.executable` 自动推导，可移植），等价于激活环境；另 matplotlib 用 `Agg` 后端（纯存图无窗口）。此修复对所有后续含绘图的脚本（第 2~4 周）通用。

**预防**：`.gitattributes` 已入库（顺带根治 CRLF 警告）；后续所有 demo 均沿用本文件的 PATH 头部写法。

## 验收对照（路线图 Day 4）

| 验收项 | 结果 |
|---|---|
| home/左偏/右偏目标稳定到达 | ✅ 三段全部到达（另加回 home 段，闭环挥手）|
| 不越限 | ✅ 三姿态均经限位与碰撞预检（joint1 ±0.6 ∈ ±2.8973，无自碰撞）|
| 无 NaN | ✅ 全程 isfinite 检查通过 |
| 无持续高频抖动 | ✅ 误差曲线平滑平台+收敛，无振荡 |
| 关节误差曲线 | ✅ 已保存并分析（含重力稳态误差发现）|

## Git 状态

- 上接 `5e538db`；本日待提交：`src/panda_mujoco/joint_trajectory.py`、`examples/joint_trajectory_demo.py`、`docs/day4_log.md`

## 明日计划（Day 5：夹爪、方块与接触）

1. `examples/gripper_contact_demo.py`：actuator8 开合三档测试（0/128/255），验证双指同步（equality）
2. 读取 `data.contact`，按 geom 名称配对识别手指-方块接触，区分单/双侧
3. 方块静置 6s 稳定性测试（z 收敛 ≈0.02，无穿模无抖动）
4. 验收：开合可重复、接触可检测、方块稳定
