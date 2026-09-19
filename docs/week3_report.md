# Week 3 验收报告：Panda 随机方块抓取环境

## 1. 本周目标与最终结论

第三周的目标是把第二周的“末端位姿控制能力”推进为一个完整的方块抓取
任务。系统不仅要把夹爪移动到目标，还必须建立接触、抬升方块、持续稳定
保持，并对成功和失败给出可追踪结论。

本周形成的完整链路为：

```text
随机方块位姿
    ↓
生成与方块yaw对齐的抓取目标
    ↓
MOVE_ABOVE与APPROACH运动原语
    ↓
闭合夹爪并监测持续双侧接触
    ↓
LIFT并监测方块抬升高度
    ↓
CHECK_SUCCESS并验证稳定保持
    ↓
reset/step任务接口
    ↓
多seed批量验收、CSV、视频和pytest
```

Week3 验收结论：

- 已实现随机且可复现的方块位置和 yaw；
- 已实现与方块 yaw 对齐的 top-down 抓取目标；
- 已实现 MOVE_ABOVE、APPROACH、CLOSE、LIFT 等运动阶段；
- 已实现持续双侧接触、抬升和稳定性监测；
- 已实现可诊断失败阶段的脚本抓取状态机；
- 已实现29维 observation 和 reset/step 任务接口；
- seed 0～19 的20次随机抓取全部成功；
- 最小抬升高度为 `7.728 cm`；
- 最小稳定保持时间为 `0.500 s`；
- 全过程无 NaN/Inf；
- 全过程无关节限位违规；
- 全部20个 episode 为正常 terminated，无 truncated；
- 完整项目测试为 `162 passed`；
- 已生成一段成功抓取和一段受控接触失败视频。

---

## 2. 每日工作脉络

### Day 15：抓取几何与固定目标

Day15 将“抓住方块”转换为具体的几何目标。

方块尺寸为：

```text
0.04 × 0.04 × 0.04 m
```

定义三个末端目标：

```text
pregrasp：位于方块上方，夹爪张开
grasp：下降到方块中心附近，夹爪仍张开
lift：保持抓取姿态并向上抬升
```

固定 top-down 姿态要求末端局部 z 轴朝向世界 `-z`。同时审计手指关节
轴和指尖 pad 相对末端坐标系的位置，确认夹爪张合方向与方块几何一致。

完成：

```text
src/panda_mujoco/grasp_task.py
examples/day15/grasp_geometry_probe.py
examples/day15/grasp_target_ik_probe.py
tests/test_grasp_task.py
```

### Day 16：随机方块复位

Day16 建立可复现的 episode 初始状态。

采样变量为：

```text
x ∈ [0.42, 0.48] m
y ∈ [-0.06, 0.06] m
yaw ∈ [-30°, 30°]
```

方块先在 `z=0.05 m` 生成，然后通过 MuJoCo 动力学落稳到约
`z=0.0199276 m`。

free joint 的 qpos 为：

```text
[x, y, z, qw, qx, qy, qz]
```

方块 yaw 转换为 MuJoCo 的 `wxyz` 四元数。相同 seed 会生成完全相同的
初始位姿，不同 seed 通常产生不同位姿。

完成：

```text
src/panda_mujoco/cube_reset.py
examples/day16/cube_reset_probe.py
examples/day16/cube_reset_visual_demo.py
tests/test_cube_reset.py
```

### Day 17：抓取运动原语

Day17 将 IK 目标转换为真实动力学运动，而不是直接改写机械臂 qpos。

运动流程为：

```text
当前实际关节状态
    ↓
独立规划场景求6D IK
    ↓
生成关节空间线性轨迹
    ↓
逐点写入位置执行器ctrl
    ↓
偏置力补偿 + mj_step
    ↓
检查最终位置、姿态、关节误差和限位
```

最初的抓取姿态只考虑“从上向下”，没有根据随机方块 yaw 调整夹爪张合轴。
通过可视化发现夹爪与旋转后的方块没有对齐，随后将目标旋转更新为 yaw-aware
姿态，使夹爪轴与方块轴对齐。

完成：

```text
src/panda_mujoco/motion.py
examples/day17/motion_primitives_probe.py
examples/day17/motion_primitives_visual_demo.py
tests/test_motion.py
```

### Day 18：持续接触与抓取监测

Day18 将底层 `data.contact` 转换成上层任务语义。

监测分层为：

```text
ContactSensor
    ↓ 当前帧接触分类
ContactSnapshot
    ↓ 连续帧计数
BilateralContactTracker
    ↓ 融合宽度、高度和速度
GraspMonitor
    ↓
GraspStatus
```

单帧双侧接触不足以证明稳定抓取，因此要求连续接触达到0.1秒。最终成功还
要求：

- 左右手指都持续接触方块；
- 夹爪宽度处于有效范围；
- 方块抬升至少5 cm；
- 方块线速度和角速度满足稳定阈值；
- 成功条件连续保持0.5秒。

完成：

```text
src/panda_mujoco/contact.py
examples/day18/contact_persistence_probe.py
examples/day18/contact_persistence_visual_demo.py
tests/test_contacts.py
```

### Day 19：脚本抓取状态机

Day19 使用有限状态机组织完整抓取流程：

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

每个阶段记录：

```text
state
success
reason
simulation_duration
```

失败时返回：

```text
final_state=FAILED
failed_stage=<发生失败的阶段>
reason=<具体物理或规划原因>
```

状态机测试覆盖：

- 固定场景成功抓取；
- 闭合超时导致无双侧接触；
- 抬升时主动张开夹爪模拟滑落；
- 不可达目标导致 MOVE_ABOVE 的 IK 失败。

完成：

```text
src/panda_mujoco/pick_controller.py
examples/day19/scripted_pick.py
examples/day19/scripted_pick_acceptance.py
tests/test_pick_state_machine.py
```

### Day 20：轻量抓取任务接口

Day20 将抓取系统包装为：

```python
observation, info = task.reset(seed=seed)
observation, reward, terminated, truncated, info = task.step(action)
```

29维 observation 包含：

```text
7维手臂关节位置
7维手臂关节速度
1维夹爪实际宽度
3维末端位置
3维方块位置
2维yaw sin/cos
3维方块线速度
3维方块角速度
```

动作包括：

```text
WAIT
RUN_SCRIPTED_PICK
```

抓取成功或失败都是明确任务结局，因此 `terminated=True`。连续 WAIT 达到
外部步数上限时，任务没有自然结局，因此 `truncated=True`。

任务接口还修正了重复 reset 问题：`task.reset()` 已经建立当前 episode，
随后状态机使用 `reset_scene=False` 复用当前场景，不能再次悄悄随机化。

完成：

```text
src/panda_mujoco/panda_pick_task.py
examples/day20/task_reset_probe.py
examples/day20/task_wait_probe.py
examples/day20/task_scripted_pick_probe.py
examples/day20/evaluate_scripted.py
tests/test_pick_task.py
```

### Day 21：20次随机抓取验收

Day21 为任务接口增加物理步回调，使验收器和录像器能够读取每一个动力学步。

正式验收使用 seed 0～19，并在整个运动过程中检查：

- qpos、qvel、ctrl 和 cube xpos 是否有限；
- 七个手臂关节是否超过模型限位；
- 最终抓取是否成功；
- 抬升高度和保持时间是否达标；
- episode 是否 terminated 而非 truncated；
- 失败案例是否具有完整阶段与原因。

完成：

```text
examples/day21/week3_acceptance.py
examples/day21/record_week3_videos.py
results/day21/grasp_evaluation.csv
results/day21/videos/
docs/day21_log.md
```

---

## 3. 最终软件架构

```text
PandaPickTask                    任务边界
├── reset(seed)
│   ├── mj_resetData
│   ├── reset_to_home
│   ├── sample_cube_pose
│   ├── reset_cube
│   └── settle_cube
│
├── get_pick_observation         状态读取
│
└── step(action)
    ├── WAIT
    └── RUN_SCRIPTED_PICK
        └── PickController       流程决策
            ├── Motion           末端规划与动力学执行
            ├── Gripper          张开与闭合
            └── GraspMonitor     接触与成功判定
                ├── ContactSensor
                └── BilateralContactTracker
```

职责边界：

```text
控制器：决定做什么并写入控制命令
监测器：读取发生了什么
状态机：根据监测结果决定下一阶段
任务接口：管理episode、观察、奖励和结束语义
验收器：跨episode统计可靠性和安全性
```

---

## 4. Week3 正式验收结果

命令：

```powershell
python examples/day21/week3_acceptance.py
```

结果：

```text
评价episode：             20
成功episode：             20
成功率：                  100.00%
最小成功抬升高度：        7.728 cm
平均成功抬升高度：        7.753 cm
最小成功保持时间：        0.500 s
非有限状态案例：          0
关节越限案例：            0
terminated：              20
truncated：               0
失败阶段缺失：            0
完整pytest：              162 passed
```

逐回合数据：

```text
results/day21/grasp_evaluation.csv
```

验收门槛对照：

| 指标 | 要求 | 实际 | 结论 |
|---|---:|---:|---|
| 随机抓取成功率 | ≥80% | 100% | 通过 |
| 抬升高度 | ≥5 cm | 最小7.728 cm | 通过 |
| 保持时间 | ≥0.5 s | 最小0.500 s | 通过 |
| NaN/Inf案例 | 0 | 0 | 通过 |
| 关节越限案例 | 0 | 0 | 通过 |
| 失败诊断完整 | 是 | 是 | 通过 |
| 全项目测试 | 通过 | 162 passed | 通过 |

---

## 5. 视频交付

### 成功抓取

```text
results/day21/videos/week3_success_seed00.mp4
```

展示 seed 0 从 MOVE_ABOVE、APPROACH、夹爪闭合、LIFT 到稳定保持的完整过程。

### 受控失败

```text
results/day21/videos/week3_failure_contact_timeout.mp4
```

通过将 `close_timeout` 设为一个物理步，制造无法形成持续双侧接触的条件。
状态机在 VERIFY_CONTACT 返回 `no_bilateral_contact`，不会进入 LIFT。

受控失败用于展示错误处理和可诊断性，不参与20次随机成功率统计。

---

## 6. 本周关键工程判断

### 6.1 目标位姿必须考虑物体朝向

只让末端 z 轴朝下不能保证夹爪张合轴与方块边缘对齐。随机 yaw 加入后，
抓取目标旋转必须随方块 yaw 更新。

### 6.2 接触不是抓取成功

一次接触可能只是擦碰。抓取成功需要持续双侧接触、有效夹爪宽度、抬升高度
和稳定速度共同成立。

### 6.3 IK成功不是动力学成功

IK只证明存在一个满足几何目标的关节配置。真实仿真还要经过轨迹、执行器、
接触和重力，必须用实际动力学状态验收。

### 6.4 失败必须可定位

`success=False` 只说明结果不好。`failed_stage=VERIFY_CONTACT` 和
`reason=no_bilateral_contact` 才能指导修改夹爪闭合时间、目标几何或摩擦参数。

### 6.5 批量结果必须可复现

固定 seed、记录初始位姿、保存逐回合 CSV，才能在代码修改后判断性能是否真的
变化，而不是随机样本换了。

---

## 7. 测试与证据链

第三周交付不是只依赖一段“看起来成功”的动画，而是由多种证据共同支持：

```text
pytest
  验证函数、边界条件和失败路径

探针脚本
  展示几何、随机化、接触和任务接口内部数据

20次CSV
  保存不同seed的逐回合量化结果

成功视频
  展示完整动力学抓取行为

受控失败视频
  展示状态机能够拒绝无效抓取

Day日志与周报告
  记录设计依据、异常、限制和验收结论
```

---

## 8. 当前局限

本周结果仍然存在明确边界：

1. 只评估20个固定 seed；
2. 方块只随机 x、y 和 yaw；
3. 方块质量、尺寸、摩擦系数没有随机化；
4. 没有相机观测，策略直接读取模拟器状态；
5. 没有传感器噪声和控制延迟；
6. 使用模型偏置力补偿；
7. 抓取策略是固定状态机，不是学习策略；
8. 没有完成 ROS 2 或真实 Panda 部署。

因此，本周成果应描述为“可复现的 MuJoCo 脚本抓取环境和固定范围随机验收”，
不能描述为通用抓取算法或 sim-to-real 已完成。

---

## 9. 面试与作品集可展示内容

可以从以下角度介绍项目：

### 技术能力

- 使用 MuJoCo XML 和 Python API 搭建 Panda 操作环境；
- 实现 FK、Jacobian、DLS IK 和6D末端位姿控制；
- 使用执行器和偏置力补偿完成动力学轨迹跟踪；
- 处理 free joint、四元数、坐标变换和随机化；
- 基于 geom 接触构建持续抓取监测器；
- 使用有限状态机管理完整抓取任务；
- 设计 reset/step、observation、reward 和终止接口；
- 用 pytest、CSV和视频建立回归与验收流程。

### 推荐演示顺序

```text
1. 展示成功抓取视频
2. 解释状态机阶段
3. 展示受控失败视频和failed_stage
4. 展示20次CSV与验收汇总
5. 展示162项测试
6. 说明当前局限和下一步sim-to-real计划
```

---

## 10. 下周建议

第四周应从“功能完成”转向“工程质量与求职展示”：

1. 统一代码格式和公共API；
2. 增加类型检查或静态检查；
3. 为更多失败边界增加测试；
4. 增加摩擦、质量和控制延迟随机化；
5. 设计更清晰的项目架构图；
6. 完善英文 README 和演示说明；
7. 配置 GitHub Actions 自动运行 pytest；
8. 准备面试讲解稿和简历项目描述。

---

## 11. 最终结论

第三周已经完成从“末端可以到达目标”到“系统可以执行、监测、判断和批量评估
随机方块抓取”的跨越。

最终交付具有：

```text
可运行代码
可复现随机实验
成功与失败路径
自动化测试
量化CSV
可视化视频
详细实验日志
周验收报告
```

Week3 验收通过，可以发布 `v0.3-week3`。
