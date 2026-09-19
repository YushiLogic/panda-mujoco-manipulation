---
exp_id: PM-B-260919-002
date: 2026-09-19
project: panda-mujoco-manipulation
day: 21
exp_type: 第三周随机抓取验收与作品集交付
platform: Windows
python: 3.10.20
mujoco: 3.12.0
evaluation_seeds: "0-19"
evaluation_cases: 20
success_count: 20
success_rate: 1.0
minimum_lift_height_m: 0.077276622070159
mean_lift_height_m: 0.0775341159776194
minimum_hold_duration_s: 0.5
non_finite_cases: 0
joint_limit_violations: 0
terminated_episodes: 20
truncated_episodes: 0
full_test_result: "162 passed in 8.42s"
video_count: 2
anomaly: true
anomaly_status: resolved
anomaly_summary: imageio编码器要求NumPy数组，初版录像器传入Pillow Image
result: passed
tags: [MuJoCo, Panda, grasping, acceptance, evaluation, video, pytest]
---

# Day 21 日志：第三周随机抓取验收与作品集交付

## 1. 今日目标

Day15～20 已经建立了完整的抓取技术链：

```text
抓取几何
  ↓
方块随机复位
  ↓
运动原语
  ↓
持续接触与抓取监测
  ↓
脚本抓取状态机
  ↓
reset/step任务接口
```

Day21 不再增加新的控制算法，而是回答四个工程问题：

1. 固定采样范围内的随机抓取成功率是多少？
2. 仿真全过程是否出现 NaN、Inf 或关节越限？
3. 失败时是否能记录发生阶段和具体原因？
4. 第三周成果能否被测试、复现、查看并上传到 GitHub？

路线表规定的硬性验收条件为：

```text
随机20次成功率 ≥ 80%
成功抬升高度 ≥ 0.05 m
成功保持时间 ≥ 0.5 s
NaN/Inf案例数 = 0
关节越限案例数 = 0
失败阶段和原因完整
全项目pytest通过
```

---

## 2. 为什么不能只看最终 success

如果只记录：

```text
success=True
```

可能掩盖以下问题：

- 运动中间出现过 NaN，最后数值偶然恢复；
- 中间关节曾越限，最终又回到范围内；
- 方块只抬高了几毫米；
- 抓住方块但没有稳定保持0.5秒；
- 失败案例只有 False，没有失败阶段和原因；
- episode 因步数上限截断，却被误当成正常任务失败。

因此 Day21 的验收单位不是一个最终布尔值，而是一条完整 episode 记录。

---

## 3. 物理步回调接口

### 3.1 为什么增加回调

Day20 的：

```python
task.step(PickAction.RUN_SCRIPTED_PICK)
```

一次会执行数千个 `mj_step()`。如果只在 step 返回后检查状态，只能看到
最终时刻，无法证明运动过程始终安全。

因此 `PandaPickTask.step()` 增加可选参数：

```python
task.step(
    action,
    step_callback=callback,
)
```

调用顺序为：

```text
控制器写入ctrl
      ↓
mujoco.mj_step()
      ↓
step_callback(scene)
```

回调发生在物理步完成后，因此读取到的是本步动力学更新后的实际状态。

### 3.2 回调不属于控制器

Day21 的回调只读取状态，不会：

- 改写 `qpos`；
- 改写 `qvel`；
- 改写 `ctrl`；
- 决定状态机跳转；
- 额外调用 `mj_step()`。

它是验收器和录像器接入仿真过程的只读观察点。

### 3.3 自动测试

新增两项测试：

```text
WAIT包含4个物理步 → 回调必须被调用4次
完整抓取 → 回调次数大于0且读取到的状态始终有限
```

Day20任务接口测试数量由19项增加到21项。

---

## 4. EpisodeSafetyMonitor

批量验收脚本定义了只读监测器：

```python
EpisodeSafetyMonitor
```

每个物理步检查：

```python
scene.data.qpos
scene.data.qvel
scene.data.ctrl
scene.data.body("cube").xpos
```

只要任意元素为 NaN 或 Inf，就设置：

```python
non_finite_detected = True
```

这个标志一旦变成 True，不会因为后续状态恢复而清零。

### 4.1 关节限位检查

监测器按名称查询七个 Panda 手臂关节：

```python
ARM_JOINT_NAMES
```

再从模型读取：

```python
model.jnt_qposadr
model.jnt_range
```

对于每个关节，计算：

```text
下界超出量 = lower_limit - q
上界超出量 = q - upper_limit
```

最大正值就是当前限位超出量。如果超过数值容差 `1e-9 rad`，记录：

```python
joint_limit_violation = True
```

没有硬编码“前七个 qpos 就一定永远是手臂关节”，而是根据模型名称和地址
查询，降低模型结构调整后读错下标的风险。

---

## 5. 批量验收流程

正式命令：

```powershell
& "C:\Users\29391\.conda\envs\mujoco\python.exe" `
  "examples\day21\week3_acceptance.py"
```

每个 seed 执行：

```text
task.reset(seed)
      ↓
记录初始cube位置和yaw
      ↓
创建EpisodeSafetyMonitor
      ↓
task.step(RUN_SCRIPTED_PICK, step_callback=monitor)
      ↓
保存结果、监测信息和状态轨迹
      ↓
进入下一个seed
```

使用同一个 `PandaPickTask` 对象连续 reset，可以同时验证 episode 之间不会
残留上一回合的终止状态或物理状态。

---

## 6. CSV字段

正式结果文件：

```text
results/day21/grasp_evaluation.csv
```

逐回合字段分为五组。

### 6.1 初始条件

```text
case
seed
cube_x_m
cube_y_m
cube_z_m
cube_yaw_deg
```

### 6.2 任务接口结果

```text
reward
terminated
truncated
success
final_state
```

### 6.3 失败诊断

```text
failed_stage
reason
failure_category
stage_trace
```

### 6.4 抓取物理指标

```text
lift_height_m
hold_duration_s
gripper_width_m
```

### 6.5 安全性和性能指标

```text
initial_observation_finite
final_observation_finite
non_finite_detected
joint_limit_violation
maximum_joint_limit_excess_rad
physics_steps_observed
simulation_duration_s
wall_duration_s
```

即使某一回合抛出 Python 异常，脚本也会生成字段完整的异常记录，然后继续
后续 seed。这样不会因为一个失败案例丢失整批实验数据。

---

## 7. 20次随机抓取结果

使用 seed：

```text
0, 1, 2, ..., 19
```

方块采样范围为：

```text
x ∈ [0.42, 0.48] m
y ∈ [-0.06, 0.06] m
yaw ∈ [-30°, 30°]
```

验收汇总：

```text
success count:              20/20
success rate:               100.00%
minimum successful lift:    7.728 cm
minimum successful hold:    0.500 s
non-finite cases:           0
joint-limit violations:     0
terminated episodes:        20
truncated episodes:         0
failure categories:         {}
failed stages:              {}
incomplete failure records: 0
mean wall time/case:        0.9557 s
```

其他统计：

```text
平均抬升高度：7.753 cm
最小抬升案例：seed 4
最大抬升案例：seed 12
平均每回合监测物理步数：5126.15
```

最小抬升为：

```text
0.0772766221 m > 0.05 m
```

因此第三周全部数值门槛通过。

---

## 8. 如何解释100%成功率

本次结果可以准确表述为：

> 在当前模型、控制参数和方块采样范围下，固定 seed 0～19 的20个仿真
> episode 全部完成脚本抓取，且未检测到非有限状态或关节越限。

不能表述为：

> Panda 在任何方块位置都能100%抓取成功。

原因包括：

- 样本只有20个；
- x/y/yaw范围较保守；
- 方块尺寸、质量和摩擦没有随机化；
- 使用理想状态读取；
- 没有传感器噪声、控制延迟和模型误差；
- 没有真实机器人硬件误差。

因此这是一项确定性仿真回归基准，不是完整鲁棒性或 sim-to-real 证明。

---

## 9. 两段演示视频

### 9.1 成功抓取

文件：

```text
results/day21/videos/week3_success_seed00.mp4
```

参数：

```text
seed=0
正常close_timeout=1.0 s
```

结果：

```text
success=True
final_state=DONE
state trace:
RESET → MOVE_ABOVE → APPROACH → CLOSE_GRIPPER
→ VERIFY_CONTACT → LIFT → CHECK_SUCCESS → DONE
```

视频共391帧，30 FPS。最终画面中方块被夹爪稳定抬离地面。

### 9.2 典型受控失败

文件：

```text
results/day21/videos/week3_failure_contact_timeout.mp4
```

参数：

```text
seed=0
close_timeout=0.002 s
```

模型时间步为0.002 s，因此夹爪只获得一个物理步的闭合时间，无法形成持续
双侧接触。结果为：

```text
success=False
final_state=FAILED
failed_stage=VERIFY_CONTACT
reason=no_bilateral_contact
```

视频共297帧，30 FPS。最终画面中方块仍在地面，状态机没有错误进入 LIFT。

这段失败由明确的故障注入产生，只用于验证失败路径和解释状态机行为，不属于
20次自然随机抓取基准，因此不应加入成功率分母。

---

## 10. 视频技术实现

录像器使用：

```text
mujoco.Renderer
imageio
imageio-ffmpeg
Pillow
```

MuJoCo 每个物理步为0.002 s，而视频为30 FPS。录像器约每17个物理步抽取
一帧：

```text
1 / (30 × 0.002) ≈ 16.67
```

这样视频时间接近仿真时间，同时避免为每个物理步编码一帧造成文件过大。

输出采用：

```text
H.264 / yuv420p / 640×480 / 30 FPS
```

这套格式适合常见浏览器和 GitHub 下载预览。画面顶部显示场景类型、seed
和当前仿真时间。

### 10.1 遇到的问题

初版标注函数返回：

```python
PIL.Image
```

但 `imageio.writer.append_data()` 要求：

```python
numpy.ndarray
```

出现错误：

```text
ValueError: append_data requires ndarray as first arg
```

修正为：

```python
return np.asarray(image)
```

重新运行后两段视频均成功编码，并抽取代表帧做了视觉检查。

---

## 11. 全项目回归测试

运行：

```powershell
& "C:\Users\29391\.conda\envs\mujoco\python.exe" -m pytest -q
```

结果：

```text
162 passed in 8.42s
```

覆盖模块包括：

- 机械臂关节控制；
- 笛卡尔轨迹；
- 接触和抓取监测；
- 方块复位和随机化；
- 抓取目标几何；
- 夹爪控制；
- 位置和六维姿态 IK；
- 正运动学与 Jacobian；
- 运动原语；
- 脚本抓取状态机；
- reset/step任务接口；
- 旋转误差和可靠复位。

---

## 12. Day21代码交付

新增：

```text
examples/day21/week3_acceptance.py
examples/day21/record_week3_videos.py
results/day21/grasp_evaluation.csv
results/day21/videos/week3_success_seed00.mp4
results/day21/videos/week3_failure_contact_timeout.mp4
docs/day21_log.md
docs/week3_report.md
```

修改：

```text
src/panda_mujoco/panda_pick_task.py
tests/test_pick_task.py
pyproject.toml
environment.yml
README.md
```

其中视频编码依赖已经同时写入 `pyproject.toml` 和 `environment.yml`，避免
代码只能在当前电脑运行。

---

## 13. Day21需要掌握的内容

完成今天后，应能解释：

1. 为什么批量验收必须固定 seed；
2. 为什么20/20不能推广为整个工作空间100%；
3. 为什么只检查最终状态不足以证明运动安全；
4. 物理步回调在控制器、验收器和录像器之间起什么作用；
5. 如何通过模型名称读取关节限位，而不是硬编码下标；
6. 为什么异常案例也必须留下CSV记录；
7. terminated 与 truncated 在批量评估中的意义；
8. 为什么受控失败案例不能混入随机成功率；
9. 如何使用 failed_stage 和 reason 定位状态机失败；
10. 如何用测试、CSV、视频和日志共同形成可复现证据链。

---

## 14. 今日结论

第三周抓取系统已经形成闭环：

```text
设计抓取目标
  ↓
随机化任务初始状态
  ↓
通过动力学执行运动和夹爪控制
  ↓
持续监测接触、抬升与稳定性
  ↓
状态机处理成功与失败
  ↓
任务接口输出observation/reward/termination/info
  ↓
多seed验收、CSV、视频、pytest和日志
```

当前固定基准全部通过，可以进入第四周的工程整理、强化测试和作品集展示。
