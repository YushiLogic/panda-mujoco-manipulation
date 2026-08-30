# Day 3 日志：可靠 reset 与 home

- 日期：2026-08-30
- 路线图任务：Day 3「可靠reset与home」
- 状态：**已完成，验收全部通过**（commit `5e538db`）

## 今日完成事项

1. **交付 `src/panda_mujoco/simulation.py`**——项目模型层（第一块地基）：
   - 常量：`PANDA_HOME_QPOS`（9 数）、`CUBE_HOME_QPOS`（7 数）、`HOME_CTRL`（8 数，前 7 格对齐 qpos）
   - `PandaScene` 类：`__init__` 加载即归位；`reset_to_home()` 幂等绝对写入（qpos → qvel 清零 → ctrl → mj_forward → 断言）；`ee_pos`/`cube_pos` 只读便捷接口（.copy() 防视图别名）
   - `_assert_valid()`：限位 + 有限性不变量检查（跳过 FREE 关节）
2. **交付 `tests/test_reset.py`**——仓库第一份 pytest 测试，5 用例全绿（0.6s）：
   - 10 次 reset 逐位一致 / joint4 合法启动 / 污染状态完全恢复 / ctrl-qpos 对齐 / 限位检查
3. **交付 `examples/panda_home_demo.py`**：viewer 中 home 姿态静置，每 2s 打印 ee/cube 坐标——**实测手肘无开机回弹、坐标稳定不漂移**
4. **学习者手写练习 `examples/viewer_mini.py`**：亲手编写第一个 viewer 程序（launch_passive + step/sync 循环），并手写 home 赋值；独立排错 4 连：不存在的 `MjRenderContextOffscreen` → 补 `import mujoco.viewer` 子模块导入 → qpos 广播长度 16 vs 9 → 自查 ctrl 对齐
5. **文档**：新增 `docs/glossary.md` 名词手册（6 分组速查表 + 已排雷清单）；更正审计表中夹爪方向（实测 0=闭合，255=张开）

## 关键数值（home 状态实测）

- home 关节角：`[0, 0, 0, -1.57079, 0, 3.0, -1.7853, 0.04, 0.04]`
- home 末端位姿（ee_center_site）：`(0.6888, 0, 0.7887)`
- home 方块位置：`(0.45, 0, 0.05)`，落地后 z≈0.02

## 关键决策与发现（后续日的输入）

1. **home 来源**：panda.xml 被注释的官方 home keyframe 的 qpos 行；但原 keyframe 的 ctrl 与 qpos 在 joint6/7 上不一致（3.0↔1.57079、-1.7853↔-0.7853），实验证明会导致开机 0.5s 内甩动 82°/57° → **HOME_CTRL 前七格改为复制 qpos**，并以 `test_home_ctrl_aligned_with_qpos` 永久把守
2. **夹爪方向实测更正**：actuator8 ctrl 0=闭合（指间距 0）、255=张开（间距 8cm）；home 双指 0.04+ctrl255 = 张开待命
3. **抓到的 bug**：`CUBE_HOME_QPOS` 初版 z 误写 0（会陷入地板），靠"reset 后立即打印实际数值"自检抓出 → 已修为 0.05
4. **术语体系成形**：MjModel/MjData、nq/nv/nu、qposadr、派生量与 mj_forward(F9)、视图 vs 副本、不变量断言——全部沉淀在 `docs/glossary.md`

## 验收对照（路线图 Day 3）

| 验收项 | 结果 |
|---|---|
| 连续 reset 10 次结果一致 | ✅ `test_reset_10_times_identical`（逐位相等）|
| joint4 不再以非法 0 位启动 | ✅ 测试断言 + demo 目视（手肘无回弹）|
| qpos/qvel/ctrl 无残留且全部有限 | ✅ 污染恢复测试 + `_assert_valid` 每次把关 |
| 记录 home 末端位姿 | ✅ (0.6888, 0, 0.7887) |

## 工程习惯建立

- 提交前必跑 pytest（全绿才 commit）；
- 数值代码"写完就打印"自检（抓到方块 z=0 bug）；
- 测试文件由 pytest 运行、不能直接 Run（考卷 vs 判卷人）；
- 浮点比较用 approx/allclose，逐位用 array_equal。

## Git 状态

- `f32802f` init → `c196ad3` README 解释器 → `5e538db` Day 2-3 全部交付物（含本日志前所有文件）
- 本日志文件待随下次提交入库

## 明日计划（Day 4：Panda 关节空间控制）

1. `examples/joint_control_demo.py`：把 UR5e 时代 JointSpaceTrajectory 思想重写为 Panda 版（不复制旧代码，重写核心逻辑）
2. 实现：平滑关节插值、position 执行器驱动、physics_dt 与 control_dt 分离意识
3. 3 组关节目标（home/左偏/右偏）依次到达；保存关节误差曲线
4. 验收：不越限、无 NaN、无持续高频抖动；连续运行稳定
