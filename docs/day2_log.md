# Day 2 日志：Panda 模型审计

- 日期：2026-08-29（实际学习时间跨越 8-29 深夜至 8-30）
- 路线图任务：Day 2「Panda模型审计」
- 状态：**已完成，验收全部通过**

## 今日完成事项

1. **Git 三区练习**（Codex Day 1 收尾建议）：亲手走完 `reset --soft` → `restore --staged` → 重新 add/commit 全流程，理解工作区/暂存区/提交记录。练习中顺手在 README 补记了 conda 解释器路径（commit `c196ad3`），补上 Day 1「解释器固定并记录」的证据。
2. **模型审计交付物**（详见下节）。
3. **交互式查看器建立直觉**：`python -m mujoco.viewer --mjcf=scene_with_cube.xml` 实机观察了 Panda 默认姿态、方块下落、手肘关节启动瞬间回弹（joint4 非法起始的现场表现）。

## 交付物

| 文件 | 说明 |
|---|---|
| `examples/inspect_panda.py` | 完整审计脚本：名称体检（无 -1 ID）、维度拆解、关节/执行器清单、正运动学位姿、隐患警告；中文注释；退出码 0/1 可供 pytest 使用 |
| `examples/inspect_panda_mini.py` | 约 60 行极简对照版（学习用）：查名字 → 打印维度 → joint4 检查 → 打印位姿，无任何工程包装 |
| `docs/panda_model_audit.md` | 模型审计表：维度推导、关节/执行器清单、名词速查、关键发现 |

## 审计核心结论

- 场景 `scene_with_cube.xml` 编译通过：**nq=16, nv=15, nu=8**。
  - nq = 9 个单值关节（7 HINGE + 2 SLIDE）+ cube 自由关节 7（xyz+四元数）；nv = 9 + 6；nq−nv=1 源自四元数
- 末端 site：场景唯一 site 为 `ee_center_site`，qpos0 下位于 (0.0880, 0.0000, 0.8210)
- cube：初始 (0.45, 0, 0.05)，静置后 z≈0.02（半边长 0.02 m，condim=4，摩擦 1.0/0.115/0.0001）
- 模型为三层结构：scene（环境+cube）→ `<include>` panda.xml（机器人）→ assets 网格；nu 按编译合并后统计
- 关节类型：HINGE×7、SLIDE×2、FREE×1（cube_joint）；FREE=6 自由度自由刚体，无关节限位

## 关键发现（Day 3 输入）

1. **joint4 默认 qpos0=0 超出限位 [-3.0718, -0.0698]（肘关节不能伸直），是全模型唯一非法默认关节**。viewer 中可见启动瞬间回弹。`panda.xml` 内有现成但被注释的 home keyframe（约 286-288 行），候选 PANDA_HOME：
   - qpos `0 0 0 -1.57079 0 3.0 -1.7853 0.04 0.04`（前 7 轴 + 双指 0.04）
   - 对应 ctrl `0 0 0 -1.57079 0 1.57079 -0.7853 255`
2. **夹爪执行器 actuator8 量纲为 0~255**（tendon `split` 分力驱动双指，equality 同步），与手臂弧度量纲并存，控制代码需分开处理。
3. 场景无 `target_site`——后续放置任务需在**场景层**（scene_with_cube.xml）添加，不改 panda.xml（分层原则）。

## 验收对照（路线图 Day 2）

| 验收项 | 结果 |
|---|---|
| 所有名称查询有效且无 -1 ID | ✅ `AUDIT PASSED` |
| 能解释 nq=16、nv=15、nu=8 | ✅ 见审计表推导 |
| 末端和方块位置可打印 | ✅ 脚本输出 + viewer 目视核对 |

## 附加学习（超出当日计划）

- Git 三区模型实操（status/diff/--cached/commit/reset/restore）
- MjModel（不变的结构规则）vs MjData（当前时刻唯一快照，无历史）概念；两份 MjData 共享同一 MjModel 实验
- xml 必备段只有 worldbody，actuator/sensor 等全部可选（nu=0 模型可编译）

## Git 状态

- 提交：`f32802f` init → `c196ad3` README 补解释器路径
- 待提交：`examples/inspect_panda.py`、`examples/inspect_panda_mini.py`、`docs/panda_model_audit.md`、`docs/day2_log.md`（本次一并提交）

## 明日计划（Day 3：可靠 reset 与 home）

1. `src/panda_mujoco/simulation.py`：实现 `reset_to_home()`——显式写 9 关节 qpos（含 joint4=-1.57079）、双指、cube 初态，清零 qvel/ctrl，调用 `mj_forward`，加限位断言
2. `examples/panda_home_demo.py`：viewer 中验证 home 姿态
3. 测试：连续 reset 10 次状态完全一致（qpos/qvel/ctrl 全有限且相等）
4. 验收：joint4 不再以非法 0 位启动；无 NaN
