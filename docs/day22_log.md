---
date: 2026-09-22
project: panda-mujoco-manipulation
day: 22
topic: Gymnasium scripted-pick environment adapter
platform: Windows
python: 3.10.20
mujoco: 3.12.0
gymnasium: 1.3.0
gym_tests: "5 passed"
full_tests: "167 passed"
env_checker: passed_with_observation_bound_warnings
result: passed
---

# Day 22 日志：把脚本抓取任务接入 Gymnasium

## 1. 今天要解决什么

Day20 已有 `PandaPickTask.reset()` / `step()`，但它尚不是一个 Gymnasium 环境。今天增加的是**适配层** `PandaScriptedPickEnv`，让已有任务遵守 Gymnasium 的空间声明、随机种子、`reset`、`step`、`render` 和 `close` 接口。

这一步没有重写抓取状态机，也没有让智能体直接控制七个关节。当前动作仍是高层命令：

| 动作编号 | 含义 | 一次 `env.step()` 做什么 |
| --- | --- | --- |
| `0` | `WAIT` | 推进一小段仿真，不执行抓取 |
| `1` | `RUN_SCRIPTED_PICK` | 执行一次完整的脚本抓取 |

因此，今天完成的是“脚本任务的 Gym 接口”，**不是**连续关节动作的强化学习控制环境。后者需要另行定义动作与控制频率。

## 2. 新增和修改的文件

- `src/panda_mujoco/gym_env.py`：Gymnasium 适配器与可选的人类可视化。
- `tests/test_gym_env.py`：接口合同的自动化测试。
- `pyproject.toml`：加入 `gymnasium>=1.3,<2.0` 依赖。

### 空间声明

```python
observation_space = spaces.Box(
    low=-np.inf,
    high=np.inf,
    shape=(29,),
    dtype=np.float64,
)
action_space = spaces.Discrete(2)
```

观测是 29 个 `float64` 数值。`Discrete(2)` 的合法动作是 `0` 和 `1`，不是任意整数。空间对象是对外的**声明**，不会自动替 `step()` 拦截非法输入；因此适配器显式调用 `action_space.contains(action)`。

目前尚未逐维确定 29 个观测量的合理上下界，所以先使用无穷界；这也使 Gymnasium 检查器给出建议性警告，而非测试失败。后续如果需要归一化或训练策略，应先根据观测定义和物理范围确定边界，不能随意填一个看似合理的数。

### `reset(seed=...)`

1. `super().reset(seed=seed)` 初始化 Gymnasium 管理的随机数生成器。
2. 有显式 seed 时，把它交给原任务的 `reset`。
3. 未提供 seed 时，从 `self.np_random` 生成任务 seed，使随机化仍受环境随机数生成器管理。
4. 检查返回的观测是否属于 `observation_space`，最后返回 `(observation, info)`。

相同 seed 的初始观测可重复。方块位置和朝向仍由现有的随机复位逻辑产生；适配器只负责衔接接口。

### `step(action)`

1. 检查动作是否属于 `action_space`；非法动作抛出 `ValueError("invalid action: ...")`。
2. 若启用 `render_mode="human"`，先打开或同步被动 Viewer，并准备物理步回调。
3. 调用 `PandaPickTask.step(...)`，由原任务执行等待或完整抓取。
4. 检查返回观测的形状、数据类型和数值是否被观测空间接受。
5. 原样返回 `(observation, reward, terminated, truncated, info)`。

这体现了适配层的边界：它负责 Gym 合同与验证；运动、接触监测、状态机和奖励判定仍由原有任务承担。

### `terminated` 与 `truncated`

这两个布尔量不能只看成“是否结束”的两种写法：

- `RUN_SCRIPTED_PICK` 完成后，成功或失败都有明确任务结果，故 `terminated=True`、`truncated=False`。
- 连续 `WAIT` 达到默认的 5 次上层动作限制时，是时间限制，故 `terminated=False`、`truncated=True`，`info["reason"] == "time_limit"`。

一次 `WAIT` 动作内部会执行多个 `mj_step()`；`episode_steps` 统计的是上层 `env.step()` 调用次数，而不是 MuJoCo 的物理步数。

### 可视化

`render_mode="human"` 使用 MuJoCo 的被动 Viewer。物理步回调在每个仿真步后同步画面；`time.sleep(model.opt.timestep)` 用于让演示速度接近仿真时间。物理状态仍由 `mj_step()` 推进，`sleep` 本身不计算动力学。`close()` 负责关闭 Viewer。

手动演示中，完整脚本抓取动画已展示，结果为 `success=True`、`reward=1.0`、`terminated=True`。

## 3. 一次测试失败及其原因

初版专项测试有 2 项通过、1 项失败。非法动作测试期待 `invalid action`，实际得到 `action must be a valid PickAction`。

这**不是**非法动作被接受：底层 `PandaPickTask` 已拒绝动作 `2`。原因是在加入动画回调时，适配层自己的 `action_space.contains(action)` 检查被覆盖，非法动作直接传到了底层。恢复适配层检查后，第三项测试通过。这个例子说明测试错误消息也有价值：它能帮助定位是哪一层首先发现问题。

## 4. 验收结果

`tests/test_gym_env.py` 的五项检查：

1. 同 seed 复位可重复，观测属于声明空间。
2. `WAIT` 不会过早结束 episode。
3. 动作空间外的动作在适配层被拒绝。
4. 脚本抓取成功时，奖励、终止标志、`DONE` 状态和抬升高度一致。
5. 连续等待至时间限制时，返回 `truncated=True` 而非 `terminated=True`。

本地运行结果：

```text
tests/test_gym_env.py: 5 passed
full pytest suite:    167 passed
Gymnasium check_env: PASSED
```

`check_env` 同时提醒 Box 观测空间上下界分别为 `-inf` 和 `inf`。这不影响目前的接口验收，但留待后续按观测含义收紧或进行归一化。

## 5. 今天需要真正理解的三件事

1. Gymnasium 环境并不等于强化学习算法；它首先是统一的交互接口。今天只是让已有脚本任务接入这个接口。
2. `action_space` / `observation_space` 是对外合同，代码仍须在关键边界检查实际输入输出。
3. `terminated` 表示任务自然结束，`truncated` 表示外部限制导致结束；二者直接影响训练或评估时对 episode 的解释。

## 6. 下一步

Day23 可以在保留本环境的同时，单独设计更细粒度的控制动作（例如连续关节目标或末端增量）。设计前必须先确定每次 `step()` 的动作含义、物理步数、安全限制、奖励和终止条件，不能把当前的 `RUN_SCRIPTED_PICK` 误认为单步机械臂控制。
