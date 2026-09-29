---
experiment_id: PANDA-MJ-SIM-260928-001
date: 2026-09-28
project: panda-mujoco-manipulation
day: 24
topic: Reward calculation and episode termination for continuous pick control
platform: Windows
python: 3.10.20
mujoco: 3.12.0
gymnasium: 1.3.0
control_period_s: 0.02
frame_skip: 10
max_episode_steps: 500
targeted_tests: 40 passed
full_tests: 223 passed
result: passed_with_initial_reward_weights
---

# Day 24 Log: Reward Calculation and Episode Termination

## 1. Objective

Day 23 established `PandaContinuousEnv`, whose seven-dimensional action incrementally changes the end-effector pose and gripper-width targets. At the beginning of Day 24, the environment still returned a constant reward of `0.0` and placeholder `terminated=False`, `truncated=False` values.

Day 24 added reward components that describe progress toward a grasp and explicit episode-ending rules. The continuous environment now compares state before and after an action, returns the summed reward, exposes its component breakdown in `info`, and distinguishes task termination from a step-limit truncation.

This is environment/reward engineering. It does not mean a policy has been trained or has learned to pick the cube.

## 2. Reward Functions

Implementation: `src/panda_mujoco/rewards.py`.

The reward functions are pure calculations: they take numeric state values and return scores without reading or modifying MuJoCo state. The two progress functions use a clipped change normalized by a scale:

```text
error progress = clip((previous_error - current_error) / scale, -1, 1)
increase progress = clip((current_value - previous_value) / scale, -1, 1)
```

The first is for values where smaller is better, such as distance to a grasp target or orientation error. The second is for values where larger is better, such as cube lift height. The current scales are 2 mm for position/lift progress and 2 degrees for orientation progress.

`RewardWeights` applies initial weights to the normalized components:

| Component | Meaning | Initial weight |
| --- | --- | ---: |
| `reach_progress` | Change in end-effector distance to the cube grasp target | 1.0 |
| `orientation_progress` | Change in end-effector orientation error | 0.25 |
| `contact_bonus` | Newly established persistent bilateral finger contact | 0.5 |
| `lift_progress` | Change in cube lift height | 1.0 |
| `success_bonus` | Explicit successful grasp result | 5.0 |
| `action_penalty` | Negative mean squared normalized action | 0.01 |

These are starting values for the project, not tuned training parameters. The contact bonus is an event bonus: it is paid when persistent bilateral contact changes from false to true, not on every subsequent step that contact remains true. This prevents repeatedly collecting the same contact reward while stationary.

`RewardBreakdown` stores the weighted contribution from each component, checks that they are finite, calculates `total`, and exposes the components as a dictionary for Gymnasium's `info` field. `compute_reward()` assembles the components. Success is passed as an explicit Boolean; it is not inferred from the numeric total reward.

## 3. Reward Probe Results

Implementation: `examples/day24/reward_probe.py`.

The probe uses synthetic state transitions to isolate the formulas; it does not run MuJoCo. Its scenarios produced:

| Scenario | Reach | Contact | Lift | Success | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Move closer | +0.500 | 0.000 | 0.000 | 0.000 | +0.500 |
| Move farther | -1.000 | 0.000 | 0.000 | 0.000 | -1.000 |
| Persistent contact begins | 0.000 | +0.500 | 0.000 | 0.000 | +0.500 |
| Persistent contact continues | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Cube lifted by 2 mm | 0.000 | 0.000 | +1.000 | 0.000 | +1.000 |
| Cube lift height falls from 20 mm to 0 | 0.000 | 0.000 | -1.000 | 0.000 | -1.000 |
| Task success flag is true | 0.000 | 0.000 | 0.000 | +5.000 | +5.000 |

The drop row is only a lift-progress score. It does not by itself end an episode. The probe does not invoke termination logic, and its label is not a state signal. In the environment, a drop must be detected from task state and passed to the termination decision separately.

The single-action environment smoke run used seed 42 and action `[0, 0, -1, 0, 0, 0, 0]`. The reach error changed from `0.795521360 m` to `0.795290145 m`, a reduction of about `0.231 mm`. The reported reward was:

```text
reach_progress:       +0.115607682
orientation_progress: -0.000320959
contact_bonus:         0.000000000
lift_progress:         approximately 0
success_bonus:         0.000000000
action_penalty:       -0.001428571
total reward:          0.113858151
terminated:            False
truncated:             False
reason:                none
```

This illustrates that a positive one-step reward means the state transition made net progress under the current formula; it does not mean the pick succeeded.

## 4. Termination and Truncation

Implementation: `src/panda_mujoco/termination.py`.

`decide_termination()` returns a `TerminationDecision` with `terminated`, `truncated`, `success`, and `reason` fields.

| Situation | `terminated` | `truncated` | Reason |
| --- | ---: | ---: | --- |
| Episode still running | False | False | `none` |
| Explicit grasp success | True | False | `success` |
| Explicit unrecoverable failure, such as a detected cube drop | True | False | failure reason |
| Maximum action-step count reached without success/failure | False | True | `time_limit` |

Success takes precedence over the time limit if both occur on the same step. An explicitly unrecoverable failure also ends the task; an ordinary rejected IK target remains recoverable and does not by itself end the episode.

For the continuous environment, the current drop heuristic remembers whether the cube has ever reached the monitor's lifted threshold. If it has, and a later monitored state reports cube-floor contact, the environment marks `cube_dropped` as the failure reason. The starting cube-floor contact after reset is not a drop because `ever_lifted` begins as false.

## 5. Continuous Environment Integration

Implementation: `src/panda_mujoco/continuous_env.py`.

- `reset()` creates a `GraspMonitor` using the settled cube height as the lift reference and clears the episode's lift/termination flags.
- The monitor is updated after every MuJoCo `mj_step()` within `frame_skip`, so bilateral-contact persistence is measured at physics-step resolution.
- `_get_task_errors()` constructs yaw-aligned grasp targets from the cube's current pose and measures end-effector distance/orientation error relative to those task targets.
- `step()` saves the prior task errors and `GraspStatus`, applies one action, advances the physics, reads the new task state, computes reward components, and calls `decide_termination()`.
- The default action-step limit is 500. At 0.02 s of simulated time per environment action, that is 10 s of simulated time.
- The Gymnasium return is `(observation, reward_total, terminated, truncated, info)`. `info` includes the component dictionary under `reward_breakdown`, plus task errors, grasp status, drop/success state, IK diagnostics, and control tracking diagnostics.
- `tracking_error_m` measures distance to the last controller target. `task_reach_error_m` measures distance to the cube's grasp target. They answer different questions.
- Calling `step()` after an episode has terminated or truncated raises an error; a new `reset()` is required.

## 6. Tests and Acceptance

Added/updated tests include `tests/test_rewards.py`, `tests/test_termination.py`, and `tests/test_continuous_env.py`.

The targeted Day 24 group passed 40 tests. The full project suite then collected and passed all 223 tests:

```text
223 passed in 24.76s
```

Coverage includes reward direction and clipping, finite-input validation, contact-event reward behavior, reward-component sums, success/time-limit precedence, failure versus timeout, environment reward reporting, time-limit truncation, and refusing additional actions after the episode ends.

## 7. Limitations and Next Work

- The weights and scales are initial engineering choices. They have not been tuned by policy training or evaluated across a reward ablation study.
- The current reward has no separate `drop_penalty` component. A dropped cube receives the lift-progress change reward, while the termination result independently records task failure. The displayed `total` is therefore not a universal success/failure label.
- The cube-drop detector is a task heuristic built from `ever_lifted` plus current floor contact; more robust tasks may require persistent floor-contact confirmation or additional slip/velocity checks.
- No policy was trained in Day 24. The work prepares a continuous-control environment with observable progress and episode boundaries for later learning and evaluation.

## 8. Files for Day 24

| File | Role |
| --- | --- |
| `src/panda_mujoco/rewards.py` | Reward components, weights, breakdown, and unified reward calculation. |
| `src/panda_mujoco/termination.py` | Pure episode termination/truncation decision. |
| `src/panda_mujoco/continuous_env.py` | Reward, monitor, and termination integration. |
| `examples/day24/reward_probe.py` | Synthetic reward-component scenarios. |
| `tests/test_rewards.py` | Reward formula and validation tests. |
| `tests/test_termination.py` | Termination semantics and validation tests. |
| `tests/test_continuous_env.py` | Environment reward reporting and time-limit integration tests. |
