---
experiment_id: PANDA-MJ-SIM-260928-002
date: 2026-09-28
project: panda-mujoco-manipulation
day: 25
topic: Environment robustness, random-action smoke testing, and deterministic failure scenarios
platform: Windows
python: 3.10.20
mujoco: 3.12.0
gymnasium: 1.3.0
random_episodes: 100
max_steps_per_episode: 100
failure_scenarios: 5 passed
day25_tests: 8 passed
full_tests: 231 passed
result: passed
---

# Day 25 Log: Environment Robustness and Failure Scenarios

## 1. Objective

Day 24 added reward calculation and episode termination/truncation to the continuous-control environment. Day 25 checked whether that environment remains numerically and physically well-formed under many actions, and whether important task failures are reported at the correct stage.

The main goals were:

1. Check that `PandaContinuousEnv` follows Gymnasium's reset/step API contract.
2. Run reproducible random-action episodes to expose crashes, invalid observations, non-finite rewards, or arm joint-limit violations.
3. Inject representative failures into the scripted pick pipeline and verify that the state machine stops safely with the expected diagnosis.
4. Add these checks to the automated test suite and define a headless GitHub Actions workflow.

This is robustness and failure-handling work, not policy training. Random actions are not expected to solve the pick task.

## 2. Gymnasium Environment Contract

Implementation under test: `src/panda_mujoco/continuous_env.py`.

`gymnasium.utils.env_checker.check_env(env, skip_render_check=True)` completed successfully. The render check is skipped because automated and CI runs may not have a display; rendering was already exercised separately through visual demos.

The environment uses a seven-dimensional continuous action and a 29-dimensional observation. The observation consists of numeric state values:

| Indices | Contents | Units / representation |
| --- | --- | --- |
| `0:7` | Panda arm joint positions | rad |
| `7:14` | Panda arm joint velocities | rad/s |
| `14` | Gripper width | m |
| `15:18` | End-effector world position | m, xyz |
| `18:21` | Cube world position | m, xyz |
| `21:23` | Cube yaw encoded as sine and cosine | dimensionless |
| `23:26` | Cube linear velocity | m/s, xyz |
| `26:29` | Cube angular velocity | rad/s, xyz |

The observation space is declared as `Box(-inf, +inf, shape=(29,))`. Gymnasium therefore prints two warnings that its declared lower and upper bounds are infinite. A reset observation was independently checked and had shape `(29,)`, dtype `float64`, only finite values, and was contained in the observation space. Thus, the warnings concern the broad space declaration; they do not mean that the observed state contains infinities. The bounds were not narrowed arbitrarily during this task.

## 3. Random-Action Smoke Test

Implementation: `examples/day25/random_smoke.py`.

The script runs seeded episodes and saves one summary row per episode to CSV. It seeds both the environment reset and the action-space sampler so a repeated seed reproduces the reset and sampled action sequence.

During each episode, the smoke test checks that:

- sampled actions are finite and belong to `action_space`;
- observations are finite and belong to `observation_space`;
- the seven arm joint positions are finite and within their model joint limits, allowing a numerical tolerance of `1e-6` rad;
- rewards are finite;
- each episode ends at its configured step limit rather than silently running past it.

Command used:

```powershell
python examples\day25\random_smoke.py --episodes 100 --max-steps 100 --seed-start 0 --csv results\day25\random_smoke.csv
```

Observed result:

```text
Episodes passed robustness checks: 100/100
Task successes under random actions: 0/100
Random smoke test: PASSED
```

`100/100` means that all 100 episodes completed without violating the smoke-test invariants. `0/100` means that none of the random policies completed a successful pick. These are different metrics: the first is environment robustness, while the second is task performance. A timeout or unsuccessful grasp is an ordinary episode outcome and does not count as a smoke-test failure unless the environment becomes invalid.

The detailed episode summaries are saved in `results/day25/random_smoke.csv`.

## 4. Deterministic Failure Scenarios

Implementation: `examples/day25/failure_scenarios.py`.

Each scenario checks both the diagnosis and that unsafe later states are not visited. The command completed with `Scenario checks: 5/5`.

| Scenario | Injected condition | Observed result | Safety assertion |
| --- | --- | --- | --- |
| Unreachable target | Replace the generated grasp poses with positions far outside the robot workspace | Fails in `MOVE_ABOVE`, reason `ik_failed` | Does not proceed to `APPROACH`, `LIFT`, or `DONE` |
| No bilateral contact | Give gripper closing only `0.002 s` to establish contact | Fails in `VERIFY_CONTACT`, reason `no_bilateral_contact` | Does not proceed to `LIFT`, `CHECK_SUCCESS`, or `DONE` |
| Single-sided contact | Construct a contact snapshot with left contact only | `bilateral_contact=False` | A single finger is not misclassified as a two-sided grasp |
| Contact loss during lift | Open the gripper once the cube has started rising | Fails in `LIFT`, reason `no_bilateral_contact` | Does not proceed to `CHECK_SUCCESS` or `DONE` |
| Time limit | Set the continuous environment limit to one action step | `terminated=False`, `truncated=True`, reason `time_limit` | A timeout is represented as truncation, not task success or a crash |

The unreachable-target case replaces the target generator temporarily and restores it in a `finally` block. The no-contact case uses the actual scripted pick and a deliberately short close timeout. The contact-loss case uses a step callback to open the gripper after lift begins. The single-sided case tests the contact classifier using a constructed `ContactSnapshot`; it does not claim to reproduce unilateral contact through physical simulation.

## 5. Automated Tests

New Day 25 integration tests are in `tests/test_environment.py`. They cover:

- Gymnasium API compliance;
- all five deterministic failure scenarios;
- treating a time-limit episode as a robustness pass but not a task success;
- repeating the same random-smoke seed and obtaining the same episode summary.

The dedicated Day 25 test file passed all 8 tests. The broader regression group for the pick state machine, contacts, and continuous environment passed 44 tests. The full project suite passed:

```text
231 passed, 2 warnings
```

The two warnings are Gymnasium's warnings about the unbounded observation-space limits described above.

## 6. Continuous Integration

Workflow definition: `.github/workflows/ci.yml`.

The workflow is configured to run on pushes to `main`, pull requests targeting `main`, and manual dispatch. It uses Ubuntu and Python 3.10, installs the package with development dependencies, runs the full pytest suite, and then runs 100 seeded random-action smoke episodes. The smoke-test CSV is written to the runner's temporary directory.

The workflow file was prepared locally, but a GitHub Actions run has not yet been observed. It will execute after the weekly changes are pushed or when manually dispatched; local test success should not be described as a completed remote CI run.

## 7. Results and Interpretation

- The Gymnasium API checker accepted the environment (render validation intentionally skipped for headless use).
- The reset observation was finite, correctly shaped, and contained in the declared observation space.
- All 100 seeded random-action episodes passed robustness checks; none achieved a successful pick, as expected for random control.
- All five injected failure scenarios were diagnosed at the intended state and did not advance into forbidden later states.
- All 231 project tests passed locally, with only the two known observation-bound warnings.

The important outcome is not “the robot learned to pick.” It is that the environment can survive a batch of untrained actions and that known failures produce inspectable, stage-specific results instead of being mistaken for success.

## 8. Limitations and Next Work

- Random smoke testing checks invariants, not controller quality or task success rate.
- The observation space still uses infinite bounds. This is valid for the present numeric observations but causes Gymnasium warnings and may be refined later using justified physical bounds or normalized features.
- The single-sided contact case validates classification from a snapshot; it is not a physical unilateral-contact experiment.
- The GitHub Actions workflow has not yet been executed remotely.
- No learned policy was trained in Day 25.

## 9. Files for Day 25

| File | Role |
| --- | --- |
| `examples/day25/random_smoke.py` | Seeded random-action robustness batch and CSV output. |
| `examples/day25/failure_scenarios.py` | Deterministic failure injection and expected state-machine outcome checks. |
| `results/day25/random_smoke.csv` | Results from 100 random-action episodes. |
| `tests/test_environment.py` | Gymnasium contract, failure-scenario, timeout, and reproducibility tests. |
| `.github/workflows/ci.yml` | Headless GitHub Actions test and smoke-test workflow. |
