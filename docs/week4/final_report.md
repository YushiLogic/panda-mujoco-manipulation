# Week 4 Final Report

Date: 2026-09-29

## Goal

Week 4 turned the scripted Panda pick task into a Gymnasium-compatible
environment, defined continuous actions and reward/termination behavior, and
compared random, scripted, and SAC Reach policies. The objective was a
reproducible engineering baseline, not a claim that reinforcement learning had
solved grasping.

## Environment and interface

- `PandaScriptedPickEnv` exposes the scripted pick episode through Gymnasium.
  Its discrete action `0` waits for one step and action `1` runs the scripted
  pick task.
- `PandaContinuousEnv` exposes a 7D normalized Cartesian/gripper action and a
  29-value observation. It reuses the project's pose IK, joint control, and
  MuJoCo simulation rather than replacing those components.
- `PandaReachEnv` exposes a 6D Cartesian action for the SAC Reach experiment.
  The gripper remains open; this environment does not train grasping or lifting.
- Each continuous control action advances 10 physics steps at a 0.002 s
  timestep, or approximately 0.02 s of simulated time.
- Reward components cover position progress, orientation progress, persistent
  contact, lift progress, success, and action effort. Task success and failure
  are represented separately from reward. `terminated` and `truncated` retain
  their distinct meanings.

## Verification in a new environment

A separate Conda environment was created from `environment.yml`; the package
was installed from the repository with the development extra.

| Check | Result |
|---|---|
| Python | 3.10.21 |
| MuJoCo | 3.12.0 |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| Full pytest suite | 253 passed |
| Headless MuJoCo check | Passed; 2.0 s simulated, ball height 0.0496 m |

The test run reported two Gymnasium advisory warnings from
`check_env`: the continuous observation `Box` uses unbounded lower and upper
limits (`-inf` and `+inf`). This is not a test failure. Observations produced
by the tested episodes were finite and passed `observation_space.contains`.
Finite limits were not guessed because the environment does not define
validated bounds for every velocity field.

## Policy evaluation

### Random robustness smoke test

The 100-episode random-action smoke test passed its robustness checks in all
100 episodes. It had 0 task successes. A smoke-test pass means the environment
continued to return valid data without crashing; it does not mean a random
policy completed the manipulation task.

### Scripted expert baseline

The Day 26 evaluation ran 20 episodes per policy:

| Policy | Task successes | Interpretation |
|---|---:|---|
| Random continuous | 0/20 | All episodes reached the time limit |
| Scripted expert | 20/20 | Scripted state machine completed the task |

### SAC Reach-only experiment

The SAC model was trained for 100,000 environment steps. Evaluation uses the
same reset seed for the random and SAC runs in each pair. The 10 final test
seeds were held out from the earlier checkpoint comparison.

| Policy | Mean final position error | Mean final orientation error | Reach successes |
|---|---:|---:|---:|
| Random | 788.61 mm | 150.90° | 0/10 |
| SAC, 100k steps | 85.99 mm | 72.53° | 0/10 |

The SAC model reduced both average errors on these 10 seeds, but every episode
still ended at the 500-step time limit. The policy showed useful reaching
behavior but did not reach the defined success threshold. These results do not
show that SAC learned to close the gripper, maintain contact, grasp, or lift the
cube. Ten test seeds are also not enough to establish broad generalization.

The detailed training-budget comparison and per-seed measurements are recorded
in `../day27_log.md` and `../../results/day27/`.

## Conclusions and limitations

1. The project now has reusable Gymnasium interfaces for scripted pick and
   continuous Cartesian control.
2. The scripted expert remains the successful manipulation baseline. The
   random policy is a robustness baseline, not a competent controller.
3. The SAC checkpoint learned a measurable Reach improvement over random
   actions, but its final held-out success rate was 0/10.
4. The SAC task is Reach-only and keeps the gripper open. It is not an RL pick
   policy and does not demonstrate learned grasping.
5. All results are from the bundled MuJoCo model and simulation. They do not
   establish sim-to-real transfer or real-robot safety.

## Release checklist status

- [x] Clean Conda environment can install the project.
- [x] Full local test suite passes.
- [x] Headless MuJoCo quick check passes.
- [x] README includes the architecture, API example, metrics, and limitations.
- [x] Local release-notes draft prepared.
- [ ] Review the two advisory warnings and decide whether to retain unbounded
      observation limits or define validated finite limits.
- [ ] Verify the GitHub Actions workflow after the Week 4 commit is pushed.
- [ ] Review the staged file list, especially generated `output/` and `tmp/`
      directories, before committing.
- [ ] Create the `v0.4-week4` tag and GitHub Release after local review and
      explicit release approval.

No commit, tag, push, or GitHub Release was created as part of this report.
