# v0.4-week4 Release Notes (Draft)

## Highlights

- Added Gymnasium interfaces for scripted pick and continuous Cartesian control.
- Added action mapping, reward components, termination decisions, seeded
  evaluation, and deterministic failure-scenario checks.
- Added random and scripted policy evaluation plus a Reach-only SAC training
  and paired-evaluation path.
- Documented the project architecture, reproducible setup, benchmark results,
  and current limitations in the README.

## Validation

- New Conda environment created from `environment.yml` and project installed
  with the development extra.
- Full test suite: 253 passed, with two Gymnasium warnings about unbounded
  observation-space limits.
- Headless MuJoCo environment check: passed.
- Random smoke test: 100/100 episodes passed robustness checks; task success
  was 0/100.
- Scripted expert evaluation: 20/20 task successes.
- SAC Reach held-out evaluation: 0/10 task successes; mean position and
  orientation errors were lower than the paired random baseline.

## Known limitations

- SAC is evaluated only on Reach. The gripper stays open; no learned grasp or
  lift behavior is claimed.
- The 100k-step SAC checkpoint did not meet the Reach success thresholds.
- Held-out SAC evaluation covers 10 seeds and is preliminary evidence only.
- All results use the bundled MuJoCo simulation and do not establish
  sim-to-real transfer.
- GitHub Actions status and the final staged-file list still need review.

This is a local draft. No tag or GitHub Release has been created.
