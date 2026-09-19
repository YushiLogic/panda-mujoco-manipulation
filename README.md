# Panda MuJoCo Manipulation

A reproducible MuJoCo project for learning robotic manipulation with the Franka Emika Panda robot.

This portfolio project covers joint-space control, end-effector control, inverse kinematics, grasping, testing, and experiment documentation.

## Current Status

- [x] Reproducible Python environment
- [x] Installable Python package
- [x] Headless MuJoCo environment check
- [x] Panda model integration
- [x] Reliable reset and home configuration
- [x] Joint-space position control
- [x] Gripper control and cube contact
- [x] End-effector control
- [x] Grasping task environment
- [x] Deterministic random-grasp evaluation

## Environment

- Python 3.10
- MuJoCo 3.12
- NumPy 2.2
- SciPy 1.15
- Conda env: `mujoco` (`C:\Users\29391\.conda\envs\mujoco\python.exe`)
- Windows development environment

## Installation

Create the Conda environment:

    conda env create -f environment.yml
    conda activate panda-mujoco

Install the package in editable mode:

    python -m pip install -e .

## Quick Check

Run the headless environment check:

    python examples/check_environment.py

Expected final output:

    Simulation time: 2.0 s
    Ball height: 0.0496 m
    Environment check: PASSED

## Week 2 Acceptance

Run the deterministic 50-target end-effector reach evaluation:

    python examples/day14/evaluate_reach.py

The current fixed-seed benchmark result is:

    IK success:                 50/50
    overall success:            50/50
    success rate:               100.00%
    mean successful error:      0.027230 mm
    maximum successful error:   0.098611 mm
    non-finite cases:           0
    joint-limit violations:     0
    Day 14 reach evaluation: PASSED

Per-target metrics are stored in:

    results/day14/reach_evaluation.csv

These results apply to the documented fixed target range, home initial
configuration, MuJoCo model, and ideal model-based bias compensation. They do
not claim full-workspace or real-robot performance.

## Week 3 Acceptance

Run the deterministic 20-episode random-grasp evaluation:

    python examples/day21/week3_acceptance.py

The current fixed-seed benchmark result is:

    success count:              20/20
    success rate:               100.00%
    minimum successful lift:    7.728 cm
    minimum successful hold:    0.500 s
    non-finite cases:           0
    joint-limit violations:     0
    terminated episodes:        20
    truncated episodes:         0
    Week 3 random-grasp acceptance: PASSED

Per-episode metrics are stored in:

    results/day21/grasp_evaluation.csv

Demonstration videos:

- [Successful scripted grasp](results/day21/videos/week3_success_seed00.mp4)
- [Injected contact-timeout failure](results/day21/videos/week3_failure_contact_timeout.mp4)

The failure video intentionally limits gripper closing to one physics step so
that `VERIFY_CONTACT` rejects the grasp. It is a state-machine diagnostic and
is not part of the 20-episode random benchmark.

These results apply only to seeds 0 through 19, the documented cube sampling
range, the bundled MuJoCo model, and ideal simulation conditions. They do not
establish full-workspace robustness or real-robot performance.

## Model Assets

The Panda model assets are stored inside this repository at:

    assets/robots/panda/scene_with_cube.xml

The scene includes `panda.xml`, which loads mesh files from:

    assets/robots/panda/assets/

The original model license and source notes are preserved in:

    assets/robots/panda/LICENSE
    assets/robots/panda/README_original.md

Keeping the model inside the repository makes the project reproducible without relying on a local external model folder.


## Project Structure

    panda_mujoco_manipulation/
    ├── assets/
    │   ├── robots/panda/
    │   └── scenes/
    ├── configs/
    ├── docs/
    ├── examples/
    ├── results/
    ├── src/panda_mujoco/
    ├── tests/
    ├── environment.yml
    ├── LICENSE
    ├── pyproject.toml
    └── README.md

## Roadmap

### Week 1: Simulation Baseline

- Integrate and audit the Panda model
- Implement reliable reset and home configuration
- Implement joint-space control
- Verify gripper and cube contact

### Week 2: End-Effector Control

- Implement forward kinematics
- Implement Jacobian-based inverse kinematics
- Implement Cartesian pose control

### Week 3: Grasping Environment

- Build a cube-grasping task
- Define observations, actions, rewards, and termination
- Implement automated grasp evaluation

### Week 4: Engineering and Portfolio

- Add tests and experiment logging
- Record demonstration videos
- Complete documentation and publish a GitHub release

## License

Except where otherwise noted, the original source code and documentation in
this repository are licensed under the [Apache License 2.0](LICENSE).

The bundled Franka Emika Panda model assets remain under their upstream
Apache License 2.0. Their license and original source notes are preserved in:

    assets/robots/panda/LICENSE
    assets/robots/panda/README_original.md
