# Panda MuJoCo Manipulation

A reproducible MuJoCo project for learning robotic manipulation with the Franka Emika Panda robot.

This portfolio project covers joint-space control, end-effector control, inverse kinematics, grasping, testing, and experiment documentation.

## Current Status

- [x] Reproducible Python environment
- [x] Installable Python package
- [x] Headless MuJoCo environment check
- [ ] Panda model integration
- [ ] Reliable reset and home configuration
- [ ] Joint-space position control
- [ ] Gripper control and cube contact
- [ ] End-effector control
- [ ] Grasping task environment

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

## Project Structure

    panda_mujoco_manipulation/
    ├── assets/
    │   ├── robots/panda/
    │   └── scenes/
    ├── configs/
    ├── docs/
    ├── examples/
    ├── src/panda_mujoco/
    ├── tests/
    ├── environment.yml
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