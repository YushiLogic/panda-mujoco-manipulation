# Day 6 Log - Model Assets and Reproducibility

## Goal

Make the Panda MuJoCo project self-contained and reproducible.

Before Day 6, the project loaded the Panda model from an external local folder:

```text
D:\Mujoco Project\model\franka_emika_panda\scene_with_cube.xml
```

This worked on my machine, but it would fail after cloning the repository on another machine.

## What I Changed

Copied the required Panda model files into the project repository:

```text
assets/robots/panda/
```

Included files:

- `scene_with_cube.xml`
- `panda.xml`
- `assets/*.obj`
- `assets/*.stl`
- `LICENSE`
- `README_original.md`

Updated `src/panda_mujoco/simulation.py` so `DEFAULT_SCENE` loads the repository-local scene:

```text
assets/robots/panda/scene_with_cube.xml
```

instead of the external model directory.

Updated `README.md` to mark the first-week baseline tasks as completed and document the model asset location.

## Why This Matters

This change makes the project portable.

Before this change, the code depended on a path that only exists on my computer. If another person cloned the repository, MuJoCo would not be able to find the Panda XML and mesh files.

After this change, the model XML, mesh assets, license, and source notes are all stored inside the repository. The project can now be installed, tested, and run from the repository itself.

This is important for an open-source portfolio project because reproducibility is part of engineering quality.

## Technical Notes

The project now uses this path in `src/panda_mujoco/simulation.py`:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE = PROJECT_ROOT / "assets" / "robots" / "panda" / "scene_with_cube.xml"
```

`PROJECT_ROOT` points to:

```text
D:\Mujoco Project\panda_mujoco_manipulation
```

The final scene path becomes:

```text
D:\Mujoco Project\panda_mujoco_manipulation\assets\robots\panda\scene_with_cube.xml
```

The scene dependency structure is:

```text
scene_with_cube.xml
└── includes panda.xml
    └── loads mesh files from assets/
```

So the following layout is required:

```text
assets/
└── robots/
    └── panda/
        ├── scene_with_cube.xml
        ├── panda.xml
        ├── LICENSE
        ├── README_original.md
        └── assets/
            ├── *.obj
            └── *.stl
```

## Verification

Checked that the default scene path points to the repository-local model:

```powershell
python -c "from panda_mujoco.simulation import DEFAULT_SCENE; print(DEFAULT_SCENE); print(DEFAULT_SCENE.exists())"
```

Result:

```text
D:\Mujoco Project\panda_mujoco_manipulation\assets\robots\panda\scene_with_cube.xml
True
```

Ran the environment check:

```powershell
python examples/check_environment.py
```

Result:

```text
Python: 3.10.20
MuJoCo: 3.12.0
NumPy: 2.2.6
SciPy: 1.15.3
Project package: D:\Mujoco Project\panda_mujoco_manipulation\src\panda_mujoco\__init__.py
Simulation time: 2.0 s
Ball height: 0.0496 m
Environment check: PASSED
```

Ran the full test suite:

```powershell
python -m pytest -v
```

Result:

```text
16 passed
```

## Git Commit

```text
c19da3d chore: make Panda model assets self-contained
```

## Remaining Notes

`examples/day5/contact_inventory.py` showed a Git status change caused by line-ending normalization only. It was not included in the Day 6 commit.

The Day 6 commit only contains:

- README update
- repository-local Panda model assets
- `simulation.py` default scene path update

## Reflection

Day 6 was not about adding a new control algorithm.

The main lesson is that a robotics simulation project is not complete just because it runs locally. It also needs to be reproducible by someone else.

For future work, all scripts and tests should load the Panda scene through `PandaScene` or `DEFAULT_SCENE`, instead of hard-coding local absolute paths.
