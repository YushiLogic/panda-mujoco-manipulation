# Day 8 Log - End-Effector Pose and Coordinate Frames

## Goal

Start Week 2 by establishing a reliable end-effector pose interface and building the coordinate-frame knowledge required for Jacobian-based inverse kinematics.

Day 8 focuses on reading and validating pose data. It does not yet command the end effector to move to a Cartesian target.

## Why This Comes Before Inverse Kinematics

An inverse-kinematics controller compares a desired end-effector pose with the current end-effector pose. Before implementing that controller, the project must answer four questions correctly:

1. Which model object represents the end effector?
2. In which coordinate frame is its position expressed?
3. How is its orientation represented?
4. How are points transformed between the end-effector and world frames?

The Panda model uses the following site as the end-effector reference:

```text
ee_center_site
```

Its MJCF definition is attached to `ee_center_body`, which is offset from the hand by 0.105 m:

```xml
<body name="ee_center_body" pos="0 0 0.105">
    <site name="ee_center_site" size="0.01" group="3"/>
</body>
```

## Pose Representation

The end-effector pose is represented by:

```text
position p: 3 values
rotation R: 3 x 3 matrix
```

MuJoCo computes both values from the current generalized position `qpos` through forward kinematics.

The project reads:

```python
site.xpos
site.xmat
```

`xpos` is the site origin expressed in the world frame. `xmat` contains nine values and is reshaped into a 3 x 3 rotation matrix.

## Implementation

Added:

```text
src/panda_mujoco/kinematics.py
```

The module provides:

```python
get_ee_position(scene)
get_ee_rotation_matrix(scene)
get_ee_pose(scene)
```

All returned arrays use `.copy()` so callers cannot accidentally modify MuJoCo's internal state through an array view.

The module does not manually implement Denavit-Hartenberg forward kinematics. MuJoCo already computes the model's forward kinematics from the MJCF kinematic tree and current `qpos`; this module provides a stable project-level interface to those results.

## Home Pose Result

At the project home configuration, the end-effector position in the world frame is:

```text
[0.688796, 0.000000, 0.788706] m
```

The rotation matrix is:

```text
[[-0.118736,  0.076256,  0.989993],
 [ 0.540385,  0.841418,  0.000000],
 [-0.832998,  0.534978, -0.141114]]
```

The columns of this matrix are the end-effector's local x, y, and z axes expressed in the world frame:

```text
local x in world = [-0.118736, 0.540385, -0.832998]
local y in world = [ 0.076256, 0.841418,  0.534978]
local z in world = [ 0.989993, 0.000000, -0.141114]
```

## Rotation-Matrix Validation

A valid three-dimensional rotation matrix must satisfy:

```text
R.T @ R = I
det(R) = +1
```

The measured results were:

```text
maximum orthogonality error = 1.443e-15
det(R)                    = 1.000000000000
```

The small nonzero orthogonality error is normal floating-point roundoff.

The `-0.` values printed in some arrays also represent very small negative floating-point values rounded to zero; they do not indicate a negative distance or a failed rotation matrix.

## Coordinate Transformation Experiment

For an end-effector-local point `point_local`, the corresponding world point is:

```python
point_world = position + rotation @ point_local
```

The inverse transformation is:

```python
point_local = rotation.T @ (point_world - position)
```

The transpose can be used as the inverse because a valid rotation matrix satisfies:

```text
R^-1 = R.T
```

The experiment transformed points located 0.1 m along each local axis.

| Local displacement | World displacement |
|---|---|
| local x +0.1 m | `[-0.011874, +0.054038, -0.083300] m` |
| local y +0.1 m | `[+0.007626, +0.084142, +0.053498] m` |
| local z +0.1 m | `[+0.098999, 0.000000, -0.014111] m` |

The local z result is especially relevant to grasping. Moving 0.1 m along the gripper's local z axis does not mean adding 0.1 m to the world z coordinate. At the current pose, it moves mainly in the positive world x direction and slightly downward.

Forward transformation followed by inverse transformation recovered all points with errors of approximately `1e-16 m`.

## Joint 1 Pose-Change Experiment

The final experiment directly increased joint1 by:

```text
0.2 rad = 11.459 degrees
```

Because joint1 rotates about the base/world z axis and its axis passes through the world origin in the current model, the theoretical pose update is:

```python
expected_position = rotation_z @ position_before
expected_rotation = rotation_z @ rotation_before
```

Measured positions:

```text
before    = [0.688796, 0.000000, 0.788706]
MuJoCo    = [0.675066, 0.136843, 0.788706]
predicted = [0.675066, 0.136843, 0.788706]
```

Errors:

```text
maximum position error = 1.110e-16 m
maximum rotation error = 2.220e-16
```

The unchanged world z coordinate and rotated x-y coordinates match the expected motion about the world z axis.

## `mj_forward` Versus `mj_step`

The joint1 experiment directly changes `data.qpos` and calls:

```python
mujoco.mj_forward(model, data)
```

This recomputes position-dependent quantities, including the site pose, without advancing simulation time.

This is appropriate for a forward-kinematics experiment. A real controlled movement still requires writing actuator commands and repeatedly calling `mujoco.mj_step()` so that forces, velocities, and positions evolve through dynamics.

## Tests

Added:

```text
tests/test_kinematics.py
```

The five tests verify:

1. The position is a finite three-dimensional vector.
2. The rotation matrix is finite, orthogonal, and has determinant +1.
3. The combined pose interface matches the individual accessors.
4. Returned arrays are independent copies.
5. The pose changes after joint1 changes and forward kinematics is recomputed.

Day 8 test result:

```text
5 passed
```

Full project regression result:

```text
21 passed
```

## Files Added

```text
src/panda_mujoco/kinematics.py
tests/test_kinematics.py
examples/day8/ee_pose_probe.py
examples/day8/frame_transform_probe.py
examples/day8/joint1_pose_change_probe.py
docs/day8_log.md
```

The supplementary reading notes are stored in:

```text
docs/mujoco_official_docs_guide.md
```

## Acceptance Summary

```text
End-effector position reading          PASSED
Rotation-matrix validation             PASSED
Local/world round-trip transformation  PASSED
Joint1 theoretical pose prediction     PASSED
Day 8 tests                            5 passed
Full regression suite                  21 passed
```

## Connection to Day 9

Day 8 established the mapping:

```text
qpos -> end-effector position and orientation
```

Day 9 will study the local differential relationship:

```text
small joint change dq -> small end-effector change dx
dx approximately equals J(q) @ dq
```

The next implementation target is to compute the translational and rotational site Jacobians with `mujoco.mj_jacSite()` and verify the translational part against finite differences.
