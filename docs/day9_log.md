# Day 9 Log - End-Effector Jacobian

## Goal

Build and validate the differential-kinematics interface needed for Cartesian end-effector control and numerical inverse kinematics.

Day 8 established the forward-kinematics mapping:

```text
joint configuration q -> end-effector pose
```

Day 9 studies the local velocity relationship at the current configuration:

```text
joint velocity qdot -> end-effector linear and angular velocity
```

The main equation is:

```text
[v, omega] = J(q) @ qdot
```

where:

- `qdot` contains the seven Panda arm joint velocities;
- `v` is the end-effector linear velocity in the world frame;
- `omega` is the end-effector angular velocity in the world frame;
- `J(q)` is the end-effector Jacobian at the current joint configuration.

This day does not yet solve inverse kinematics or send Cartesian commands to the actuators. It establishes and verifies the mathematical component those controllers will use.

## Jacobian Interpretation

For the Panda arm, the linear and angular Jacobians both have seven columns:

```text
linear Jacobian  Jp: (3, 7)
angular Jacobian Jr: (3, 7)
```

Each column corresponds to one arm joint velocity. For example:

```text
Jp[:, 0] = effect of joint1 velocity on end-effector linear velocity
Jr[:, 0] = effect of joint1 velocity on end-effector angular velocity
```

The rows of `Jp` represent world-frame x, y, and z linear velocity. The rows of `Jr` represent world-frame x, y, and z angular velocity.

Stacking the two matrices produces the complete 6D geometric Jacobian:

```python
J = np.vstack((Jp, Jr))
```

with shape:

```text
(6, 7)
```

The first three rows describe translation and the final three rows describe rotation.

## `nq` Versus `nv`

The complete scene reports:

```text
model.nq = 16
model.nv = 15
```

The difference comes from the cube free joint:

| Model part | Position values in `qpos` | Velocity DOFs in `qvel` |
|---|---:|---:|
| Panda arm | 7 | 7 |
| Two fingers | 2 | 2 |
| Cube free joint | 7 | 6 |
| Total | 16 | 15 |

The cube free joint stores three position values and a four-value quaternion, but its instantaneous velocity has three linear and three angular components. Therefore, Jacobian columns correspond to `nv`/`qvel` DOFs rather than directly to `nq`/`qpos` entries.

This distinction does not change the first seven single-DOF arm joints, but it becomes important when working with free joints or ball joints.

## Implementation

Extended:

```text
src/panda_mujoco/kinematics.py
```

Added the arm joint-name definition:

```python
ARM_JOINT_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
)
```

Added the project-level interface:

```python
get_ee_jacobian(scene)
```

The function calls:

```python
mujoco.mj_jacSite(...)
```

MuJoCo fills two full-scene arrays, each with shape `(3, model.nv)`. The function then uses each named arm joint's `jnt_dofadr` to select the correct seven Jacobian columns.

The returned values are:

```text
linear_jacobian:  shape (3, 7)
angular_jacobian: shape (3, 7)
```

The arrays are copied before being returned so callers cannot modify internal or temporary calculation data through the returned objects.

## Home-Configuration Jacobian

At the project home configuration, the measured linear Jacobian was:

```text
[[-0.000000,  0.455706, -0.000000, -0.139706, -0.000000, -0.057203,  0.000000],
 [ 0.688796,  0.000000,  0.688796,  0.000000, -0.057202,  0.000000,  0.000000],
 [ 0.000000, -0.688796,  0.000000,  0.606296,  0.000000,  0.222297,  0.000000]]
```

The angular Jacobian was:

```text
[[ 0.000000,  0.000000,  0.000000,  0.000000,  1.000000,  0.000000,  0.989993],
 [ 0.000000,  1.000000,  0.000000, -1.000000,  0.000000, -1.000000,  0.000000],
 [ 1.000000,  0.000000,  1.000000,  0.000000,  0.000006,  0.000000, -0.141114]]
```

Values displayed as `-0.0` are tiny negative floating-point values rounded for printing and should be interpreted as zero.

## Joint 1 Interpretation

At home, the end-effector position is approximately:

```text
p = [0.688796, 0.000000, 0.788706] m
```

Joint1 rotates around the world z axis. For a point rotated around that axis:

```text
dp/dq1 = [-y, x, 0]
```

Substituting the home position gives:

```text
dp/dq1 = [0.000000, 0.688796, 0.000000]
```

This matches the first linear-Jacobian column:

```text
Jp[:, 0] = [-0.000000, 0.688796, 0.000000]
```

The first angular-Jacobian column is:

```text
Jr[:, 0] = [0.000000, 0.000000, 1.000000]
```

It shows that positive joint1 velocity produces positive angular velocity around the world z axis.

## Joint 7 Interpretation

At the home configuration:

```text
Jp[:, 6] = [0.000000, 0.000000, 0.000000]
Jr[:, 6] = [0.989993, 0.000000, -0.141114]
```

Joint7 changes the end-effector orientation but does not instantaneously translate `ee_center_site` in this configuration. Its rotation axis passes through the site, so a point on that axis can remain stationary while the end-effector frame rotates.

The angular column also matches the end-effector local z axis measured on Day 8:

```text
EE local z in world = [0.989993, 0.000000, -0.141114]
```

This demonstrates why seven Jacobian columns mean seven joint-velocity inputs; they do not guarantee that every joint produces nonzero translation at every configuration.

## Finite-Difference Validation

The analytical linear Jacobian from MuJoCo was independently checked with central finite differences.

For joint `j`, the numerical column was calculated as:

```text
Jp[:, j] approximately equals
    [p(qj + epsilon) - p(qj - epsilon)] / (2 * epsilon)
```

with:

```text
epsilon = 1e-6 rad
```

Before each positive and negative perturbation, the complete `qpos` array was restored to the same reference configuration. This isolates one partial derivative at a time and prevents perturbations from different joints from accumulating.

After changing `qpos`, `mujoco.mj_forward()` was called to recompute the site position without advancing simulation time.

Measured maximum absolute errors by column were:

| Joint | Maximum absolute error |
|---|---:|
| joint1 | `6.021e-11` |
| joint2 | `1.312e-10` |
| joint3 | `4.058e-11` |
| joint4 | `1.309e-10` |
| joint5 | `5.551e-11` |
| joint6 | `1.383e-11` |
| joint7 | `1.110e-10` |

Overall result:

```text
maximum error = 1.312e-10
acceptance tolerance = 1.000e-7
Finite-difference Jacobian check: PASSED
```

The error is far below the acceptance tolerance. The remaining difference is normal floating-point and finite-difference roundoff.

Only the linear Jacobian was numerically differentiated on Day 9. The angular Jacobian was checked for shape, finite values, and interpretable joint-axis directions, but a full orientation finite-difference test is left for later work.

## Velocity-Prediction Experiment

The final experiment tested all seven columns together using:

```text
v = Jp @ qdot
```

The chosen joint-velocity vector was:

```text
qdot = [0.10, -0.05, 0.08, 0.04, -0.03, 0.02, 0.06] rad/s
```

Each joint contributes one scaled Jacobian column:

```text
joint contribution = Jp[:, joint] * qdot[joint]
```

The predicted world-frame linear velocity was:

```text
[-0.029517582, 0.125699352, 0.063137580] m/s
```

An independent central-difference measurement used:

```text
[p(q + qdot*dt) - p(q - qdot*dt)] / (2*dt)
```

with:

```text
dt = 1e-6 s
```

The measured velocity was:

```text
[-0.029517582, 0.125699352, 0.063137580] m/s
```

The maximum velocity error was:

```text
5.574e-11 m/s
```

Result:

```text
Jacobian velocity prediction: PASSED
```

This experiment verifies that the Jacobian is a linear map from joint velocity to instantaneous end-effector velocity at the current configuration. It does not imply that the same constant Jacobian can predict a large movement, because `J(q)` changes as the joint configuration changes.

## `mj_forward` Versus Dynamic Motion

The finite-difference and velocity-prediction experiments modify `qpos` directly and call:

```python
mujoco.mj_forward(model, data)
```

This recomputes kinematic quantities without advancing simulation time. It is appropriate for validating derivatives.

The experiments do not apply actuator commands, integrate forces, or simulate a controller. Real Cartesian control will repeatedly:

1. read the current pose and Jacobian;
2. calculate a desired joint update or velocity;
3. respect joint limits and velocity limits;
4. issue actuator targets;
5. advance the simulation with `mujoco.mj_step()`;
6. recompute the Jacobian at the new configuration.

## Tests

Extended:

```text
tests/test_kinematics.py
```

Three Day 9 tests were added:

1. Both Jacobian components have shape `(3, 7)` and contain finite values.
2. Modifying returned arrays does not affect later Jacobian calculations.
3. The analytical linear Jacobian agrees with a seven-joint central finite-difference calculation.

Kinematics test result:

```text
8 passed
```

Full project regression result:

```text
24 passed
```

## Files Added or Updated

Updated:

```text
src/panda_mujoco/kinematics.py
tests/test_kinematics.py
```

Added:

```text
examples/day9/jacobian_probe.py
examples/day9/jacobian_finite_difference_probe.py
examples/day9/jacobian_velocity_prediction.py
docs/day9_log.md
```

## Acceptance Summary

```text
Linear Jacobian shape                    (3, 7) PASSED
Angular Jacobian shape                   (3, 7) PASSED
Complete geometric Jacobian shape        (6, 7) PASSED
Named arm-DOF selection                         PASSED
Finite-value checks                             PASSED
Independent returned arrays                     PASSED
Linear finite-difference validation      1.312e-10 PASSED
Joint-velocity prediction validation     5.574e-11 PASSED
Kinematics test suite                    8 passed
Full regression suite                    24 passed
```

## Main Understanding Gained

Day 9 established the following mental model:

```text
Forward kinematics:
    q -> end-effector pose

Differential kinematics:
    qdot -> J(q) @ qdot -> end-effector velocity
```

A Jacobian column describes the end-effector velocity produced by one unit of one joint's velocity while the other joint velocities are zero. Multiplying by a general `qdot` scales those columns and adds their contributions.

The Jacobian is local and configuration-dependent. It must be recalculated as the robot moves.

## Connection to Day 10

Day 9 solved the forward velocity problem:

```text
known qdot -> compute end-effector velocity
```

Day 10 will begin the inverse differential-kinematics problem:

```text
desired end-effector displacement or velocity -> compute joint update
```

Because the Panda position Jacobian is `(3, 7)`, it is not a square matrix and cannot be inverted with an ordinary matrix inverse. The next step is to study the pseudoinverse and damped least-squares method, then use them to iteratively reduce end-effector position error while monitoring singularities, joint limits, and convergence.
