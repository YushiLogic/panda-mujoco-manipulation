"""比较绕世界轴和末端局部轴旋转时的姿态误差。"""

import numpy as np
from scipy.spatial.transform import Rotation

from panda_mujoco.kinematics import (
    get_ee_rotation_matrix,
)
from panda_mujoco.rotations import rotation_error
from panda_mujoco.simulation import PandaScene



scene = PandaScene()

current_rotation = get_ee_rotation_matrix(scene)

angle_degrees = 15.0
angle_radians = np.deg2rad(angle_degrees)

# 绕 z 轴旋转15度的增量旋转矩阵。
z_rotation = Rotation.from_euler(
    "z",
    angle_degrees,
    degrees=True,
).as_matrix()

# 左乘：绕世界坐标系的 z 轴旋转。
world_target_rotation = (
    z_rotation @ current_rotation
)

# 右乘：绕末端局部坐标系的 z 轴旋转。
local_target_rotation = (
    current_rotation @ z_rotation
)

world_error = rotation_error(
    world_target_rotation,
    current_rotation,
)

local_error = rotation_error(
    local_target_rotation,
    current_rotation,
)

# current_rotation 的第3列，就是末端局部 z 轴
# 在世界坐标系中的方向。
ee_z_axis_in_world = current_rotation[:, 2]

print("Current Panda end-effector rotation:")
print(current_rotation)

print("\nEE local z-axis expressed in world frame:")
print(ee_z_axis_in_world)

print("\nWorld-z rotation error vector:")
print(world_error)

print("\nLocal-z rotation error vector:")
print(local_error)

print("\nWorld-z error angle:")
print(
    f"{np.rad2deg(np.linalg.norm(world_error)):.6f} degrees"
)

print("\nLocal-z error angle:")
print(
    f"{np.rad2deg(np.linalg.norm(local_error)):.6f} degrees"
)

expected_world_error = np.array(
    [0.0, 0.0, angle_radians]
)

expected_local_error = (
    ee_z_axis_in_world * angle_radians
)

assert np.allclose(
    world_error,
    expected_world_error,
    atol=1e-12,
)

assert np.allclose(
    local_error,
    expected_local_error,
    atol=1e-12,
)

print("\nPanda rotation error probe: PASSED")