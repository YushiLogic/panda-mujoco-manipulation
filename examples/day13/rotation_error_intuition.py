"""理解旋转矩阵误差和旋转向量。"""

import numpy as np
from scipy.spatial.transform import Rotation


# 当前姿态：单位矩阵表示没有旋转。
current_rotation = np.eye(3)

# 目标姿态：绕世界坐标系 z 轴旋转 15 度。
target_rotation = Rotation.from_euler(
    "z",
    15.0,
    degrees=True,
).as_matrix()

# 计算从当前姿态旋转到目标姿态所需的相对旋转。
error_rotation = (
    target_rotation @ current_rotation.T
)

# 将旋转矩阵转换成旋转向量。
rotation_error = Rotation.from_matrix(
    error_rotation
).as_rotvec()

# 旋转向量的模长就是旋转角，单位为弧度。
error_angle_radians = np.linalg.norm(
    rotation_error
)

error_angle_degrees = np.rad2deg(
    error_angle_radians
)

print("Current rotation:")
print(current_rotation)

print("\nTarget rotation:")
print(target_rotation)

print("\nError rotation:")
print(error_rotation)

print("\nRotation error vector:")
print(rotation_error)

print("\nRotation error angle:")
print(f"{error_angle_radians:.9f} rad")
print(f"{error_angle_degrees:.6f} degrees")

expected_rotation_error = np.array(
    [0.0, 0.0, np.deg2rad(15.0)]
)

assert np.allclose(
    rotation_error,
    expected_rotation_error,
    atol=1e-12,
)

print("\nRotation error intuition: PASSED")