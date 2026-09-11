"""旋转误差工具测试。"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from panda_mujoco.rotations import rotation_error


def test_identical_rotations_have_zero_error():
    """当前姿态和目标姿态相同时，旋转误差应为零。"""

    rotation = Rotation.from_euler(
        "xyz",
        [10.0, -20.0, 30.0],
        degrees=True,
    ).as_matrix()

    error = rotation_error(
        rotation,
        rotation,
    )

    assert np.allclose(
        error,
        np.zeros(3),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("axis", "angle_degrees", "expected_axis"),
    [
        ("x", 15.0, np.array([1.0, 0.0, 0.0])),
        ("x", -15.0, np.array([1.0, 0.0, 0.0])),
        ("y", 15.0, np.array([0.0, 1.0, 0.0])),
        ("z", 15.0, np.array([0.0, 0.0, 1.0])),
    ],
)
def test_world_axis_rotation_error(
    axis,
    angle_degrees,
    expected_axis,
):
    """单位姿态下应得到正确的世界轴旋转误差。"""

    current_rotation = np.eye(3)

    target_rotation = Rotation.from_euler(
        axis,
        angle_degrees,
        degrees=True,
    ).as_matrix()

    error = rotation_error(
        target_rotation,
        current_rotation,
    )

    expected_error = (
        expected_axis * np.deg2rad(angle_degrees)
    )

    assert np.allclose(
        error,
        expected_error,
        atol=1e-12,
    )


def test_local_z_rotation_is_expressed_in_world_frame():
    """局部z轴旋转误差应转换到世界坐标系表达。"""

    current_rotation = Rotation.from_euler(
        "xyz",
        [20.0, -30.0, 10.0],
        degrees=True,
    ).as_matrix()

    angle_radians = np.deg2rad(15.0)

    local_z_rotation = Rotation.from_euler(
        "z",
        15.0,
        degrees=True,
    ).as_matrix()

    target_rotation = (
        current_rotation @ local_z_rotation
    )

    error = rotation_error(
        target_rotation,
        current_rotation,
    )

    # 当前旋转矩阵第3列，是局部z轴在世界坐标系中的方向。
    expected_error = (
        current_rotation[:, 2] * angle_radians
    )

    assert np.allclose(
        error,
        expected_error,
        atol=1e-12,
    )


def test_wrong_shape_is_rejected():
    """错误尺寸不能作为旋转矩阵。"""

    with pytest.raises(ValueError):
        rotation_error(
            np.eye(4),
            np.eye(3),
        )


def test_non_finite_matrix_is_rejected():
    """含NaN的矩阵必须被拒绝。"""

    invalid_rotation = np.eye(3)
    invalid_rotation[0, 0] = np.nan

    with pytest.raises(ValueError):
        rotation_error(
            invalid_rotation,
            np.eye(3),
        )


def test_non_orthogonal_matrix_is_rejected():
    """坐标轴不正交的矩阵必须被拒绝。"""

    invalid_rotation = np.eye(3)
    invalid_rotation[0, 1] = 0.1

    with pytest.raises(ValueError):
        rotation_error(
            invalid_rotation,
            np.eye(3),
        )


def test_reflection_matrix_is_rejected():
    """行列式为-1的镜像矩阵不是旋转矩阵。"""

    reflection = np.diag(
        [-1.0, 1.0, 1.0]
    )

    with pytest.raises(ValueError):
        rotation_error(
            reflection,
            np.eye(3),
        )