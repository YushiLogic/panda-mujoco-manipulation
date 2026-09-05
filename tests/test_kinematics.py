"""测试 Panda 末端位姿读取接口。"""

'''
测试1：防止位置返回错误维度或 NaN
测试2：防止姿态矩阵不是合法旋转
测试3：防止组合接口和单独接口结果不一致
测试4：防止返回 MuJoCo 内部数组的可修改视图
测试5：防止位姿函数返回固定常数，不响应 qpos
'''
import mujoco
import numpy as np

from panda_mujoco.kinematics import (
    get_ee_pose,
    get_ee_position,
    get_ee_rotation_matrix,
)
from panda_mujoco.simulation import PandaScene


def test_ee_position_is_finite_xyz_vector() -> None:
    """末端位置应当是包含有限数值的三维向量。"""

    scene = PandaScene()

    position = get_ee_position(scene)

    assert position.shape == (3,)
    assert np.all(np.isfinite(position))


def test_ee_rotation_matrix_is_valid() -> None:
    """末端旋转矩阵应当正交，并且行列式等于 1。"""

    scene = PandaScene()

    rotation = get_ee_rotation_matrix(scene)

    assert rotation.shape == (3, 3)
    assert np.all(np.isfinite(rotation))

    np.testing.assert_allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=1e-9,
    )

    assert np.isclose(
        np.linalg.det(rotation),
        1.0,
        atol=1e-9,
    )


def test_get_ee_pose_matches_individual_functions() -> None:
    """组合接口的结果应与两个单独读取函数一致。"""

    scene = PandaScene()

    position, rotation = get_ee_pose(scene)

    np.testing.assert_allclose(
        position,
        get_ee_position(scene),
    )
    np.testing.assert_allclose(
        rotation,
        get_ee_rotation_matrix(scene),
    )


def test_get_ee_pose_returns_independent_copies() -> None:
    """修改函数返回值时，不应该改坏 MuJoCo 内部数据。"""

    scene = PandaScene()

    expected_position = get_ee_position(scene)
    expected_rotation = get_ee_rotation_matrix(scene)

    position, rotation = get_ee_pose(scene)

    # 故意破坏函数返回的数组。
    position[:] = 100.0
    rotation[:] = 0.0

    # 再次读取 MuJoCo 状态，应该仍然是原来的结果。
    np.testing.assert_allclose(
        get_ee_position(scene),
        expected_position,
    )
    np.testing.assert_allclose(
        get_ee_rotation_matrix(scene),
        expected_rotation,
    )


def test_ee_pose_changes_when_joint1_changes() -> None:
    """改变 joint1 后重新计算正运动学，末端位姿应当变化。"""

    scene = PandaScene()

    position_before, rotation_before = get_ee_pose(scene)

    # 直接将 joint1 增加 0.2 rad。
    # 这里只检查正运动学，不是在模拟真实动力学运动。
    scene.data.qpos[0] += 0.2

    # 根据新的 qpos 重新计算 site 的世界位姿，但不推进时间。
    mujoco.mj_forward(scene.model, scene.data)

    position_after, rotation_after = get_ee_pose(scene)

    assert not np.allclose(position_after, position_before)
    assert not np.allclose(rotation_after, rotation_before)