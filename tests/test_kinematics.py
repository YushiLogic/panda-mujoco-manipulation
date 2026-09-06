"""测试 Panda 末端位姿与 Jacobian 读取接口。

测试1～5覆盖末端位置与旋转，测试6～8覆盖 Jacobian 的接口、
返回值独立性，以及解析结果与中心有限差分的一致性。
"""
import mujoco
import numpy as np

from panda_mujoco.kinematics import (
    ARM_JOINT_NAMES,
    get_ee_jacobian,
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


def test_ee_jacobian_has_expected_shape() -> None:
    """末端线速度和角速度 Jacobian 都应为有限的 3×7 矩阵。"""

    scene = PandaScene()

    linear_jacobian, angular_jacobian = (
        get_ee_jacobian(scene)
    )

    assert linear_jacobian.shape == (3, 7)
    assert angular_jacobian.shape == (3, 7)

    assert np.all(np.isfinite(linear_jacobian))
    assert np.all(np.isfinite(angular_jacobian))


def test_ee_jacobian_returns_independent_arrays() -> None:
    """修改函数返回值不应影响下一次 Jacobian 计算。"""

    scene = PandaScene()

    expected_linear, expected_angular = (
        get_ee_jacobian(scene)
    )

    linear_jacobian, angular_jacobian = (
        get_ee_jacobian(scene)
    )

    # 故意破坏本次返回的数组。
    linear_jacobian[:] = 100.0
    angular_jacobian[:] = 100.0

    actual_linear, actual_angular = (
        get_ee_jacobian(scene)
    )

    np.testing.assert_allclose(
        actual_linear,
        expected_linear,
    )

    np.testing.assert_allclose(
        actual_angular,
        expected_angular,
    )


def test_linear_jacobian_matches_finite_difference() -> None:
    """位置 Jacobian 应当与中心有限差分结果一致。"""

    scene = PandaScene()
    model = scene.model
    data = scene.data

    analytical_jacobian, _ = get_ee_jacobian(scene)

    numerical_jacobian = np.zeros((3, 7))
    reference_qpos = data.qpos.copy()
    epsilon = 1e-6

    for column, joint_name in enumerate(ARM_JOINT_NAMES):
        joint_id = model.joint(joint_name).id
        qpos_address = model.jnt_qposadr[joint_id]

        # 正向扰动。
        data.qpos[:] = reference_qpos
        data.qpos[qpos_address] += epsilon
        mujoco.mj_forward(model, data)
        position_plus = get_ee_position(scene)

        # 反向扰动。
        data.qpos[:] = reference_qpos
        data.qpos[qpos_address] -= epsilon
        mujoco.mj_forward(model, data)
        position_minus = get_ee_position(scene)

        numerical_jacobian[:, column] = (
            position_plus - position_minus
        ) / (2.0 * epsilon)

    # 测试结束后恢复原始状态。
    data.qpos[:] = reference_qpos
    mujoco.mj_forward(model, data)

    np.testing.assert_allclose(
        analytical_jacobian,
        numerical_jacobian,
        atol=1e-7,
        rtol=0.0,
    )
