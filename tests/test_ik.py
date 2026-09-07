"""Tests for damped least-squares inverse kinematics."""

import mujoco
import numpy as np
import pytest

from panda_mujoco.ik import (
    damped_least_squares,
    solve_position_ik,
)
from panda_mujoco.kinematics import (
    ARM_JOINT_NAMES,
    get_ee_jacobian,
    get_ee_position,
)
from panda_mujoco.simulation import PandaScene


def test_dls_matches_known_solution() -> None:
    """用可以手算的二维例子验证 DLS 公式。"""
    jacobian = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.02],
        ]
    )
    task_error = np.array([0.0, 0.01])

    delta_q = damped_least_squares(
        jacobian,
        task_error,
        damping=0.05,
    )

    expected = np.array([0.0, 0.06896551724137931])

    np.testing.assert_allclose(delta_q, expected, atol=1e-12)


def test_zero_error_produces_zero_update() -> None:
    """末端没有误差时，关节不应该继续运动。"""
    jacobian = np.array(
        [
            [1.0, 2.0, 3.0],
            [0.0, 1.0, -1.0],
        ]
    )
    task_error = np.zeros(2)

    delta_q = damped_least_squares(
        jacobian,
        task_error,
        damping=0.05,
    )

    np.testing.assert_allclose(delta_q, np.zeros(3), atol=1e-12)


def test_invalid_damping_is_rejected() -> None:
    """阻尼系数必须是有限的正数。"""
    jacobian = np.eye(2)
    task_error = np.ones(2)

    invalid_values = (0.0, -0.1, np.inf, np.nan)

    for invalid_damping in invalid_values:
        with pytest.raises(
            ValueError,
            match="positive and finite",
        ):
            damped_least_squares(
                jacobian,
                task_error,
                damping=invalid_damping,
            )


def test_invalid_shapes_are_rejected() -> None:
    """Jacobian 必须是矩阵，误差维数必须和 Jacobian 行数一致。"""
    with pytest.raises(ValueError, match="2D"):
        damped_least_squares(
            np.array([1.0, 2.0]),
            np.array([1.0]),
            damping=0.05,
        )

    with pytest.raises(ValueError, match="shape"):
        damped_least_squares(
            np.eye(2),
            np.array([1.0]),
            damping=0.05,
        )


def test_panda_dls_step_reduces_position_error() -> None:
    """在真实 Panda 模型上验证一次 DLS 更新确实减小末端误差。"""
    scene = PandaScene()

    position_before = get_ee_position(scene)
    target_position = position_before + np.array(
        [0.02, -0.01, 0.015]
    )
    initial_error = target_position - position_before

    jacobian_position, _ = get_ee_jacobian(scene)

    delta_q = damped_least_squares(
        jacobian_position,
        initial_error,
        damping=0.05,
    )

    joint_qpos_addresses = np.array(
        [
            scene.model.jnt_qposadr[
                scene.model.joint(joint_name).id
            ]
            for joint_name in ARM_JOINT_NAMES
        ],
        dtype=int,
    )

    current_qpos = scene.data.qpos[joint_qpos_addresses].copy()
    scene.data.qpos[joint_qpos_addresses] = current_qpos + delta_q

    mujoco.mj_forward(scene.model, scene.data)

    position_after = get_ee_position(scene)
    final_error = target_position - position_after

    assert np.linalg.norm(final_error) < np.linalg.norm(initial_error)
def test_solve_position_ik_reaches_target() -> None:
    """迭代 IK 应当让 Panda 末端到达可达目标。"""
    scene = PandaScene()

    initial_position = get_ee_position(scene)
    target_position = initial_position + np.array(
        [0.02, -0.01, 0.015]
    )

    result = solve_position_ik(
        scene,
        target_position,
        damping=0.05,
        tolerance=1e-4,
        max_iterations=50,
        max_joint_step=0.1,
    )

    assert result.success is True
    assert 0 < result.iterations <= 50
    assert result.final_error_norm < 1e-4

    np.testing.assert_allclose(
        result.final_position,
        get_ee_position(scene),
        atol=1e-12,
    )

    assert result.final_error_norm == pytest.approx(
        np.linalg.norm(result.final_error)
    )


def test_solve_position_ik_uses_zero_updates_at_target() -> None:
    """如果末端已经在目标位置，就不应该修改关节角。"""
    scene = PandaScene()

    target_position = get_ee_position(scene)
    qpos_before = scene.data.qpos.copy()

    result = solve_position_ik(
        scene,
        target_position,
    )

    assert result.success is True
    assert result.iterations == 0
    assert result.final_error_norm == pytest.approx(0.0)

    np.testing.assert_allclose(
        scene.data.qpos,
        qpos_before,
        atol=0.0,
    )


def test_solve_position_ik_rejects_invalid_inputs() -> None:
    """IK 求解器应当拒绝错误的目标和配置参数。"""
    scene = PandaScene()

    with pytest.raises(ValueError, match="shape"):
        solve_position_ik(
            scene,
            np.array([0.1, 0.2]),
        )

    with pytest.raises(ValueError, match="max_iterations"):
        solve_position_ik(
            scene,
            get_ee_position(scene),
            max_iterations=0,
        )

    with pytest.raises(ValueError, match="damping"):
        solve_position_ik(
            scene,
            get_ee_position(scene),
            damping=0.0,
        )