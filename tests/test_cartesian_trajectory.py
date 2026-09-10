"""Tests for Cartesian waypoint generation and sequential IK planning."""

import numpy as np
import pytest

from panda_mujoco.cartesian_trajectory import (
    linear_position_waypoints,
    plan_position_waypoints,
)
from panda_mujoco.kinematics import get_ee_position
from panda_mujoco.simulation import PandaScene


def test_linear_waypoints_include_endpoints_and_equal_steps() -> None:
    """The Cartesian line must include both endpoints and equal steps."""

    start = np.array([0.1, -0.2, 0.3])
    target = np.array([0.2, 0.0, 0.5])

    waypoints = linear_position_waypoints(
        start,
        target,
        waypoint_count=6,
    )

    assert waypoints.shape == (6, 3)
    np.testing.assert_allclose(waypoints[0], start)
    np.testing.assert_allclose(waypoints[-1], target)

    segment_vectors = np.diff(waypoints, axis=0)

    np.testing.assert_allclose(
        segment_vectors,
        np.tile((target - start) / 5, (5, 1)),
        atol=1e-12,
    )


def test_linear_waypoints_reject_invalid_inputs() -> None:
    """Positions and waypoint count must have valid shapes and values."""

    with pytest.raises(ValueError, match="start_position"):
        linear_position_waypoints(
            np.zeros(2),
            np.zeros(3),
            waypoint_count=6,
        )

    with pytest.raises(ValueError, match="finite"):
        linear_position_waypoints(
            np.array([0.0, np.nan, 0.0]),
            np.zeros(3),
            waypoint_count=6,
        )

    with pytest.raises(ValueError, match="at least 2"):
        linear_position_waypoints(
            np.zeros(3),
            np.ones(3),
            waypoint_count=1,
        )


def test_sequential_waypoint_ik_produces_continuous_targets() -> None:
    """A short reachable line should produce finite nearby joint targets."""

    scene = PandaScene()
    start = get_ee_position(scene)
    target = start + np.array(
        [0.020, -0.010, 0.015]
    )
    waypoints = linear_position_waypoints(
        start,
        target,
        waypoint_count=6,
    )

    plan = plan_position_waypoints(
        scene,
        waypoints,
    )

    assert plan.cartesian_waypoints.shape == (6, 3)
    assert plan.joint_targets.shape == (6, 7)
    assert plan.ik_iterations.shape == (6,)
    assert plan.ik_errors.shape == (6,)
    assert plan.ik_iterations[0] == 0
    assert np.all(plan.ik_iterations[1:] > 0)
    assert np.all(plan.ik_errors < 1e-4)
    assert np.all(np.isfinite(plan.joint_targets))

    joint_segment_norms = np.linalg.norm(
        np.diff(plan.joint_targets, axis=0),
        axis=1,
    )

    assert np.max(joint_segment_norms) < 0.05

    np.testing.assert_allclose(
        scene.data.qpos[:7],
        plan.joint_targets[-1],
        atol=0.0,
    )


def test_waypoint_plan_owns_independent_array_copies() -> None:
    """Changing the caller's waypoint array must not alter a saved plan."""

    scene = PandaScene()
    start = get_ee_position(scene)
    waypoints = linear_position_waypoints(
        start,
        start + np.array([0.005, 0.0, 0.0]),
        waypoint_count=2,
    )

    plan = plan_position_waypoints(
        scene,
        waypoints,
    )
    saved_waypoints = plan.cartesian_waypoints.copy()

    waypoints[:] = 999.0

    np.testing.assert_array_equal(
        plan.cartesian_waypoints,
        saved_waypoints,
    )
