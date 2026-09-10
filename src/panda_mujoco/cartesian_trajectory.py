"""Cartesian position waypoint generation and sequential IK planning."""

from dataclasses import dataclass

import numpy as np

from panda_mujoco.ik import solve_position_ik
from panda_mujoco.simulation import PandaScene


@dataclass
class CartesianWaypointPlan:
    """Result of sequential IK planning for Cartesian position waypoints."""

    cartesian_waypoints: np.ndarray
    joint_targets: np.ndarray
    ik_iterations: np.ndarray
    ik_errors: np.ndarray


def _as_position(
    value: np.ndarray,
    name: str,
) -> np.ndarray:
    """Return a validated three-dimensional position copy."""

    position = np.asarray(
        value,
        dtype=float,
    ).copy()

    if position.shape != (3,):
        raise ValueError(
            f"{name} must have shape (3,)"
        )

    if not np.all(np.isfinite(position)):
        raise ValueError(
            f"{name} must contain finite values"
        )

    return position


def linear_position_waypoints(
    start_position: np.ndarray,
    target_position: np.ndarray,
    waypoint_count: int,
) -> np.ndarray:
    """Generate equally spaced Cartesian position waypoints.

    Both the start and target positions are included in the result.
    """

    start_position = _as_position(
        start_position,
        "start_position",
    )
    target_position = _as_position(
        target_position,
        "target_position",
    )

    if (
        not isinstance(waypoint_count, (int, np.integer))
        or isinstance(waypoint_count, bool)
        or waypoint_count < 2
    ):
        raise ValueError(
            "waypoint_count must be an integer of at least 2"
        )

    return np.linspace(
        start_position,
        target_position,
        int(waypoint_count),
    )


def plan_position_waypoints(
    scene: PandaScene,
    cartesian_waypoints: np.ndarray,
    *,
    damping: float = 0.05,
    tolerance: float = 1e-4,
    max_iterations: int = 50,
    max_joint_step: float = 0.1,
) -> CartesianWaypointPlan:
    """Solve sequential position IK for a Cartesian waypoint array.

    The same scene is used for every waypoint. Because
    ``solve_position_ik`` updates its input scene, each waypoint starts
    from the previous waypoint's joint solution. This encourages a
    continuous sequence of joint targets for a redundant arm.
    """

    cartesian_waypoints = np.asarray(
        cartesian_waypoints,
        dtype=float,
    ).copy()

    if (
        cartesian_waypoints.ndim != 2
        or cartesian_waypoints.shape[1] != 3
        or len(cartesian_waypoints) < 1
    ):
        raise ValueError(
            "cartesian_waypoints must have shape (N, 3) with N >= 1"
        )

    if not np.all(np.isfinite(cartesian_waypoints)):
        raise ValueError(
            "cartesian_waypoints must contain finite values"
        )

    joint_targets = []
    ik_iterations = []
    ik_errors = []

    for waypoint_index, waypoint in enumerate(
        cartesian_waypoints
    ):
        result = solve_position_ik(
            scene,
            waypoint,
            damping=damping,
            tolerance=tolerance,
            max_iterations=max_iterations,
            max_joint_step=max_joint_step,
        )

        if not result.success:
            raise RuntimeError(
                "position IK failed at waypoint "
                f"{waypoint_index}: final error "
                f"{result.final_error_norm:.6e} m"
            )

        joint_targets.append(
            scene.data.qpos[:7].copy()
        )
        ik_iterations.append(
            result.iterations
        )
        ik_errors.append(
            result.final_error_norm
        )

    return CartesianWaypointPlan(
        cartesian_waypoints=(
            cartesian_waypoints.copy()
        ),
        joint_targets=np.stack(
            joint_targets,
            axis=0,
        ),
        ik_iterations=np.asarray(
            ik_iterations,
            dtype=int,
        ),
        ik_errors=np.asarray(
            ik_errors,
            dtype=float,
        ),
    )
