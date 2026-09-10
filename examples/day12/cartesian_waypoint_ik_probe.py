"""依次为笛卡尔直线路点求连续的 Panda 关节目标。"""

import numpy as np

from panda_mujoco.cartesian_trajectory import (
    linear_position_waypoints,
    plan_position_waypoints,
)
from panda_mujoco.kinematics import get_ee_position
from panda_mujoco.simulation import PandaScene


TARGET_OFFSET = np.array(
    [0.020, -0.010, 0.015]
)

WAYPOINT_COUNT = 6


def main() -> None:
    planning_scene = PandaScene()

    start_position = get_ee_position(
        planning_scene
    )

    target_position = (
        start_position + TARGET_OFFSET
    )

    cartesian_waypoints = linear_position_waypoints(
        start_position,
        target_position,
        waypoint_count=WAYPOINT_COUNT,
    )

    plan = plan_position_waypoints(
        planning_scene,
        cartesian_waypoints,
        damping=0.05,
        tolerance=1e-4,
        max_iterations=50,
        max_joint_step=0.1,
    )

    joint_targets = plan.joint_targets
    waypoint_errors = plan.ik_errors

    # plan_position_waypoints() 完成后，planning_scene 已停在最后一个
    # 路点的关节姿态。因此打印变化量时要从计划中的第一组关节角开始，
    # 不能再把 scene 当前（最终）状态误当作起点。
    previous_joint_positions = (
        joint_targets[0].copy()
    )

    print("Sequential waypoint IK")
    print()

    for index, current_joint_positions in enumerate(
        joint_targets
    ):

        joint_change = (
            current_joint_positions
            - previous_joint_positions
        )

        joint_change_norm = float(
            np.linalg.norm(joint_change)
        )

        print(
            f"waypoint {index} | "
            f"success=True | "
            f"updates={plan.ik_iterations[index]} | "
            f"position error="
            f"{plan.ik_errors[index]:.6e} m | "
            f"joint change norm="
            f"{joint_change_norm:.6e} rad"
        )

        previous_joint_positions = (
            current_joint_positions.copy()
        )

    # 相邻两组关节目标的变化。
    joint_segment_changes = np.diff(
        joint_targets,
        axis=0,
    )

    joint_segment_norms = np.linalg.norm(
        joint_segment_changes,
        axis=1,
    )

    np.set_printoptions(
        precision=9,
        suppress=True,
    )

    print()
    print("Joint targets:")
    print(joint_targets)

    print()
    print("Joint-space segment norms:")
    print(joint_segment_norms)

    print()
    print(
        "Maximum waypoint position error: "
        f"{np.max(waypoint_errors):.6e} m"
    )

    print(
        "Maximum joint-space segment norm: "
        f"{np.max(joint_segment_norms):.6e} rad"
    )

    assert joint_targets.shape == (
        WAYPOINT_COUNT,
        7,
    )

    assert np.all(np.isfinite(joint_targets))

    assert np.max(waypoint_errors) < 1e-4

    # 当前5.4 mm的笛卡尔路点间距，对应的关节变化
    # 应明显小于单步限制0.1 rad。
    assert np.max(joint_segment_norms) < 0.05

    # 所有关节目标都必须处于执行器控制范围。
    control_ranges = (
        planning_scene.model.actuator_ctrlrange[:7]
    )

    assert np.all(
        joint_targets >= control_ranges[:, 0]
    )

    assert np.all(
        joint_targets <= control_ranges[:, 1]
    )

    print()
    print("Sequential Cartesian waypoint IK: PASSED")


if __name__ == "__main__":
    main()
