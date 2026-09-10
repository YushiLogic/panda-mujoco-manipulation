"""生成末端起点到目标点之间的笛卡尔直线路点。"""

import numpy as np

from panda_mujoco.cartesian_trajectory import (
    linear_position_waypoints,
)
from panda_mujoco.kinematics import get_ee_position
from panda_mujoco.simulation import PandaScene


TARGET_OFFSET = np.array(
    [0.020, -0.010, 0.015]
)

# 包括起点和终点，共生成6个点，也就是5段直线。
WAYPOINT_COUNT = 6


def main() -> None:
    scene = PandaScene()

    start_position = get_ee_position(scene)
    target_position = (
        start_position + TARGET_OFFSET
    )

    waypoints = linear_position_waypoints(
        start_position,
        target_position,
        waypoint_count=WAYPOINT_COUNT,
    )

    # 相邻路点之差，形状为 (5, 3)。
    segment_vectors = np.diff(
        waypoints,
        axis=0,
    )

    # 每一段在三维空间中的欧氏长度。
    segment_lengths = np.linalg.norm(
        segment_vectors,
        axis=1,
    )

    np.set_printoptions(
        precision=9,
        suppress=True,
    )

    print("Start position:")
    print(start_position)

    print()
    print("Target position:")
    print(target_position)

    print()
    print("Cartesian waypoints:")

    for index, waypoint in enumerate(waypoints):
        print(
            f"waypoint {index}: {waypoint}"
        )

    print()
    print("Segment vectors:")
    print(segment_vectors)

    print()
    print("Segment lengths:")
    print(segment_lengths)

    expected_segment_vector = (
        TARGET_OFFSET / (WAYPOINT_COUNT - 1)
    )

    assert waypoints.shape == (
        WAYPOINT_COUNT,
        3,
    )

    np.testing.assert_allclose(
        waypoints[0],
        start_position,
    )

    np.testing.assert_allclose(
        waypoints[-1],
        target_position,
    )

    # 每段的方向和长度都应相同。
    for segment_vector in segment_vectors:
        np.testing.assert_allclose(
            segment_vector,
            expected_segment_vector,
            atol=1e-12,
        )

    np.testing.assert_allclose(
        segment_lengths,
        np.full(
            WAYPOINT_COUNT - 1,
            segment_lengths[0],
        ),
        atol=1e-12,
    )

    print()
    print("Cartesian waypoint generation: PASSED")


if __name__ == "__main__":
    main()
