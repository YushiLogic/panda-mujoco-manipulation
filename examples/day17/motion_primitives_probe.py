"""Day17：验证MOVE_ABOVE和APPROACH运动原语。"""

import numpy as np

from panda_mujoco.cube_reset import (
    reset_cube,
    sample_cube_pose,
    settle_cube,
)
from panda_mujoco.grasp_task import (
    generate_grasp_targets,
)
from panda_mujoco.gripper import (
    GRIPPER_OPEN_WIDTH,
    open_gripper,
)
from panda_mujoco.motion import (
    MotionResult,
    move_end_effector_to,
)
from panda_mujoco.simulation import PandaScene


TEST_SEEDS = (42, 43, 44)

POSITION_TOLERANCE = 0.002
ORIENTATION_TOLERANCE = np.deg2rad(1.0)
GRIPPER_WIDTH_TOLERANCE = 0.001


def get_cube_yaw(
    scene: PandaScene,
) -> float:
    """读取方块落稳后的实际世界yaw。"""

    cube_rotation = (
        scene.data.body("cube")
        .xmat.reshape(3, 3)
        .copy()
    )

    return float(
        np.arctan2(
            cube_rotation[1, 0],
            cube_rotation[0, 0],
        )
    )


def print_stage_result(
    stage_name: str,
    result: MotionResult,
) -> None:
    """打印一个运动阶段的核心验收指标。"""

    print(
        f"{stage_name:<10} | "
        f"success={result.success} | "
        f"reason={result.reason} | "
        f"IK={result.ik_iterations:2d} | "
        f"position="
        f"{result.position_error_norm * 1000.0:.6f} mm | "
        f"orientation="
        f"{np.degrees(result.orientation_error_norm):.6f} deg | "
        f"joint="
        f"{result.maximum_joint_error:.3e} rad | "
        f"time="
        f"{result.simulation_duration:.3f} s | "
        f"gripper="
        f"{result.gripper_width:.6f} m"
    )


def assert_stage_passed(
    result: MotionResult,
) -> None:
    """执行Day17单阶段验收。"""

    assert result.success
    assert result.reason == "none"

    assert (
        result.position_error_norm
        <= POSITION_TOLERANCE
    )

    assert (
        result.orientation_error_norm
        <= ORIENTATION_TOLERANCE
    )

    assert not result.joint_limit_violation

    assert np.all(
        np.isfinite(result.final_position)
    )

    assert np.all(
        np.isfinite(result.final_rotation)
    )

    assert abs(
        result.gripper_width
        - GRIPPER_OPEN_WIDTH
    ) <= GRIPPER_WIDTH_TOLERANCE


def main() -> None:
    """从home开始连续验证三个随机方块案例。"""

    maximum_position_error = 0.0
    maximum_orientation_error = 0.0
    maximum_joint_error = 0.0
    minimum_gripper_width = float("inf")
    total_motion_duration = 0.0

    print("Day 17 motion primitives acceptance")
    print()

    for case_index, seed in enumerate(
        TEST_SEEDS
    ):
        # 每个案例都创建全新场景，因此都从home开始。
        scene = PandaScene()

        sampled_pose = sample_cube_pose(seed)

        reset_cube(
            scene,
            sampled_pose,
        )

        settle_result = settle_cube(scene)

        assert settle_result.settled

        cube_position = (
            settle_result.final_position.copy()
        )

        cube_yaw = get_cube_yaw(scene)

        grasp_targets = generate_grasp_targets(
            cube_position,
            cube_yaw=cube_yaw,
        )

        # 在MOVE_ABOVE和APPROACH之前保持夹爪张开。
        open_gripper(scene)

        print(
            f"case {case_index} | "
            f"seed={seed} | "
            f"cube="
            f"({cube_position[0]:+.4f}, "
            f"{cube_position[1]:+.4f}, "
            f"{cube_position[2]:+.4f}) m | "
            f"yaw={np.degrees(cube_yaw):+.3f} deg"
        )

        # ---------- MOVE_ABOVE ----------

        move_above_result = (
            move_end_effector_to(
                scene,
                grasp_targets.pregrasp_position,
                grasp_targets.rotation,
                command_count=201,
                physics_steps_per_command=10,
                settle_steps=250,
            )
        )

        print_stage_result(
            "MOVE_ABOVE",
            move_above_result,
        )

        assert_stage_passed(
            move_above_result
        )

        # ---------- APPROACH ----------

        approach_result = (
            move_end_effector_to(
                scene,
                grasp_targets.grasp_position,
                grasp_targets.rotation,
                command_count=101,
                physics_steps_per_command=10,
                settle_steps=250,
            )
        )

        print_stage_result(
            "APPROACH",
            approach_result,
        )

        assert_stage_passed(
            approach_result
        )

        # ---------- 汇总本案例指标 ----------

        for result in (
            move_above_result,
            approach_result,
        ):
            maximum_position_error = max(
                maximum_position_error,
                result.position_error_norm,
            )

            maximum_orientation_error = max(
                maximum_orientation_error,
                result.orientation_error_norm,
            )

            maximum_joint_error = max(
                maximum_joint_error,
                result.maximum_joint_error,
            )

            minimum_gripper_width = min(
                minimum_gripper_width,
                result.gripper_width,
            )

            total_motion_duration += (
                result.simulation_duration
            )

        print()

    print("Summary")

    print(
        "cases:",
        len(TEST_SEEDS),
    )

    print(
        "maximum position error:",
        maximum_position_error * 1000.0,
        "mm",
    )

    print(
        "maximum orientation error:",
        np.degrees(
            maximum_orientation_error
        ),
        "degrees",
    )

    print(
        "maximum joint error:",
        maximum_joint_error,
        "rad",
    )

    print(
        "minimum gripper width:",
        minimum_gripper_width,
        "m",
    )

    print(
        "total motion simulation duration:",
        total_motion_duration,
        "s",
    )

    print()
    print(
        "Day 17 motion primitives: PASSED"
    )


if __name__ == "__main__":
    main()