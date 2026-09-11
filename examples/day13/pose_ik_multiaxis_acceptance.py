"""对三个世界坐标轴小角度目标进行 6D IK 动力学验收。"""

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from panda_mujoco.arm_control import (
    apply_arm_bias_compensation,
    command_arm_joint_positions,
)
from panda_mujoco.ik import solve_pose_ik
from panda_mujoco.joint_trajectory import JointPath
from panda_mujoco.kinematics import get_ee_pose
from panda_mujoco.rotations import rotation_error
from panda_mujoco.simulation import PandaScene


CONTROL_EVERY = 10
MOVE_TICKS = 200
SETTLE_STEPS = 500


@dataclass
class PoseTrackingMetrics:
    """一个姿态目标的最终动力学跟踪指标。"""

    name: str
    ik_iterations: int
    position_error: float
    orientation_error: float
    maximum_joint_error: float
    joint_velocity_norm: float


def step_with_bias_compensation(scene: PandaScene) -> None:
    """执行一个带理想偏置力补偿的物理步。"""

    apply_arm_bias_compensation(scene)
    mujoco.mj_step(scene.model, scene.data)


def run_case(
    name: str,
    axis: str,
    angle_degrees: float,
) -> PoseTrackingMetrics:
    """从 home 出发规划并执行一个世界轴旋转目标。"""

    planning_scene = PandaScene()
    control_scene = PandaScene()

    target_position, initial_rotation = get_ee_pose(
        planning_scene
    )

    target_rotation = (
        Rotation.from_euler(
            axis,
            angle_degrees,
            degrees=True,
        ).as_matrix()
        @ initial_rotation
    )

    ik_result = solve_pose_ik(
        planning_scene,
        target_position,
        target_rotation,
        damping=0.01,
        position_tolerance=1e-4,
        orientation_tolerance=1e-3,
        max_iterations=100,
        max_joint_step=0.1,
    )

    if not ik_result.success:
        raise RuntimeError(
            f"{name}: pose IK failed after "
            f"{ik_result.iterations} updates"
        )

    target_joint_positions = (
        planning_scene.data.qpos[:7].copy()
    )

    path = JointPath(
        goals=[
            control_scene.data.qpos[:7].copy(),
            target_joint_positions,
        ],
        ticks_per_move=MOVE_TICKS,
    )

    while not path.finished():
        command_arm_joint_positions(
            control_scene,
            path.target(),
        )

        for _ in range(CONTROL_EVERY):
            step_with_bias_compensation(control_scene)

        path.advance()

    command_arm_joint_positions(
        control_scene,
        target_joint_positions,
    )

    for _ in range(SETTLE_STEPS):
        step_with_bias_compensation(control_scene)

    actual_position, actual_rotation = get_ee_pose(
        control_scene
    )

    position_error = float(
        np.linalg.norm(target_position - actual_position)
    )
    orientation_error = float(
        np.linalg.norm(
            rotation_error(target_rotation, actual_rotation)
        )
    )
    maximum_joint_error = float(
        np.max(
            np.abs(
                target_joint_positions
                - control_scene.data.qpos[:7]
            )
        )
    )
    joint_velocity_norm = float(
        np.linalg.norm(control_scene.data.qvel[:7])
    )

    return PoseTrackingMetrics(
        name=name,
        ik_iterations=ik_result.iterations,
        position_error=position_error,
        orientation_error=orientation_error,
        maximum_joint_error=maximum_joint_error,
        joint_velocity_norm=joint_velocity_norm,
    )


def main() -> None:
    cases = (
        ("world +x 15 deg", "x", 15.0),
        ("world -y 15 deg", "y", -15.0),
        ("world +z 15 deg", "z", 15.0),
    )

    results = []

    print("Day 13 multiaxis pose tracking acceptance")
    print()

    for name, axis, angle_degrees in cases:
        metrics = run_case(name, axis, angle_degrees)
        results.append(metrics)

        print(
            f"{metrics.name:18s} | "
            f"IK updates={metrics.ik_iterations:2d} | "
            f"position={metrics.position_error * 1000:.6f} mm | "
            "orientation="
            f"{np.rad2deg(metrics.orientation_error):.6f} deg | "
            f"joint={metrics.maximum_joint_error:.3e} rad | "
            f"qvel={metrics.joint_velocity_norm:.3e} rad/s"
        )

    maximum_position_error = max(
        result.position_error for result in results
    )
    maximum_orientation_error = max(
        result.orientation_error for result in results
    )
    maximum_joint_error = max(
        result.maximum_joint_error for result in results
    )
    maximum_velocity_norm = max(
        result.joint_velocity_norm for result in results
    )

    print()
    print("Summary")
    print(
        "maximum position error:    "
        f"{maximum_position_error * 1000:.6f} mm"
    )
    print(
        "maximum orientation error: "
        f"{np.rad2deg(maximum_orientation_error):.6f} degrees"
    )
    print(
        "maximum joint error:       "
        f"{maximum_joint_error:.6e} rad"
    )
    print(
        "maximum velocity norm:     "
        f"{maximum_velocity_norm:.6e} rad/s"
    )

    # 路线表要求姿态误差不超过5度、位置误差不超过2 cm。
    # 当前模型与理想补偿下采用更严格的自动验收门槛。
    assert maximum_position_error < 1e-4
    assert maximum_orientation_error < 1e-3
    assert maximum_joint_error < 1e-5
    assert maximum_velocity_norm < 1e-3

    print()
    print("Day 13 multiaxis pose tracking: PASSED")


if __name__ == "__main__":
    main()
