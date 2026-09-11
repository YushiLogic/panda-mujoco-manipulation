"""使用迭代DLS同时控制Panda末端位置和姿态。"""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from panda_mujoco.ik import damped_least_squares
from panda_mujoco.kinematics import (
    ARM_JOINT_NAMES,
    get_ee_jacobian,
    get_ee_pose,
)
from panda_mujoco.rotations import rotation_error
from panda_mujoco.simulation import PandaScene


TARGET_ANGLE_DEGREES = 15.0

DAMPING = 0.01
MAX_JOINT_STEP = 0.1
MAX_ITERATIONS = 50

POSITION_TOLERANCE = 1e-4
ORIENTATION_TOLERANCE = np.deg2rad(0.1)


def main() -> None:
    scene = PandaScene()
    model = scene.model
    data = scene.data

    initial_position, initial_rotation = get_ee_pose(
        scene
    )

    # 目标位置保持不变。
    target_position = initial_position.copy()

    # 目标姿态：绕世界z轴旋转15度。
    world_z_rotation = Rotation.from_euler(
        "z",
        TARGET_ANGLE_DEGREES,
        degrees=True,
    ).as_matrix()

    target_rotation = (
        world_z_rotation @ initial_rotation
    )

    joint_ids = np.array(
        [
            model.joint(joint_name).id
            for joint_name in ARM_JOINT_NAMES
        ],
        dtype=int,
    )

    qpos_addresses = np.array(
        [
            model.jnt_qposadr[joint_id]
            for joint_id in joint_ids
        ],
        dtype=int,
    )

    lower_limits = model.jnt_range[joint_ids, 0]
    upper_limits = model.jnt_range[joint_ids, 1]

    success = False
    updates_used = 0

    print("Iterative 6D pose IK progress")
    print()

    for iteration in range(MAX_ITERATIONS + 1):
        current_position, current_rotation = (
            get_ee_pose(scene)
        )

        position_error = (
            target_position - current_position
        )

        orientation_error = rotation_error(
            target_rotation,
            current_rotation,
        )

        position_error_norm = np.linalg.norm(
            position_error
        )

        orientation_error_angle = np.linalg.norm(
            orientation_error
        )

        print(
            f"iteration={iteration:2d} | "
            f"position error="
            f"{position_error_norm * 1000:.6f} mm | "
            f"orientation error="
            f"{np.rad2deg(orientation_error_angle):.6f} deg"
        )

        # 位置和姿态必须同时满足容差。
        if (
            position_error_norm < POSITION_TOLERANCE
            and orientation_error_angle
            < ORIENTATION_TOLERANCE
        ):
            success = True
            break

        # 防止最后一次检查后继续执行更新。
        if iteration == MAX_ITERATIONS:
            break

        pose_error = np.concatenate(
            [position_error, orientation_error]
        )

        linear_jacobian, angular_jacobian = (
            get_ee_jacobian(scene)
        )

        pose_jacobian = np.vstack(
            [linear_jacobian, angular_jacobian]
        )

        raw_delta_q = damped_least_squares(
            pose_jacobian,
            pose_error,
            damping=DAMPING,
        )

        maximum_raw_step = np.max(
            np.abs(raw_delta_q)
        )

        step_scale = min(
            1.0,
            MAX_JOINT_STEP / maximum_raw_step,
        )

        delta_q = raw_delta_q * step_scale

        current_joint_positions = (
            data.qpos[qpos_addresses].copy()
        )

        candidate_joint_positions = (
            current_joint_positions + delta_q
        )

        # 这个实验应当在不触碰限位的情况下完成。
        assert np.all(
            candidate_joint_positions >= lower_limits
        )

        assert np.all(
            candidate_joint_positions <= upper_limits
        )

        data.qpos[qpos_addresses] = (
            candidate_joint_positions
        )

        # 更新qpos后重新进行正运动学。
        mujoco.mj_forward(model, data)

        updates_used += 1

    final_position, final_rotation = get_ee_pose(
        scene
    )

    final_position_error = (
        target_position - final_position
    )

    final_orientation_error = rotation_error(
        target_rotation,
        final_rotation,
    )

    final_position_error_norm = np.linalg.norm(
        final_position_error
    )

    final_orientation_error_angle = np.linalg.norm(
        final_orientation_error
    )

    np.set_printoptions(
        precision=9,
        suppress=True,
    )

    print()
    print("Target position:")
    print(target_position)

    print()
    print("Final position:")
    print(final_position)

    print()
    print("Final position error:")
    print(final_position_error)

    print()
    print("Final orientation error vector:")
    print(final_orientation_error)

    print()
    print(f"Updates used: {updates_used}")

    print(
        "Final position error:    "
        f"{final_position_error_norm * 1000:.6f} mm"
    )

    print(
        "Final orientation error: "
        f"{np.rad2deg(final_orientation_error_angle):.6f} degrees"
    )

    print(f"Success: {success}")

    assert success
    assert np.all(np.isfinite(final_position))
    assert np.all(np.isfinite(final_rotation))

    assert (
        final_position_error_norm
        < POSITION_TOLERANCE
    )

    assert (
        final_orientation_error_angle
        < ORIENTATION_TOLERANCE
    )

    print()
    print("Panda iterative 6D pose IK: PASSED")


if __name__ == "__main__":
    main()