"""使用完整6D Jacobian执行一次末端位姿DLS更新。"""

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


def main() -> None:
    scene = PandaScene()
    model = scene.model
    data = scene.data

    position_before, rotation_before = get_ee_pose(scene)

    # 目标位置与当前位置相同：
    # 我们要求末端旋转时尽量保持位置不动。
    target_position = position_before.copy()

    world_z_rotation = Rotation.from_euler(
        "z",
        TARGET_ANGLE_DEGREES,
        degrees=True,
    ).as_matrix()

    target_rotation = (
        world_z_rotation @ rotation_before
    )

    position_error = (
        target_position - position_before
    )

    orientation_error = rotation_error(
        target_rotation,
        rotation_before,
    )

    # 把位置和姿态误差连接成6维任务误差。
    pose_error = np.concatenate(
        [position_error, orientation_error]
    )

    linear_jacobian, angular_jacobian = (
        get_ee_jacobian(scene)
    )

    # 前3行描述线速度，后3行描述角速度。
    pose_jacobian = np.vstack(
        [linear_jacobian, angular_jacobian]
    )

    raw_delta_q = damped_least_squares(
        pose_jacobian,
        pose_error,
        damping=DAMPING,
    )

    # 如果某个关节更新超过最大单步值，
    # 对整个向量统一缩放，保留DLS给出的运动方向。
    maximum_raw_step = np.max(
        np.abs(raw_delta_q)
    )

    step_scale = min(
        1.0,
        MAX_JOINT_STEP / maximum_raw_step,
    )

    delta_q = raw_delta_q * step_scale

    predicted_pose_change = (
        pose_jacobian @ delta_q
    )

    predicted_position_change = (
        predicted_pose_change[:3]
    )

    predicted_rotation_change = (
        predicted_pose_change[3:]
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

    joint_positions_before = (
        data.qpos[qpos_addresses].copy()
    )

    joint_positions_after = (
        joint_positions_before + delta_q
    )

    lower_limits = model.jnt_range[joint_ids, 0]
    upper_limits = model.jnt_range[joint_ids, 1]

    assert np.all(
        joint_positions_after >= lower_limits
    )

    assert np.all(
        joint_positions_after <= upper_limits
    )

    data.qpos[qpos_addresses] = joint_positions_after
    mujoco.mj_forward(model, data)

    position_after, rotation_after = get_ee_pose(scene)

    final_position_error = (
        target_position - position_after
    )

    final_orientation_error = rotation_error(
        target_rotation,
        rotation_after,
    )

    position_error_norm = np.linalg.norm(
        final_position_error
    )

    initial_orientation_error_angle = np.linalg.norm(
        orientation_error
    )

    final_orientation_error_angle = np.linalg.norm(
        final_orientation_error
    )

    np.set_printoptions(
        precision=9,
        suppress=True,
    )

    print("Pose Jacobian shape:")
    print(pose_jacobian.shape)

    print()
    print("Initial 6D pose error:")
    print(pose_error)

    print()
    print("Raw DLS joint update:")
    print(raw_delta_q)

    print()
    print("Uniform step scale:")
    print(f"{step_scale:.9f}")

    print()
    print("Limited joint update:")
    print(delta_q)

    print()
    print("Predicted position change:")
    print(predicted_position_change)

    print()
    print("Predicted rotation change:")
    print(predicted_rotation_change)

    print()
    print("Actual position change:")
    print(position_after - position_before)

    print()
    print("Final position error:")
    print(final_position_error)

    print()
    print("Final orientation error:")
    print(final_orientation_error)

    print()
    print(
        "Final position error norm: "
        f"{position_error_norm:.6e} m"
    )

    print(
        "Initial orientation error: "
        f"{np.rad2deg(initial_orientation_error_angle):.6f} degrees"
    )

    print(
        "Final orientation error:   "
        f"{np.rad2deg(final_orientation_error_angle):.6f} degrees"
    )

    print(
        "Maximum joint update:      "
        f"{np.max(np.abs(delta_q)):.6f} rad"
    )

    assert pose_jacobian.shape == (6, 7)
    assert np.all(np.isfinite(delta_q))

    assert (
        np.max(np.abs(delta_q))
        <= MAX_JOINT_STEP + 1e-12
    )

    # 加入位置约束后，一次更新的位置漂移应小于5 mm。
    assert position_error_norm < 0.005

    # 一次受限更新不要求直接到达，但姿态误差必须下降。
    assert (
        final_orientation_error_angle
        < initial_orientation_error_angle
    )

    print()
    print("Panda single 6D DLS step: PASSED")


if __name__ == "__main__":
    main()