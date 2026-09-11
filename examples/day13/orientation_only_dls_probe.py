"""仅使用旋转 Jacobian 执行一次姿态 DLS 更新。"""

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
DAMPING = 0.05


def main() -> None:
    scene = PandaScene()
    model = scene.model
    data = scene.data

    position_before, rotation_before = get_ee_pose(scene)

    # 创建一个绕世界z轴旋转15度的目标姿态。
    world_z_rotation = Rotation.from_euler(
        "z",
        TARGET_ANGLE_DEGREES,
        degrees=True,
    ).as_matrix()

    target_rotation = (
        world_z_rotation @ rotation_before
    )

    # 当前姿态到目标姿态的旋转误差。
    initial_rotation_error = rotation_error(
        target_rotation,
        rotation_before,
    )

    # 这次只使用旋转Jacobain，不使用位置Jacobian。
    _, angular_jacobian = get_ee_jacobian(scene)

    delta_q = damped_least_squares(
        angular_jacobian,
        initial_rotation_error,
        damping=DAMPING,
    )

    # Jr @ delta_q 是一阶模型预测的末端旋转变化。
    predicted_rotation_change = (
        angular_jacobian @ delta_q
    )

    # 根据关节名称取得7个手臂关节的qpos地址和限位。
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

    # 修改qpos之前检查关节目标没有越限。
    lower_limits = model.jnt_range[joint_ids, 0]
    upper_limits = model.jnt_range[joint_ids, 1]

    assert np.all(
        joint_positions_after >= lower_limits
    )

    assert np.all(
        joint_positions_after <= upper_limits
    )

    # 直接写入qpos，只进行运动学验证，不推进仿真时间。
    data.qpos[qpos_addresses] = joint_positions_after
    mujoco.mj_forward(model, data)

    position_after, rotation_after = get_ee_pose(scene)

    # 末端实际发生的旋转变化。
    actual_rotation_change = rotation_error(
        rotation_after,
        rotation_before,
    )

    # 更新后距离目标姿态还剩多少误差。
    final_rotation_error = rotation_error(
        target_rotation,
        rotation_after,
    )

    initial_error_angle = np.linalg.norm(
        initial_rotation_error
    )

    final_error_angle = np.linalg.norm(
        final_rotation_error
    )

    position_drift = np.linalg.norm(
        position_after - position_before
    )

    orientation_linearization_error = np.linalg.norm(
        actual_rotation_change
        - predicted_rotation_change
    )

    np.set_printoptions(
        precision=9,
        suppress=True,
    )

    print("Position before:")
    print(position_before)

    print()
    print("Initial rotation error:")
    print(initial_rotation_error)

    print()
    print("Angular Jacobian shape:")
    print(angular_jacobian.shape)

    print()
    print("DLS joint update delta_q:")
    print(delta_q)

    print()
    print("Predicted rotation change from Jr @ delta_q:")
    print(predicted_rotation_change)

    print()
    print("Actual rotation change:")
    print(actual_rotation_change)

    print()
    print("Remaining rotation error:")
    print(final_rotation_error)

    print()
    print(
        "Initial orientation error: "
        f"{np.rad2deg(initial_error_angle):.6f} degrees"
    )

    print(
        "Final orientation error:   "
        f"{np.rad2deg(final_error_angle):.6f} degrees"
    )

    print(
        "Orientation linearization error: "
        f"{orientation_linearization_error:.6e} rad"
    )

    print(
        "End-effector position drift: "
        f"{position_drift:.6f} m"
    )

    assert np.all(np.isfinite(delta_q))
    assert final_error_angle < initial_error_angle
    assert np.rad2deg(final_error_angle) < 0.1
    assert orientation_linearization_error < 1e-3

    print()
    print("Orientation-only DLS step: PASSED")


if __name__ == "__main__":
    main()