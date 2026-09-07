"""在 Panda 上执行一次运动学 DLS 位置 IK 更新。"""

import mujoco
import numpy as np

from panda_mujoco.ik import damped_least_squares
from panda_mujoco.kinematics import (
    ARM_JOINT_NAMES,
    get_ee_jacobian,
    get_ee_position,
)
from panda_mujoco.simulation import PandaScene


# 希望末端从当前位置产生的世界坐标位移，单位为米。
TARGET_OFFSET = np.array(
    [0.020, -0.010, 0.015]
)

DAMPING = 0.05


def main() -> None:
    scene = PandaScene()
    model = scene.model
    data = scene.data

    position_before = get_ee_position(scene)

    target_position = (
        position_before + TARGET_OFFSET
    )

    # 当前位置到目标位置的误差。
    position_error = (
        target_position - position_before
    )

    linear_jacobian, _ = get_ee_jacobian(scene)

    # 计算一次关节角更新。
    delta_q = damped_least_squares(
        linear_jacobian,
        position_error,
        damping=DAMPING,
    )

    # Jacobian 对这一次位移的线性预测。
    predicted_displacement = (
        linear_jacobian @ delta_q
    )

    predicted_position = (
        position_before + predicted_displacement
    )

    # 根据关节名称查找7个关节在 qpos 中的位置。
    qpos_addresses = []
    joint_ids = []

    for joint_name in ARM_JOINT_NAMES:
        joint_id = model.joint(joint_name).id
        qpos_address = model.jnt_qposadr[joint_id]

        joint_ids.append(joint_id)
        qpos_addresses.append(qpos_address)

    qpos_addresses = np.array(qpos_addresses)

    joint_positions_before = (
        data.qpos[qpos_addresses].copy()
    )

    joint_positions_after = (
        joint_positions_before + delta_q
    )

    # 在真正修改 qpos 前检查关节限位。
    for index, joint_id in enumerate(joint_ids):
        lower, upper = model.jnt_range[joint_id]
        candidate = joint_positions_after[index]

        assert lower <= candidate <= upper, (
            f"{ARM_JOINT_NAMES[index]} target "
            f"{candidate} exceeds [{lower}, {upper}]"
        )

    # 直接写入关节角，进行运动学采样。
    data.qpos[qpos_addresses] = joint_positions_after

    # 根据新的 qpos 重新计算末端位置。
    mujoco.mj_forward(model, data)

    position_after = get_ee_position(scene)

    actual_displacement = (
        position_after - position_before
    )

    remaining_error = (
        target_position - position_after
    )

    # Jacobian线性预测与真实正运动学变化之间的差异。
    linearization_error = (
        actual_displacement - predicted_displacement
    )

    initial_error_norm = np.linalg.norm(
        position_error
    )

    final_error_norm = np.linalg.norm(
        remaining_error
    )

    error_reduction = (
        1.0 - final_error_norm / initial_error_norm
    )

    np.set_printoptions(precision=9, suppress=True)

    print("Position before:")
    print(position_before)

    print()
    print("Target position:")
    print(target_position)

    print()
    print("Initial position error:")
    print(position_error)

    print()
    print("DLS joint update delta_q:")
    print(delta_q)

    print()
    print("Predicted displacement from Jp @ delta_q:")
    print(predicted_displacement)

    print()
    print("Predicted position:")
    print(predicted_position)

    print()
    print("Actual position from forward kinematics:")
    print(position_after)

    print()
    print("Actual displacement:")
    print(actual_displacement)

    print()
    print("Linearization error:")
    print(linearization_error)

    print()
    print("Remaining position error:")
    print(remaining_error)

    print()
    print(f"Initial error norm: {initial_error_norm:.6e} m")
    print(f"Final error norm:   {final_error_norm:.6e} m")
    print(f"Error reduction:    {error_reduction:.2%}")

    assert np.all(np.isfinite(delta_q))
    assert np.all(np.isfinite(position_after))

    # 一次DLS更新应该让末端更接近目标。
    assert final_error_norm < initial_error_norm

    # 当前实验参数下，误差应至少下降80%。
    assert error_reduction > 0.80

    print()
    print("Panda single DLS step: PASSED")


if __name__ == "__main__":
    main()