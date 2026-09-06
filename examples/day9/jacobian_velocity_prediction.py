"""使用 Jacobian 预测关节运动产生的末端线速度。"""

import mujoco
import numpy as np

from panda_mujoco.kinematics import (
    ARM_JOINT_NAMES,
    get_ee_jacobian,
    get_ee_position,
)
from panda_mujoco.simulation import PandaScene


# 一组假设的机械臂关节速度，单位为 rad/s。
JOINT_VELOCITIES = np.array(
    [0.10, -0.05, 0.08, 0.04, -0.03, 0.02, 0.06]
)

# 用于中心有限差分的极短时间，单位为秒。
DELTA_TIME = 1e-6


def main() -> None:
    scene = PandaScene()
    model = scene.model
    data = scene.data

    # 保存计算开始前的标准姿态。
    reference_qpos = data.qpos.copy()

    # 读取 home 姿态下的末端位置 Jacobian。
    linear_jacobian, _ = get_ee_jacobian(scene)

    # 根据 v = Jp @ qdot 预测末端线速度。
    predicted_velocity = (
        linear_jacobian @ JOINT_VELOCITIES
    )

    # 找出7个机械臂关节在 qpos 中的位置。
    qpos_addresses = []

    for joint_name in ARM_JOINT_NAMES:
        joint_id = model.joint(joint_name).id
        qpos_address = model.jnt_qposadr[joint_id]
        qpos_addresses.append(qpos_address)

    qpos_addresses = np.array(qpos_addresses)

    # ---------- 用中心有限差分测量实际速度 ----------

    # q_plus = q + qdot * dt
    data.qpos[:] = reference_qpos
    data.qpos[qpos_addresses] = (
        reference_qpos[qpos_addresses]
        + JOINT_VELOCITIES * DELTA_TIME
    )
    mujoco.mj_forward(model, data)
    position_plus = get_ee_position(scene)

    # q_minus = q - qdot * dt
    data.qpos[:] = reference_qpos
    data.qpos[qpos_addresses] = (
        reference_qpos[qpos_addresses]
        - JOINT_VELOCITIES * DELTA_TIME
    )
    mujoco.mj_forward(model, data)
    position_minus = get_ee_position(scene)

    # v ≈ [p(q + qdot*dt) - p(q - qdot*dt)] / (2*dt)
    measured_velocity = (
        position_plus - position_minus
    ) / (2.0 * DELTA_TIME)

    # 恢复场景原始状态。
    data.qpos[:] = reference_qpos
    mujoco.mj_forward(model, data)

    velocity_error = (
        measured_velocity - predicted_velocity
    )
    maximum_error = np.max(np.abs(velocity_error))

    np.set_printoptions(precision=9, suppress=True)

    print("Joint velocities qdot:")
    print(JOINT_VELOCITIES)
    print()

    print("Velocity contribution from each joint:")

    # Jp 的一列乘对应关节速度，就是该关节的速度贡献。
    for column, joint_name in enumerate(ARM_JOINT_NAMES):
        contribution = (
            linear_jacobian[:, column]
            * JOINT_VELOCITIES[column]
        )

        print(f"{joint_name}: {contribution}")

    print()
    print("Predicted velocity from Jp @ qdot:")
    print(predicted_velocity)

    print()
    print("Measured velocity from finite difference:")
    print(measured_velocity)

    print()
    print("Velocity error:")
    print(velocity_error)

    print()
    print(f"Maximum velocity error: {maximum_error:.3e} m/s")

    assert predicted_velocity.shape == (3,)
    assert measured_velocity.shape == (3,)
    assert np.all(np.isfinite(predicted_velocity))
    assert maximum_error < 1e-7

    print()
    print("Jacobian velocity prediction: PASSED")


if __name__ == "__main__":
    main()