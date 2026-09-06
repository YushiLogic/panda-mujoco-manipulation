"""使用中心有限差分验证 MuJoCo 的末端位置 Jacobian。"""

import mujoco
import numpy as np

from panda_mujoco.kinematics import (
    ARM_JOINT_NAMES,
    get_ee_jacobian,
    get_ee_position,
)
from panda_mujoco.simulation import PandaScene


# 对关节角施加的微小扰动，单位是弧度。
EPSILON = 1e-6


def main() -> None:
    scene = PandaScene()
    model = scene.model
    data = scene.data

    # ---------- 方法一：MuJoCo解析 Jacobian ----------

    analytical_jacobian, _ = get_ee_jacobian(scene)

    # ---------- 方法二：中心有限差分 ----------

    numerical_jacobian = np.zeros((3, 7))

    # 保存计算前的完整 qpos，防止不同关节的扰动互相影响。
    reference_qpos = data.qpos.copy()

    for column, joint_name in enumerate(ARM_JOINT_NAMES):
        # 根据关节名称查找它在模型中的关节ID。
        joint_id = model.joint(joint_name).id

        # qpos_address：
        # 该关节角度保存在 data.qpos 的哪个位置。
        qpos_address = model.jnt_qposadr[joint_id]

        # 将状态恢复到参考姿态，再给当前关节增加 epsilon。
        data.qpos[:] = reference_qpos
        data.qpos[qpos_address] += EPSILON

        # qpos 被手动修改后，需要重新计算末端位置等派生量。
        mujoco.mj_forward(model, data)

        position_plus = get_ee_position(scene)

        # 再从参考姿态出发，让当前关节减少 epsilon。
        data.qpos[:] = reference_qpos
        data.qpos[qpos_address] -= EPSILON
        mujoco.mj_forward(model, data)

        position_minus = get_ee_position(scene)

        # 中心有限差分：
        # 当前这一列表示末端位置对该关节角的偏导数。
        numerical_jacobian[:, column] = (
            position_plus - position_minus
        ) / (2.0 * EPSILON)

    # 计算完成后恢复场景，避免将最后一次扰动留在 data 中。
    data.qpos[:] = reference_qpos
    mujoco.mj_forward(model, data)

    # ---------- 比较两种方法 ----------

    error_matrix = numerical_jacobian - analytical_jacobian
    maximum_error = np.max(np.abs(error_matrix))

    np.set_printoptions(precision=6, suppress=True)

    print("MuJoCo analytical position Jacobian:")
    print(analytical_jacobian)
    print()

    print("Finite-difference position Jacobian:")
    print(numerical_jacobian)
    print()

    print("Difference:")
    print(error_matrix)
    print()

    # 分别显示每个关节所在列的最大误差。
    for column, joint_name in enumerate(ARM_JOINT_NAMES):
        column_error = np.max(np.abs(error_matrix[:, column]))

        print(
            f"{joint_name}: "
            f"maximum error={column_error:.3e}"
        )

    print()
    print(f"Overall maximum error: {maximum_error:.3e}")

    assert analytical_jacobian.shape == (3, 7)
    assert numerical_jacobian.shape == (3, 7)
    assert np.all(np.isfinite(numerical_jacobian))

    # 有限差分不是精确运算，因此用容差判断，而不是直接使用 ==。
    assert maximum_error < 1e-7

    print()
    print("Finite-difference Jacobian check: PASSED")


if __name__ == "__main__":
    main()
