"""读取并检查 Panda 末端执行器的 Jacobian。"""

import numpy as np

from panda_mujoco.kinematics import get_ee_jacobian
from panda_mujoco.simulation import PandaScene


def main() -> None:
    # 创建场景后，机械臂已经被 reset 到 home 状态。
    # 此时 MuJoCo 已经计算好了当前姿态下的运动学数据。
    scene = PandaScene()

    # 读取末端 site 相对于7个机械臂关节的 Jacobian。
    arm_jac_position, arm_jac_rotation = get_ee_jacobian(scene)

    # 把线速度 Jacobian 和角速度 Jacobian上下拼接，
    # 得到一个 6×7 的末端完整 Jacobian。
    arm_jacobian = np.vstack(
        (arm_jac_position, arm_jac_rotation)
    )

    np.set_printoptions(precision=6, suppress=True)

    print(f"model.nq: {scene.model.nq}")
    print(f"model.nv: {scene.model.nv}")
    print()

    print("Position Jacobian of the Panda arm:")
    print(arm_jac_position)
    print(f"shape: {arm_jac_position.shape}")
    print()

    print("Rotational Jacobian of the Panda arm:")
    print(arm_jac_rotation)
    print(f"shape: {arm_jac_rotation.shape}")
    print()

    print("Complete 6D Jacobian:")
    print(arm_jacobian)
    print(f"shape: {arm_jacobian.shape}")
    print()

    # 最基本的自动检查。
    assert arm_jac_position.shape == (3, 7)
    assert arm_jac_rotation.shape == (3, 7)
    assert arm_jacobian.shape == (6, 7)
    assert np.all(np.isfinite(arm_jacobian))

    print()
    print("Jacobian probe: PASSED")


if __name__ == "__main__":
    main()
