"""读取并验证 Panda 末端 site 的位姿。负责如何显示并验证末端位姿"""

import numpy as np

from panda_mujoco.kinematics import get_ee_pose
from panda_mujoco.simulation import PandaScene

def main() -> None:
    """读取末端位置和旋转矩阵，并进行基本合法性检查。"""

    # 创建场景时会自动：
    # 1. 加载 Panda XML；
    # 2. 设置 home qpos；
    # 3. 调用 mj_forward() 更新正运动学结果。
    scene = PandaScene()
    # 通过项目的运动学接口读取末端位姿。
    # position 是 world frame 中的位置；
    # rotation 是末端坐标系相对于 world frame 的旋转矩阵。
    position,rotation=get_ee_pose(scene)

    # 为了让终端中的矩阵更容易阅读：
    # precision=6 表示显示 6 位小数；
    # suppress=True 表示很小的数字显示成 0，而不是科学计数法。
    np.set_printoptions(precision=6, suppress=True)

    print("End-effector position in world frame:")
    print(position)

    print("\nEnd-effector rotation matrix in world frame:")
    print(rotation)

    # 旋转矩阵的三列分别表示：
    # 末端局部 x、y、z 轴在 world frame 中的方向。
    print("\nEE local x-axis expressed in world frame:")
    print(rotation[:, 0])

    print("\nEE local y-axis expressed in world frame:")
    print(rotation[:, 1])

    print("\nEE local z-axis expressed in world frame:")
    print(rotation[:, 2])

    # 合法旋转矩阵应满足 R.T @ R = I。
    identity_result = rotation.T @ rotation
    identity_error = float(
        np.max(np.abs(identity_result - np.eye(3)))
    )

    # 合法的三维旋转矩阵还应满足 det(R) = +1。
    determinant = float(np.linalg.det(rotation))

    print("\nR.T @ R:")
    print(identity_result)

    print(f"\nMaximum orthogonality error: {identity_error:.3e}")
    print(f"det(R): {determinant:.12f}")

    # 基本形状检查。
    assert position.shape == (3,)
    assert rotation.shape == (3, 3)

    # 位置和姿态不能包含 NaN 或 Inf。
    assert np.all(np.isfinite(position))
    assert np.all(np.isfinite(rotation))

    # 验证旋转矩阵的正交性。
    np.testing.assert_allclose(
        identity_result,
        np.eye(3),
        atol=1e-9,
    )

    # 验证它是正常旋转而不是镜像变换。
    assert np.isclose(determinant, 1.0, atol=1e-9)

    print("\nEnd-effector pose check: PASSED")


if __name__ == "__main__":
    main()