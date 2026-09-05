"""练习末端坐标系与世界坐标系之间的点变换。"""

import numpy as np

from panda_mujoco.kinematics import get_ee_pose
from panda_mujoco.simulation import PandaScene


def main() -> None:
    """将末端局部坐标点转换到世界坐标，再转换回来。"""

    scene = PandaScene()

    # position：末端原点在 world frame 中的位置。
    # rotation：末端坐标系相对于 world frame 的旋转矩阵。
    position, rotation = get_ee_pose(scene)

    # 定义四个末端局部坐标点。
    # 单位都是米。
    local_points = {
        "origin": np.array([0.0, 0.0, 0.0]),
        "x_axis_10cm": np.array([0.1, 0.0, 0.0]),
        "y_axis_10cm": np.array([0.0, 0.1, 0.0]),
        "z_axis_10cm": np.array([0.0, 0.0, 0.1]),
    }

    np.set_printoptions(precision=6, suppress=True)

    print("End-effector origin in world frame:")
    print(position)

    for name, point_local in local_points.items():
        # 从末端局部坐标系转换到世界坐标系：
        #
        # 1. rotation @ point_local：旋转局部向量；
        # 2. + position：加上末端原点的世界位置。
        point_world = position + rotation @ point_local

        # 从世界坐标系转换回末端局部坐标系：
        #
        # 1. point_world - position：去掉平移；
        # 2. rotation.T：执行反向旋转。
        point_local_recovered = (
            rotation.T @ (point_world - position)
        )

        recovery_error = float(
            np.linalg.norm(
                point_local_recovered - point_local
            )
        )

        print(f"\n{name}")
        print(f"local point:     {point_local}")
        print(f"world point:     {point_world}")
        print(f"recovered point: {point_local_recovered}")
        print(f"recovery error:  {recovery_error:.3e} m")

        # 正向转换再反向转换，应当恢复原始局部点。
        np.testing.assert_allclose(
            point_local_recovered,
            point_local,
            atol=1e-12,
        )

        # 旋转不能改变向量长度。
        np.testing.assert_allclose(
            np.linalg.norm(point_world - position),
            np.linalg.norm(point_local),
            atol=1e-12,
        )

    print("\nFrame transformation check: PASSED")


if __name__ == "__main__":
    main()