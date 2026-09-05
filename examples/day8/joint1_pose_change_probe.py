"""验证 joint1 旋转与末端位姿变化之间的关系。"""

import mujoco
import numpy as np

from panda_mujoco.kinematics import get_ee_pose
from panda_mujoco.simulation import PandaScene


def rotation_about_world_z(angle: float) -> np.ndarray:
    """返回绕世界 z 轴旋转 angle 弧度的旋转矩阵。"""

    cosine = np.cos(angle)
    sine = np.sin(angle)

    rotation_z = np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    return rotation_z


def main() -> None:
    """改变 joint1，并比较理论末端位姿与 MuJoCo 计算结果。"""

    scene = PandaScene()

    # 读取 joint1 变化前的末端位姿。
    position_before, rotation_before = get_ee_pose(scene)

    # joint1 绕 base/world 的 z 轴旋转 0.2 rad。
    joint1_change = 0.2

    # 直接改变 qpos，只研究正运动学关系。
    scene.data.qpos[0] += joint1_change

    # 根据新 qpos 更新末端位姿，不推进仿真时间。
    mujoco.mj_forward(scene.model, scene.data)

    # MuJoCo 计算出的实际结果。
    position_after, rotation_after = get_ee_pose(scene)

    # 理论上的绕世界 z 轴旋转矩阵。
    rotation_z = rotation_about_world_z(joint1_change)

    # 当前模型的 joint1 轴经过世界原点，所以整个末端位置
    # 应当绕世界 z 轴旋转。
    expected_position = rotation_z @ position_before

    # 整个末端坐标系也随 joint1 一起绕世界 z 轴旋转。
    expected_rotation = rotation_z @ rotation_before

    position_error = float(
        np.max(np.abs(position_after - expected_position))
    )
    rotation_error = float(
        np.max(np.abs(rotation_after - expected_rotation))
    )

    np.set_printoptions(precision=6, suppress=True)

    print(f"joint1 change: {joint1_change:.3f} rad")
    print(
        f"joint1 change: "
        f"{np.degrees(joint1_change):.3f} degrees"
    )

    print("\nPosition before:")
    print(position_before)

    print("\nPosition after, calculated by MuJoCo:")
    print(position_after)

    print("\nPosition after, predicted by Rz:")
    print(expected_position)

    print("\nPosition maximum error:")
    print(f"{position_error:.3e} m")

    print("\nRotation maximum error:")
    print(f"{rotation_error:.3e}")

    # MuJoCo 的正运动学结果应与理论旋转一致。
    np.testing.assert_allclose(
        position_after,
        expected_position,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        rotation_after,
        expected_rotation,
        atol=1e-12,
    )

    print("\nJoint1 pose change check: PASSED")


if __name__ == "__main__":
    main()