"""验证 IK 规划场景和动力学控制场景相互独立。"""

import numpy as np

from panda_mujoco.ik import solve_position_ik
from panda_mujoco.kinematics import get_ee_position
from panda_mujoco.simulation import PandaScene


TARGET_OFFSET = np.array(
    [0.020, -0.010, 0.015]
)


def main() -> None:
    # planning_scene 只用于计算目标关节角。
    planning_scene = PandaScene()

    # control_scene 将在下一步通过 ctrl 和 mj_step 真正运动。
    control_scene = PandaScene()

    initial_position = get_ee_position(control_scene)
    target_position = initial_position + TARGET_OFFSET

    initial_joint_positions = (
        control_scene.data.qpos[:7].copy()
    )

    result = solve_position_ik(
        planning_scene,
        target_position,
        damping=0.05,
        tolerance=1e-4,
        max_iterations=50,
        max_joint_step=0.1,
    )

    # IK 求出的关节角需要立即 copy。
    # 否则可能只是指向 MuJoCo 内部 data.qpos 的视图。
    target_joint_positions = (
        planning_scene.data.qpos[:7].copy()
    )

    control_joint_positions = (
        control_scene.data.qpos[:7].copy()
    )

    np.set_printoptions(
        precision=9,
        suppress=True,
    )

    print("Initial end-effector position:")
    print(initial_position)

    print()
    print("Target end-effector position:")
    print(target_position)

    print()
    print("IK final end-effector position:")
    print(result.final_position)

    print()
    print("Initial joint positions:")
    print(initial_joint_positions)

    print()
    print("IK target joint positions:")
    print(target_joint_positions)

    print()
    print("Required joint displacement:")
    print(
        target_joint_positions
        - initial_joint_positions
    )

    print()
    print("Control-scene joint positions:")
    print(control_joint_positions)

    print()
    print(f"IK updates: {result.iterations}")
    print(
        "IK final error: "
        f"{result.final_error_norm:.6e} m"
    )

    # IK 应当成功找到目标关节角。
    assert result.success

    # IK 结果应该与初始关节角不同。
    assert not np.allclose(
        target_joint_positions,
        initial_joint_positions,
    )

    # planning_scene 的 IK 不应该改变 control_scene。
    np.testing.assert_array_equal(
        control_joint_positions,
        initial_joint_positions,
    )

    # IK 目标必须落在七个手臂执行器允许的控制范围内。
    control_ranges = (
        control_scene.model.actuator_ctrlrange[:7]
    )

    assert np.all(
        target_joint_positions >= control_ranges[:, 0]
    )
    assert np.all(
        target_joint_positions <= control_ranges[:, 1]
    )

    print()
    print("IK target separation check: PASSED")


if __name__ == "__main__":
    main()