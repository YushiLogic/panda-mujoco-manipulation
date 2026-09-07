"""调用正式位置 IK 接口，让 Panda 末端移动到目标位置。"""

import numpy as np

from panda_mujoco.ik import solve_position_ik
from panda_mujoco.kinematics import get_ee_position
from panda_mujoco.simulation import PandaScene


TARGET_OFFSET = np.array(
    [0.020, -0.010, 0.015]
)

DAMPING = 0.05

# 末端位置误差小于 0.1 mm 时认为收敛。
POSITION_TOLERANCE = 1e-4

MAX_ITERATIONS = 50

# 每次迭代中，每个关节最多改变 0.1 rad。
MAX_JOINT_STEP = 0.1


def main() -> None:
    scene = PandaScene()

    initial_position = get_ee_position(scene)
    target_position = initial_position + TARGET_OFFSET

    initial_error_norm = np.linalg.norm(
        target_position - initial_position
    )

    # 示例不再自己实现迭代循环，而是调用正式 IK 接口。
    result = solve_position_ik(
        scene,
        target_position,
        damping=DAMPING,
        tolerance=POSITION_TOLERANCE,
        max_iterations=MAX_ITERATIONS,
        max_joint_step=MAX_JOINT_STEP,
    )

    error_reduction = (
        1.0
        - result.final_error_norm / initial_error_norm
    )

    np.set_printoptions(
        precision=9,
        suppress=True,
    )

    print("Initial position:")
    print(initial_position)

    print()
    print("Target position:")
    print(target_position)

    print()
    print("Final position:")
    print(result.final_position)

    print()
    print("Final position error:")
    print(result.final_error)

    print()
    print(f"DLS updates used: {result.iterations}")
    print(f"Initial error norm: {initial_error_norm:.6e} m")
    print(
        "Final error norm:   "
        f"{result.final_error_norm:.6e} m"
    )
    print(f"Error reduction:    {error_reduction:.2%}")
    print(f"Success:            {result.success}")

    assert result.success
    assert result.final_error_norm < initial_error_norm
    assert np.all(np.isfinite(result.final_position))

    print()
    print("Panda iterative position IK: PASSED")


if __name__ == "__main__":
    main()