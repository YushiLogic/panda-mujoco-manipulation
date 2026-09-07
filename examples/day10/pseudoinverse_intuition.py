"""用一个简化问题理解普通逆矩阵、伪逆和零空间。"""

import numpy as np


def main() -> None:
    # 一个末端任务，两个关节输入。
    #
    # 末端移动量：
    #     delta_x = delta_q1 + delta_q2
    #
    # shape = (1, 2)，即1行2列。
    jacobian = np.array(
        [[1.0, 1.0]]
    )

    # 希望末端移动0.1。
    # shape = (1,)。
    target_displacement = np.array([0.1])

    print("Jacobian:")
    print(jacobian)
    print(f"shape: {jacobian.shape}")
    print()

    # 普通逆矩阵只适用于方阵。
    try:
        ordinary_inverse = np.linalg.inv(jacobian)
        print(ordinary_inverse)
    except np.linalg.LinAlgError as error:
        print("Ordinary inverse is unavailable:")
        print(error)

    print()

    # 伪逆允许处理非方阵。
    pseudoinverse = np.linalg.pinv(jacobian)

    print("Pseudoinverse:")
    print(pseudoinverse)
    print(f"shape: {pseudoinverse.shape}")
    print()

    # 使用伪逆求最小范数解。
    minimum_norm_solution = (
        pseudoinverse @ target_displacement
    )

    print("Minimum-norm solution:")
    print(minimum_norm_solution)

    predicted_displacement = (
        jacobian @ minimum_norm_solution
    )

    print("Predicted end-effector displacement:")
    print(predicted_displacement)

    print(
        "Solution norm:",
        np.linalg.norm(minimum_norm_solution),
    )
    print()

    # 另外两组同样能够完成任务的解。
    candidate_solutions = (
        np.array([0.10, 0.00]),
        np.array([0.05, 0.05]),
        np.array([0.20, -0.10]),
    )

    print("Candidate solutions:")

    for candidate in candidate_solutions:
        task_result = jacobian @ candidate
        solution_norm = np.linalg.norm(candidate)

        print(
            f"delta_q={candidate} | "
            f"J @ delta_q={task_result} | "
            f"norm={solution_norm:.6f}"
        )

    print()

    # 这个方向属于 Jacobian 的零空间。
    null_space_direction = np.array([1.0, -1.0])

    print("Null-space direction:")
    print(null_space_direction)

    print("J @ null-space direction:")
    print(jacobian @ null_space_direction)
    print()

    # 在伪逆解上添加一部分零空间运动。
    scale = 0.02

    alternative_solution = (
        minimum_norm_solution
        + scale * null_space_direction
    )

    print("Alternative solution:")
    print(alternative_solution)

    print("Alternative task result:")
    print(jacobian @ alternative_solution)

    print(
        "Alternative solution norm:",
        np.linalg.norm(alternative_solution),
    )

    # 自动检查。
    np.testing.assert_allclose(
        minimum_norm_solution,
        np.array([0.05, 0.05]),
    )

    np.testing.assert_allclose(
        predicted_displacement,
        target_displacement,
    )

    np.testing.assert_allclose(
        jacobian @ null_space_direction,
        np.array([0.0]),
    )

    np.testing.assert_allclose(
        jacobian @ alternative_solution,
        target_displacement,
    )

    assert (
        np.linalg.norm(minimum_norm_solution)
        < np.linalg.norm(alternative_solution)
    )

    print()
    print("Pseudoinverse intuition check: PASSED")


if __name__ == "__main__":
    main()