"""用简化矩阵理解伪逆在奇异附近的问题和阻尼作用。"""

import numpy as np

from panda_mujoco.ik import damped_least_squares

def main() -> None:
    # 第二个方向的运动能力很弱。
    jacobian = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.02],
        ]
    )

    # 希望末端只沿第二个方向移动0.01。
    target_displacement = np.array([0.0, 0.01])

    singular_values = np.linalg.svd(
        jacobian,
        compute_uv=False,
    )

    print("Jacobian:")
    print(jacobian)

    print()
    print("Singular values:")
    print(singular_values)

    # ---------- 伪逆解 ----------

    pseudoinverse_solution = (
        np.linalg.pinv(jacobian)
        @ target_displacement
    )

    pseudoinverse_result = (
        jacobian @ pseudoinverse_solution
    )

    print()
    print("Pseudoinverse solution:")
    print(pseudoinverse_solution)

    print("Pseudoinverse task result:")
    print(pseudoinverse_result)

    print(
        "Pseudoinverse solution norm:",
        np.linalg.norm(pseudoinverse_solution),
    )

    # ---------- 不同阻尼系数 ----------

    damping_values = (0.01, 0.05, 0.10)

    print()
    print("Damped least-squares solutions:")

    for damping in damping_values:
        delta_q = damped_least_squares(
            jacobian,
            target_displacement,
            damping,
        )

        achieved_displacement = jacobian @ delta_q

        remaining_error = (
            target_displacement
            - achieved_displacement
        )

        print()
        print(f"damping={damping:.2f}")
        print(f"delta_q={delta_q}")
        print(
            "achieved displacement="
            f"{achieved_displacement}"
        )
        print(f"remaining error={remaining_error}")
        print(
            "solution norm="
            f"{np.linalg.norm(delta_q):.6f}"
        )

    # 使用中间阻尼值进行自动检查。
    damped_solution = damped_least_squares(
        jacobian,
        target_displacement,
        damping=0.05,
    )

    damped_result = jacobian @ damped_solution

    assert np.all(np.isfinite(damped_solution))

    # 阻尼解的关节变化应该比伪逆解小。
    assert (
        np.linalg.norm(damped_solution)
        < np.linalg.norm(pseudoinverse_solution)
    )

    # 阻尼会保留一部分末端误差，所以不要求单步完全到达。
    assert (
        np.linalg.norm(
            target_displacement - damped_result
        )
        > 0.0
    )

    print()
    print("Damped least-squares intuition check: PASSED")


if __name__ == "__main__":
    main()