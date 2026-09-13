"""对前5个随机目标进行位置IK预检。"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from panda_mujoco.ik import solve_position_ik
from panda_mujoco.kinematics import (
    ARM_JOINT_NAMES,
    get_ee_position,
)
from panda_mujoco.simulation import PandaScene
from target_sampling_probe import generate_target_positions, RANDOM_SEED, TARGET_COUNT

PREFLIGHT_COUNT = 5


@dataclass
class KinematicCaseResult:
    """保存一个目标点的IK评测结果。"""

    case_id: int
    target_position: np.ndarray

    # solve_position_ik() 自己报告的求解状态。
    ik_success: bool

    # 综合考虑IK、有限数值和关节限位后的最终状态。
    success: bool

    failure_reason: str
    iterations: int
    final_error_m: float
    solve_time_s: float
    all_finite: bool
    joint_limit_violation: bool


def get_arm_joint_state_and_limits(
    scene: PandaScene,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取7个手臂关节的位置和上下限。"""

    joint_ids = np.array(
        [
            scene.model.joint(name).id
            for name in ARM_JOINT_NAMES
        ],
        dtype=int,
    )

    qpos_addresses = np.array(
        [
            scene.model.jnt_qposadr[joint_id]
            for joint_id in joint_ids
        ],
        dtype=int,
    )

    joint_positions = (
        scene.data.qpos[qpos_addresses].copy()
    )

    lower_limits = (
        scene.model.jnt_range[joint_ids, 0].copy()
    )

    upper_limits = (
        scene.model.jnt_range[joint_ids, 1].copy()
    )

    return (
        joint_positions,
        lower_limits,
        upper_limits,
    )


def evaluate_target(
    case_id: int,
    target_position: np.ndarray,
) -> KinematicCaseResult:
    """从home姿态开始评测一个目标点。"""

    # 每个目标都使用一个全新的场景。
    #
    # 这样每次IK都从完全相同的home姿态开始，
    # 避免前一个目标的最终姿态影响下一个目标。
    scene = PandaScene()

    start_time = perf_counter()

    ik_result = solve_position_ik(
        scene,
        target_position,
        damping=0.05,
        tolerance=1e-4,
        max_iterations=100,
        max_joint_step=0.1,
    )

    solve_time_s = perf_counter() - start_time

    (
        joint_positions,
        lower_limits,
        upper_limits,
    ) = get_arm_joint_state_and_limits(scene)

    # 检查求解后是否出现NaN或Inf。
    all_finite = bool(
        np.all(np.isfinite(scene.data.qpos))
        and np.all(np.isfinite(scene.data.qvel))
        and np.all(np.isfinite(scene.data.ctrl))
        and np.all(np.isfinite(ik_result.final_position))
        and np.all(np.isfinite(ik_result.final_error))
        and np.isfinite(ik_result.final_error_norm)
    )

    # 容许1e-9的浮点误差。
    joint_limit_violation = bool(
        np.any(
            joint_positions
            < lower_limits - 1e-9
        )
        or np.any(
            joint_positions
            > upper_limits + 1e-9
        )
    )

    # 一个案例通过需要同时满足三个条件：
    # 1. IK到达目标；
    # 2. 没有NaN或Inf；
    # 3. 没有违反关节限位。
    success = bool(
        ik_result.success
        and all_finite
        and not joint_limit_violation
    )

    # 给失败案例分配明确的失败原因。
    if not all_finite:
        failure_reason = "non_finite_state"
    elif joint_limit_violation:
        failure_reason = "joint_limit_violation"
    elif not ik_result.success:
        failure_reason = "ik_not_converged"
    else:
        failure_reason = "none"

    return KinematicCaseResult(
        case_id=case_id,
        target_position=target_position.copy(),
        ik_success=ik_result.success,
        success=success,
        failure_reason=failure_reason,
        iterations=ik_result.iterations,
        final_error_m=ik_result.final_error_norm,
        solve_time_s=solve_time_s,
        all_finite=all_finite,
        joint_limit_violation=joint_limit_violation,
    )


def main() -> None:
    home_scene = PandaScene()
    home_position = get_ee_position(home_scene)

    targets = generate_target_positions(
        home_position,
        target_count=TARGET_COUNT,
        seed=RANDOM_SEED,
    )

    results = []

    print("Day 14 kinematic reach preflight")
    print()

    for case_id, target_position in enumerate(
        targets[:PREFLIGHT_COUNT]
    ):
        result = evaluate_target(
            case_id,
            target_position,
        )

        results.append(result)

        print(
            f"case {result.case_id:02d}"
            f" | success={result.success}"
            f" | iterations={result.iterations:2d}"
            f" | error={result.final_error_m * 1000:.6f} mm"
            f" | finite={result.all_finite}"
            f" | limit_violation="
            f"{result.joint_limit_violation}"
            f" | time={result.solve_time_s:.6f} s"
            f" | reason={result.failure_reason}"
        )

    success_count = sum(
        result.success
        for result in results
    )

    maximum_error_m = max(
        result.final_error_m
        for result in results
    )

    print()
    print("Preflight summary")
    print(
        f"success count: "
        f"{success_count}/{PREFLIGHT_COUNT}"
    )
    print(
        f"maximum error: "
        f"{maximum_error_m * 1000:.6f} mm"
    )

    assert success_count == PREFLIGHT_COUNT
    assert maximum_error_m < 1e-4
    assert all(
        result.all_finite
        for result in results
    )
    assert not any(
        result.joint_limit_violation
        for result in results
    )

    print()
    print("Kinematic reach preflight: PASSED")


if __name__ == "__main__":
    main()