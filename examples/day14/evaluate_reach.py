"""对 50 个固定目标执行 Panda 末端位置到达评测并导出 CSV。"""

import csv
from collections import Counter
from pathlib import Path

import numpy as np

from dynamic_reach_preflight import (
    DynamicCaseResult,
    evaluate_target,
)
from panda_mujoco.kinematics import get_ee_position
from panda_mujoco.simulation import PandaScene
from target_sampling_probe import (
    RANDOM_SEED,
    TARGET_COUNT,
    generate_target_positions,
)


# parents[0] = day14
# parents[1] = examples
# parents[2] = 项目根目录 panda_mujoco_manipulation
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    PROJECT_ROOT
    / "results"
    / "day14"
    / "reach_evaluation.csv"
)

# Day14 的正式验收阈值。
MINIMUM_SUCCESS_RATE = 0.90
MAXIMUM_MEAN_SUCCESS_ERROR_M = 0.02


def result_to_csv_row(
    result: DynamicCaseResult,
) -> dict[str, object]:
    """把一个 dataclass 结果转换成一行普通 CSV 数据。"""

    return {
        "case_id": result.case_id,
        "target_x_m": result.target_position[0],
        "target_y_m": result.target_position[1],
        "target_z_m": result.target_position[2],
        "ik_success": result.ik_success,
        "success": result.success,
        "failure_reason": result.failure_reason,
        "ik_iterations": result.ik_iterations,
        "ik_error_m": result.ik_error_m,
        "tracking_error_m": result.tracking_error_m,
        "max_joint_error_rad": result.max_joint_error_rad,
        "joint_velocity_norm_rad_s": (
            result.joint_velocity_norm_rad_s
        ),
        "all_finite": result.all_finite,
        "joint_limit_violation": (
            result.joint_limit_violation
        ),
        "simulation_time_s": result.simulation_time_s,
        "wall_time_s": result.wall_time_s,
    }


def write_results_csv(
    results: list[DynamicCaseResult],
    csv_path: Path,
) -> None:
    """将全部逐目标结果写入 CSV 文件。"""

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = [
        result_to_csv_row(result)
        for result in results
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    home_scene = PandaScene()
    home_position = get_ee_position(home_scene)

    targets = generate_target_positions(
        home_position,
        target_count=TARGET_COUNT,
        seed=RANDOM_SEED,
    )

    results: list[DynamicCaseResult] = []

    print("Day 14: 50-target reach evaluation")
    print(f"random seed: {RANDOM_SEED}")
    print(f"target count: {TARGET_COUNT}")
    print()

    for case_id, target_position in enumerate(targets):
        result = evaluate_target(
            case_id,
            target_position,
        )

        results.append(result)

        print(
            f"case {result.case_id:02d}/{TARGET_COUNT - 1:02d}"
            f" | success={result.success}"
            f" | iterations={result.ik_iterations:2d}"
            f" | IK={result.ik_error_m * 1000:8.4f} mm"
            f" | tracking="
            f"{result.tracking_error_m * 1000:8.4f} mm"
            f" | wall={result.wall_time_s:.4f} s"
            f" | reason={result.failure_reason}"
        )

    write_results_csv(results, CSV_PATH)

    # ---------- 汇总指标 ----------

    success_results = [
        result
        for result in results
        if result.success
    ]

    success_count = len(success_results)
    success_rate = success_count / len(results)

    ik_success_count = sum(
        result.ik_success
        for result in results
    )

    non_finite_count = sum(
        not result.all_finite
        for result in results
    )

    joint_limit_violation_count = sum(
        result.joint_limit_violation
        for result in results
    )

    failure_reason_counts = Counter(
        result.failure_reason
        for result in results
        if not result.success
    )

    if success_results:
        successful_errors = np.array(
            [
                result.tracking_error_m
                for result in success_results
            ],
            dtype=float,
        )

        mean_success_error_m = float(
            np.mean(successful_errors)
        )

        maximum_success_error_m = float(
            np.max(successful_errors)
        )
    else:
        mean_success_error_m = float("inf")
        maximum_success_error_m = float("inf")

    mean_ik_iterations = float(
        np.mean(
            [
                result.ik_iterations
                for result in results
            ]
        )
    )

    mean_wall_time_s = float(
        np.mean(
            [
                result.wall_time_s
                for result in results
            ]
        )
    )

    total_wall_time_s = float(
        np.sum(
            [
                result.wall_time_s
                for result in results
            ]
        )
    )

    print()
    print("Evaluation summary")
    print(
        f"IK success:                 "
        f"{ik_success_count}/{TARGET_COUNT}"
    )
    print(
        f"overall success:            "
        f"{success_count}/{TARGET_COUNT}"
    )
    print(
        f"success rate:               "
        f"{success_rate * 100:.2f}%"
    )
    print(
        f"mean successful error:      "
        f"{mean_success_error_m * 1000:.6f} mm"
    )
    print(
        f"maximum successful error:   "
        f"{maximum_success_error_m * 1000:.6f} mm"
    )
    print(
        f"mean IK iterations:         "
        f"{mean_ik_iterations:.3f}"
    )
    print(
        f"mean wall time per target:  "
        f"{mean_wall_time_s:.6f} s"
    )
    print(
        f"total wall time:            "
        f"{total_wall_time_s:.6f} s"
    )
    print(
        f"non-finite cases:           "
        f"{non_finite_count}"
    )
    print(
        f"joint-limit violations:     "
        f"{joint_limit_violation_count}"
    )
    print(
        f"failure reasons:            "
        f"{dict(failure_reason_counts)}"
    )
    print(f"CSV: {CSV_PATH}")

    # ---------- Day14 自动验收 ----------

    assert success_rate >= MINIMUM_SUCCESS_RATE
    assert (
        mean_success_error_m
        <= MAXIMUM_MEAN_SUCCESS_ERROR_M
    )
    assert non_finite_count == 0
    assert joint_limit_violation_count == 0

    print()
    print("Day 14 reach evaluation: PASSED")


if __name__ == "__main__":
    main()
