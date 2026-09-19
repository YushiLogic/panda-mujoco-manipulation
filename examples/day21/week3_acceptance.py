"""Day 21：第三周随机抓取批量验收。

本脚本在20个确定性seed上运行Day20任务接口，同时通过物理步回调检查
整个抓取过程中的数值有限性和Panda七个关节的限位。逐回合数据写入
``results/day21/grasp_evaluation.csv``，然后依据第三周验收标准给出结论。
"""

import argparse
import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import time

import numpy as np

from panda_mujoco.kinematics import ARM_JOINT_NAMES
from panda_mujoco.panda_pick_task import (
    PandaPickTask,
    PickAction,
)
from panda_mujoco.simulation import PandaScene


DEFAULT_CASE_COUNT = 20
DEFAULT_START_SEED = 0
DEFAULT_CSV_PATH = Path(
    "results/day21/grasp_evaluation.csv"
)

MINIMUM_SUCCESS_RATE = 0.80
MINIMUM_LIFT_HEIGHT = 0.05
MINIMUM_HOLD_DURATION = 0.50
JOINT_LIMIT_TOLERANCE = 1e-9


@dataclass
class EpisodeSafetyMonitor:
    """在每个物理步读取状态，不对场景施加任何控制。"""

    scene: PandaScene
    physics_steps_observed: int = 0
    non_finite_detected: bool = False
    joint_limit_violation: bool = False
    maximum_joint_limit_excess: float = 0.0
    _qpos_addresses: np.ndarray = field(init=False, repr=False)
    _lower_limits: np.ndarray = field(init=False, repr=False)
    _upper_limits: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """从模型名称查询七个关节的qpos地址和限位。"""

        joint_ids = np.array(
            [
                self.scene.model.joint(name).id
                for name in ARM_JOINT_NAMES
            ],
            dtype=int,
        )
        self._qpos_addresses = self.scene.model.jnt_qposadr[
            joint_ids
        ].copy()
        self._lower_limits = self.scene.model.jnt_range[
            joint_ids,
            0,
        ].copy()
        self._upper_limits = self.scene.model.jnt_range[
            joint_ids,
            1,
        ].copy()

    def __call__(self, scene: PandaScene) -> None:
        """记录当前物理步是否存在非有限数或关节越限。"""

        self.physics_steps_observed += 1

        state_is_finite = bool(
            np.all(np.isfinite(scene.data.qpos))
            and np.all(np.isfinite(scene.data.qvel))
            and np.all(np.isfinite(scene.data.ctrl))
            and np.all(
                np.isfinite(
                    scene.data.body("cube").xpos
                )
            )
        )
        if not state_is_finite:
            self.non_finite_detected = True

        arm_positions = scene.data.qpos[
            self._qpos_addresses
        ]
        lower_excess = self._lower_limits - arm_positions
        upper_excess = arm_positions - self._upper_limits
        current_excess = float(
            max(
                0.0,
                np.max(lower_excess),
                np.max(upper_excess),
            )
        )
        self.maximum_joint_limit_excess = max(
            self.maximum_joint_limit_excess,
            current_excess,
        )

        if current_excess > JOINT_LIMIT_TOLERANCE:
            self.joint_limit_violation = True


def parse_arguments() -> argparse.Namespace:
    """读取命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "Run the Week 3 deterministic random-grasp acceptance."
        )
    )
    parser.add_argument(
        "--cases",
        type=int,
        default=DEFAULT_CASE_COUNT,
        help="number of deterministic episodes (default: 20)",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=DEFAULT_START_SEED,
        help="first episode seed (default: 0)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=(
            "output CSV path "
            "(default: results/day21/grasp_evaluation.csv)"
        ),
    )
    arguments = parser.parse_args()

    if arguments.cases <= 0:
        parser.error("--cases must be positive")

    return arguments


def _empty_row(
    case_index: int,
    seed: int,
) -> dict[str, object]:
    """建立固定字段的默认失败记录，保证异常时仍可导出CSV。"""

    return {
        "case": case_index,
        "seed": seed,
        "cube_x_m": np.nan,
        "cube_y_m": np.nan,
        "cube_z_m": np.nan,
        "cube_yaw_deg": np.nan,
        "reward": 0.0,
        "terminated": False,
        "truncated": False,
        "success": False,
        "final_state": "EXCEPTION",
        "failed_stage": "EXCEPTION",
        "reason": "not_run",
        "failure_category": "runtime_exception",
        "lift_height_m": 0.0,
        "hold_duration_s": 0.0,
        "gripper_width_m": np.nan,
        "initial_observation_finite": False,
        "final_observation_finite": False,
        "non_finite_detected": False,
        "joint_limit_violation": False,
        "maximum_joint_limit_excess_rad": 0.0,
        "physics_steps_observed": 0,
        "simulation_duration_s": 0.0,
        "wall_duration_s": 0.0,
        "stage_trace": "",
    }


def evaluate_week3(
    *,
    case_count: int,
    start_seed: int,
) -> list[dict[str, object]]:
    """执行一批episode，并返回可直接写入CSV的逐回合记录。"""

    if case_count <= 0:
        raise ValueError("case_count must be positive")

    task = PandaPickTask()
    rows: list[dict[str, object]] = []

    print("Week 3 random-grasp acceptance\n")

    for case_index in range(case_count):
        seed = start_seed + case_index
        row = _empty_row(case_index, seed)
        wall_start = time.perf_counter()

        try:
            initial_observation, reset_info = task.reset(
                seed=seed
            )
            initial_cube_position = np.asarray(
                reset_info["cube_position"],
                dtype=float,
            )
            initial_cube_yaw = float(
                reset_info["cube_yaw"]
            )
            initial_observation_finite = bool(
                np.all(np.isfinite(initial_observation))
            )

            row.update(
                {
                    "cube_x_m": float(
                        initial_cube_position[0]
                    ),
                    "cube_y_m": float(
                        initial_cube_position[1]
                    ),
                    "cube_z_m": float(
                        initial_cube_position[2]
                    ),
                    "cube_yaw_deg": float(
                        np.rad2deg(initial_cube_yaw)
                    ),
                    "initial_observation_finite": (
                        initial_observation_finite
                    ),
                }
            )

            monitor = EpisodeSafetyMonitor(task.scene)
            simulation_start = float(task.scene.data.time)

            (
                final_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = task.step(
                PickAction.RUN_SCRIPTED_PICK,
                step_callback=monitor,
            )

            simulation_duration = (
                float(task.scene.data.time)
                - simulation_start
            )
            final_observation_finite = bool(
                np.all(np.isfinite(final_observation))
            )

            row.update(
                {
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "success": bool(info["success"]),
                    "final_state": str(info["final_state"]),
                    "failed_stage": (
                        ""
                        if info["failed_stage"] is None
                        else str(info["failed_stage"])
                    ),
                    "reason": str(info["reason"]),
                    "failure_category": str(
                        info["failure_category"]
                    ),
                    "lift_height_m": float(
                        info["lift_height"]
                    ),
                    "hold_duration_s": float(
                        info["hold_duration"]
                    ),
                    "gripper_width_m": float(
                        info["gripper_width"]
                    ),
                    "final_observation_finite": (
                        final_observation_finite
                    ),
                    "non_finite_detected": (
                        monitor.non_finite_detected
                    ),
                    "joint_limit_violation": (
                        monitor.joint_limit_violation
                    ),
                    "maximum_joint_limit_excess_rad": (
                        monitor.maximum_joint_limit_excess
                    ),
                    "physics_steps_observed": (
                        monitor.physics_steps_observed
                    ),
                    "simulation_duration_s": (
                        simulation_duration
                    ),
                    "stage_trace": ">".join(
                        str(state)
                        for state in info["stage_trace"]
                    ),
                }
            )

        except Exception as error:  # noqa: BLE001 - batch必须保留失败记录
            row["reason"] = (
                f"{type(error).__name__}: {error}"
            )

        row["wall_duration_s"] = (
            time.perf_counter() - wall_start
        )
        rows.append(row)

        failed_stage = (
            "none"
            if row["failed_stage"] == ""
            else str(row["failed_stage"])
        )
        print(
            f"case {case_index:02d} | seed={seed:02d} | "
            f"cube=({float(row['cube_x_m']):+.4f}, "
            f"{float(row['cube_y_m']):+.4f}) m | "
            f"yaw={float(row['cube_yaw_deg']):+7.3f} deg | "
            f"success={str(row['success']):5s} | "
            f"lift={100.0 * float(row['lift_height_m']):6.2f} cm | "
            f"failed_stage={failed_stage:14s} | "
            f"reason={row['reason']}"
        )

    return rows


def write_csv(
    rows: list[dict[str, object]],
    path: Path,
) -> None:
    """写出逐回合CSV。"""

    if not rows:
        raise ValueError("rows must not be empty")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    """计算第三周验收所需汇总指标。"""

    successes = [
        row for row in rows if bool(row["success"])
    ]
    failures = [
        row for row in rows if not bool(row["success"])
    ]

    failure_categories = Counter(
        str(row["failure_category"])
        for row in failures
    )
    failed_stages = Counter(
        str(row["failed_stage"])
        for row in failures
    )

    incomplete_failure_diagnostics = sum(
        row["failed_stage"] == ""
        or str(row["reason"]) in {"", "none"}
        for row in failures
    )

    return {
        "case_count": len(rows),
        "success_count": len(successes),
        "success_rate": len(successes) / len(rows),
        "minimum_successful_lift_height": (
            min(
                float(row["lift_height_m"])
                for row in successes
            )
            if successes
            else 0.0
        ),
        "minimum_successful_hold_duration": (
            min(
                float(row["hold_duration_s"])
                for row in successes
            )
            if successes
            else 0.0
        ),
        "non_finite_case_count": sum(
            not bool(row["initial_observation_finite"])
            or not bool(row["final_observation_finite"])
            or bool(row["non_finite_detected"])
            for row in rows
        ),
        "joint_limit_violation_count": sum(
            bool(row["joint_limit_violation"])
            for row in rows
        ),
        "terminated_count": sum(
            bool(row["terminated"])
            for row in rows
        ),
        "truncated_count": sum(
            bool(row["truncated"])
            for row in rows
        ),
        "incomplete_failure_diagnostics": (
            incomplete_failure_diagnostics
        ),
        "failure_categories": dict(failure_categories),
        "failed_stages": dict(failed_stages),
        "mean_wall_duration": float(
            np.mean(
                [
                    float(row["wall_duration_s"])
                    for row in rows
                ]
            )
        ),
    }


def print_summary(
    summary: dict[str, object],
    csv_path: Path,
) -> None:
    """输出人类可读的验收汇总。"""

    print("\nAcceptance summary")
    print(
        "success count:              "
        f"{summary['success_count']}/{summary['case_count']}"
    )
    print(
        "success rate:               "
        f"{100.0 * float(summary['success_rate']):.2f}%"
    )
    print(
        "minimum successful lift:    "
        f"{100.0 * float(summary['minimum_successful_lift_height']):.3f} cm"
    )
    print(
        "minimum successful hold:    "
        f"{float(summary['minimum_successful_hold_duration']):.3f} s"
    )
    print(
        "non-finite cases:           "
        f"{summary['non_finite_case_count']}"
    )
    print(
        "joint-limit violations:     "
        f"{summary['joint_limit_violation_count']}"
    )
    print(
        "terminated episodes:        "
        f"{summary['terminated_count']}"
    )
    print(
        "truncated episodes:         "
        f"{summary['truncated_count']}"
    )
    print(
        "failure categories:         "
        f"{summary['failure_categories']}"
    )
    print(
        "failed stages:              "
        f"{summary['failed_stages']}"
    )
    print(
        "incomplete failure records: "
        f"{summary['incomplete_failure_diagnostics']}"
    )
    print(
        "mean wall time/case:        "
        f"{float(summary['mean_wall_duration']):.4f} s"
    )
    print(f"CSV: {csv_path.resolve()}")


def assert_acceptance(
    summary: dict[str, object],
) -> None:
    """依据第三周路线表检查全部硬性验收条件。"""

    failures: list[str] = []

    if summary["case_count"] != DEFAULT_CASE_COUNT:
        failures.append(
            f"expected {DEFAULT_CASE_COUNT} cases"
        )
    if float(summary["success_rate"]) < MINIMUM_SUCCESS_RATE:
        failures.append("success rate is below 80%")
    if (
        float(summary["minimum_successful_lift_height"])
        < MINIMUM_LIFT_HEIGHT
    ):
        failures.append("successful lift is below 5 cm")
    if (
        float(summary["minimum_successful_hold_duration"])
        < MINIMUM_HOLD_DURATION
    ):
        failures.append("successful hold is below 0.5 s")
    if int(summary["non_finite_case_count"]) != 0:
        failures.append("non-finite state was detected")
    if int(summary["joint_limit_violation_count"]) != 0:
        failures.append("joint-limit violation was detected")
    if int(summary["incomplete_failure_diagnostics"]) != 0:
        failures.append("a failed case lacks stage or reason")
    if int(summary["terminated_count"]) != int(
        summary["case_count"]
    ):
        failures.append("not every episode terminated")
    if int(summary["truncated_count"]) != 0:
        failures.append("an evaluation episode was truncated")

    if failures:
        raise RuntimeError(
            "Week 3 acceptance failed: "
            + "; ".join(failures)
        )


def main() -> None:
    """运行批量验收并保存结果。"""

    arguments = parse_arguments()
    rows = evaluate_week3(
        case_count=arguments.cases,
        start_seed=arguments.start_seed,
    )
    write_csv(rows, arguments.csv)
    summary = summarize(rows)
    print_summary(summary, arguments.csv)

    # 路线表的正式验收固定为20个case。自定义--cases仍可用于调试，
    # 但只有默认数量才执行最终PASS/FAIL门槛。
    if arguments.cases == DEFAULT_CASE_COUNT:
        assert_acceptance(summary)
        print("\nWeek 3 random-grasp acceptance: PASSED")
    else:
        print(
            "\nCustom case count completed; "
            "formal 20-case acceptance was not applied."
        )


if __name__ == "__main__":
    main()
