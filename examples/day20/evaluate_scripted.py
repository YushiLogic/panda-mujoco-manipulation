"""Day 20：在多个可复现随机场景中评估脚本抓取策略。

这个脚本使用轻量任务接口完成完整episode：

1. ``reset(seed)`` 随机化并落稳方块；
2. ``step(RUN_SCRIPTED_PICK)`` 执行Day19脚本抓取；
3. 记录每个seed的初始条件、结果和失败阶段；
4. 将逐回合结果写入CSV并打印汇总。

默认只运行5个case，适合Day20快速检查。后续可以用 ``--cases``
增加样本数，进行更完整的策略评估。
"""

import argparse
import csv
from collections import Counter
from pathlib import Path
import time

import numpy as np

from panda_mujoco.panda_pick_task import (
    PandaPickTask,
    PickAction,
)


DEFAULT_CASE_COUNT = 5
DEFAULT_START_SEED = 0
DEFAULT_CSV_PATH = Path(
    "results/day20/scripted_pick_evaluation.csv"
)


def parse_arguments() -> argparse.Namespace:
    """读取命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the scripted Panda pick policy "
            "over deterministic random cube poses."
        )
    )
    parser.add_argument(
        "--cases",
        type=int,
        default=DEFAULT_CASE_COUNT,
        help="number of episodes to evaluate (default: 5)",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=DEFAULT_START_SEED,
        help="seed of the first episode (default: 0)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=(
            "output CSV path "
            "(default: results/day20/scripted_pick_evaluation.csv)"
        ),
    )

    arguments = parser.parse_args()

    if arguments.cases <= 0:
        parser.error("--cases must be a positive integer")

    return arguments


def evaluate_scripted_policy(
    *,
    case_count: int,
    start_seed: int,
) -> list[dict[str, object]]:
    """运行一批抓取episode并返回逐回合记录。"""

    if case_count <= 0:
        raise ValueError("case_count must be positive")

    task = PandaPickTask()
    rows: list[dict[str, object]] = []

    print("Day 20 scripted-pick evaluation\n")

    for case_index in range(case_count):
        seed = start_seed + case_index

        # reset_info描述动作开始前的随机初始条件。必须在step前保存，
        # 因为成功抓取后方块位置已经发生改变。
        _, reset_info = task.reset(seed=seed)
        initial_cube_position = np.asarray(
            reset_info["cube_position"],
            dtype=float,
        )
        initial_cube_yaw = float(
            reset_info["cube_yaw"]
        )

        wall_start = time.perf_counter()
        _, reward, terminated, truncated, info = task.step(
            PickAction.RUN_SCRIPTED_PICK
        )
        wall_duration = time.perf_counter() - wall_start

        row: dict[str, object] = {
            "case": case_index,
            "seed": seed,
            "cube_x_m": float(initial_cube_position[0]),
            "cube_y_m": float(initial_cube_position[1]),
            "cube_z_m": float(initial_cube_position[2]),
            "cube_yaw_deg": float(
                np.rad2deg(initial_cube_yaw)
            ),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "success": bool(info["success"]),
            "final_state": str(info["final_state"]),
            "failed_stage": info["failed_stage"],
            "reason": str(info["reason"]),
            "failure_category": str(
                info["failure_category"]
            ),
            "lift_height_m": float(info["lift_height"]),
            "hold_duration_s": float(info["hold_duration"]),
            "gripper_width_m": float(info["gripper_width"]),
            "wall_duration_s": wall_duration,
        }
        rows.append(row)

        failed_stage = (
            "none"
            if row["failed_stage"] is None
            else str(row["failed_stage"])
        )
        print(
            f"case {case_index:02d} | seed={seed:3d} | "
            f"cube=({row['cube_x_m']:+.4f}, "
            f"{row['cube_y_m']:+.4f}) m | "
            f"yaw={row['cube_yaw_deg']:+7.3f} deg | "
            f"success={str(row['success']):5s} | "
            f"failed_stage={failed_stage:14s} | "
            f"reason={row['reason']}"
        )

    return rows


def write_results_csv(
    rows: list[dict[str, object]],
    csv_path: Path,
) -> None:
    """把逐回合记录写入CSV。"""

    if not rows:
        raise ValueError("rows must not be empty")

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with csv_path.open(
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


def print_summary(
    rows: list[dict[str, object]],
    csv_path: Path,
) -> None:
    """打印整批episode的统计结果。"""

    successes = sum(
        bool(row["success"])
        for row in rows
    )
    total = len(rows)
    success_rate = successes / total

    failure_categories = Counter(
        str(row["failure_category"])
        for row in rows
        if not bool(row["success"])
    )
    failed_stages = Counter(
        str(row["failed_stage"])
        for row in rows
        if row["failed_stage"] is not None
    )

    mean_wall_duration = float(
        np.mean(
            [
                float(row["wall_duration_s"])
                for row in rows
            ]
        )
    )

    print("\nEvaluation summary")
    print(f"success count:          {successes}/{total}")
    print(f"success rate:           {100.0 * success_rate:.2f}%")
    print(f"terminated episodes:    {sum(bool(row['terminated']) for row in rows)}")
    print(f"truncated episodes:     {sum(bool(row['truncated']) for row in rows)}")
    print(f"failure categories:     {dict(failure_categories)}")
    print(f"failed stages:          {dict(failed_stages)}")
    print(f"mean wall time/case:    {mean_wall_duration:.4f} s")
    print(f"CSV: {csv_path.resolve()}")


def main() -> None:
    """运行命令行评估。"""

    arguments = parse_arguments()
    rows = evaluate_scripted_policy(
        case_count=arguments.cases,
        start_seed=arguments.start_seed,
    )
    write_results_csv(rows, arguments.csv)
    print_summary(rows, arguments.csv)


if __name__ == "__main__":
    main()
