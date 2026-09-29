"""Evaluate random and SAC reach policies on identical held-out cube poses.

Each policy is run in a fresh environment with the same reset seed. The
random policy has its own reproducible action-space seed. This is a paired
reach-only comparison; it is not comparable to the full scripted pick task.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from panda_mujoco.pick_controller import get_cube_yaw
from panda_mujoco.reach_env import PandaReachEnv


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = (
    REPOSITORY_ROOT / "outputs" / "day26" / "sac_reach" / "panda_sac_reach"
)
DEFAULT_CSV_PATH = (
    REPOSITORY_ROOT / "results" / "day27" / "reach_paired_evaluation.csv"
)
ACTION_SEED_OFFSET = 1_000_000
CSV_FIELDS = [
    "policy",
    "seed",
    "cube_x_m",
    "cube_y_m",
    "cube_z_m",
    "cube_yaw_deg",
    "initial_position_error_mm",
    "initial_orientation_error_deg",
    "success",
    "reason",
    "steps",
    "final_position_error_mm",
    "final_orientation_error_deg",
    "episode_return",
]


def run_episode(
    *,
    policy_name: str,
    seed: int,
    max_episode_steps: int,
    policy: Any | None,
) -> dict[str, object]:
    """Run one policy from a reproducible cube pose and collect metrics."""

    env = PandaReachEnv(max_episode_steps=max_episode_steps)
    try:
        env.action_space.seed(seed + ACTION_SEED_OFFSET)
        observation, reset_info = env.reset(seed=seed)
        scene = env.base_env.task.scene
        cube_position = scene.data.body("cube").xpos.copy()
        cube_yaw = get_cube_yaw(scene)
        initial_position_error_mm = (
            1000.0 * float(reset_info["reach_position_error_m"])
        )
        initial_orientation_error_deg = float(
            np.rad2deg(reset_info["reach_orientation_error_rad"])
        )

        episode_return = 0.0
        terminated = False
        truncated = False
        info: dict[str, object] = {}
        steps = 0
        while not (terminated or truncated):
            if policy is None:
                action = env.action_space.sample()
            else:
                action, _state = policy.predict(observation, deterministic=True)

            observation, reward, terminated, truncated, info = env.step(action)
            if not np.isfinite(reward) or not np.all(np.isfinite(observation)):
                raise RuntimeError("policy produced a non-finite state or reward")
            episode_return += float(reward)
            steps += 1

        return {
            "policy": policy_name,
            "seed": seed,
            "cube_x_m": float(cube_position[0]),
            "cube_y_m": float(cube_position[1]),
            "cube_z_m": float(cube_position[2]),
            "cube_yaw_deg": float(np.rad2deg(cube_yaw)),
            "initial_position_error_mm": initial_position_error_mm,
            "initial_orientation_error_deg": initial_orientation_error_deg,
            "success": bool(info.get("reach_success", False)),
            "reason": str(info.get("reason", "unknown")),
            "steps": steps,
            "final_position_error_mm": 1000.0
            * float(info["task_reach_error_m"]),
            "final_orientation_error_deg": float(
                np.rad2deg(info["task_orientation_error_rad"])
            ),
            "episode_return": episode_return,
        }
    finally:
        env.close()


def evaluate_paired(
    *,
    model: Any,
    episodes: int,
    seed_start: int,
    max_episode_steps: int,
) -> list[dict[str, object]]:
    """Evaluate random then SAC on each identical held-out reset seed."""

    rows: list[dict[str, object]] = []
    for seed in range(seed_start, seed_start + episodes):
        random_row = run_episode(
            policy_name="random",
            seed=seed,
            max_episode_steps=max_episode_steps,
            policy=None,
        )
        sac_row = run_episode(
            policy_name="sac",
            seed=seed,
            max_episode_steps=max_episode_steps,
            policy=model,
        )

        # These checks guard against a silent comparison of different tasks.
        shared_pose_fields = (
            "cube_x_m",
            "cube_y_m",
            "cube_z_m",
            "cube_yaw_deg",
            "initial_position_error_mm",
            "initial_orientation_error_deg",
        )
        for field in shared_pose_fields:
            if not np.isclose(
                float(random_row[field]),
                float(sac_row[field]),
                rtol=0.0,
                atol=1e-10,
            ):
                raise RuntimeError(
                    f"paired initial condition mismatch for seed {seed}: {field}"
                )

        rows.extend((random_row, sac_row))
        print(
            f"seed={seed} | cube=({random_row['cube_x_m']:+.4f}, "
            f"{random_row['cube_y_m']:+.4f}) m "
            f"yaw={random_row['cube_yaw_deg']:+.2f} deg | "
            f"initial={random_row['initial_position_error_mm']:.1f} mm / "
            f"{random_row['initial_orientation_error_deg']:.1f} deg"
        )
        for row in (random_row, sac_row):
            print(
                f"  {row['policy']:>6} | success={row['success']} "
                f"| reason={row['reason']} | steps={row['steps']:>3} "
                f"| final={row['final_position_error_mm']:.2f} mm / "
                f"{row['final_orientation_error_deg']:.2f} deg "
                f"| return={row['episode_return']:.3f}"
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write per-policy results while keeping each paired seed adjacent."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, object]]) -> None:
    """Summarize task success and final errors separately by policy."""

    print("\nPaired Reach evaluation summary")
    for policy_name in ("random", "sac"):
        selected = [row for row in rows if row["policy"] == policy_name]
        successes = sum(bool(row["success"]) for row in selected)
        mean_position_error = float(
            np.mean([float(row["final_position_error_mm"]) for row in selected])
        )
        mean_orientation_error = float(
            np.mean(
                [float(row["final_orientation_error_deg"]) for row in selected]
            )
        )
        mean_return = float(
            np.mean([float(row["episode_return"]) for row in selected])
        )
        print(
            f"{policy_name:>6} | success={successes}/{len(selected)} "
            f"({successes / len(selected):.1%}) "
            f"| mean final error={mean_position_error:.2f} mm / "
            f"{mean_orientation_error:.2f} deg "
            f"| mean return={mean_return:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=200_000)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    args = parser.parse_args()

    if args.episodes <= 0 or args.max_episode_steps <= 0:
        parser.error("episodes and max-episode-steps must be positive")

    try:
        import torch
        from stable_baselines3 import SAC
    except ImportError as error:
        raise SystemExit(
            "SAC dependencies are missing. Install with: "
            'python -m pip install -e ".[rl]"'
        ) from error

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is False")
    if not args.model_path.with_suffix(".zip").exists():
        raise SystemExit(f"SAC model not found: {args.model_path.with_suffix('.zip')}")

    model = SAC.load(str(args.model_path), device=args.device)
    print("Day 27: paired, held-out Reach evaluation")
    print(f"Model: {args.model_path.with_suffix('.zip')}")
    print(f"Inference device: {model.device}")
    print(
        f"Seeds: {args.seed_start}..{args.seed_start + args.episodes - 1} "
        f"| max steps per episode: {args.max_episode_steps}"
    )
    print("Task scope: Reach only; gripper stays open; no cube grasp/lift.\n")

    rows = evaluate_paired(
        model=model,
        episodes=args.episodes,
        seed_start=args.seed_start,
        max_episode_steps=args.max_episode_steps,
    )
    print_summary(rows)
    write_csv(args.csv, rows)
    print(f"\nCSV: {args.csv.resolve()}")
    print(
        f"Interpretation: this evaluation covers {args.episodes} seeds only. "
        "Use it to compare this checkpoint with baselines; by itself, it "
        "does not establish convergence or broad generalization."
    )


if __name__ == "__main__":
    main()
