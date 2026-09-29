"""Run deterministic random-action smoke episodes for PandaContinuousEnv.

This is a robustness check, not a policy-training script. A task failure such
as timeout or cube drop is recorded as an episode outcome; it only fails the
smoke test if the simulation/API becomes invalid or the arm leaves its limits.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from panda_mujoco.continuous_env import PandaContinuousEnv


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV_PATH = REPOSITORY_ROOT / "results" / "day25" / "random_smoke.csv"
ARM_DOF = 7
JOINT_LIMIT_TOLERANCE_RAD = 1e-6


def validate_observation_and_joints(
    env: PandaContinuousEnv,
    observation: np.ndarray,
) -> None:
    """Raise a clear error if a state is numerically or physically invalid."""

    if not np.all(np.isfinite(observation)):
        raise AssertionError("observation contains NaN or infinity")

    if not env.observation_space.contains(observation):
        raise AssertionError("observation is outside observation_space")

    # In this Panda model, the first seven qpos entries are the arm joints.
    arm_qpos = env.task.scene.data.qpos[:ARM_DOF]
    arm_ranges = env.task.scene.model.jnt_range[:ARM_DOF]
    lower = arm_ranges[:, 0]
    upper = arm_ranges[:, 1]

    if not np.all(np.isfinite(arm_qpos)):
        raise AssertionError("arm joint position contains NaN or infinity")

    below_limit = arm_qpos < lower - JOINT_LIMIT_TOLERANCE_RAD
    above_limit = arm_qpos > upper + JOINT_LIMIT_TOLERANCE_RAD
    if np.any(below_limit | above_limit):
        raise AssertionError("arm joint position is outside joint limits")


def run_episode(seed: int, max_steps: int) -> dict[str, object]:
    """Run one episode and return a CSV-friendly summary row."""

    env = PandaContinuousEnv(max_episode_steps=max_steps)
    row: dict[str, object] = {
        "seed": seed,
        "steps": 0,
        "episode_return": 0.0,
        "terminated": False,
        "truncated": False,
        "success": False,
        "termination_reason": "not_started",
        "ik_failures": 0,
        "smoke_passed": False,
        "error": "",
    }

    try:
        # Seed both the environment and its action sampler. This makes a
        # repeated seed reproduce the same reset and random action sequence.
        env.action_space.seed(seed)
        observation, _ = env.reset(seed=seed)
        validate_observation_and_joints(env, observation)

        episode_return = 0.0
        ik_failures = 0
        terminated = False
        truncated = False
        info: dict[str, object] = {}

        for step_index in range(1, max_steps + 1):
            action = env.action_space.sample()
            if not np.all(np.isfinite(action)):
                raise AssertionError("sampled action contains NaN or infinity")
            if not env.action_space.contains(action):
                raise AssertionError("sampled action is outside action_space")

            observation, reward, terminated, truncated, info = env.step(action)
            validate_observation_and_joints(env, observation)

            if not np.isfinite(reward):
                raise AssertionError("reward is NaN or infinity")

            episode_return += float(reward)
            ik_failures += int(not bool(info.get("ik_success", True)))

            if terminated or truncated:
                break

        if not (terminated or truncated):
            raise AssertionError("episode did not end at its configured step limit")

        row.update(
            {
                "steps": step_index,
                "episode_return": episode_return,
                "terminated": terminated,
                "truncated": truncated,
                "success": bool(info.get("success", False)),
                "termination_reason": str(
                    info.get("termination_reason", info.get("reason", "unknown"))
                ),
                "ik_failures": ik_failures,
                "smoke_passed": True,
            }
        )
    except Exception as exc:  # Record the failed seed, then continue the batch.
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["termination_reason"] = "smoke_check_error"
    finally:
        env.close()

    return row


def run_batch(episodes: int, max_steps: int, seed_start: int) -> list[dict[str, object]]:
    """Run a reproducible batch whose episode seeds are consecutive integers."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    return [
        run_episode(seed=seed_start + index, max_steps=max_steps)
        for index in range(episodes)
    ]


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    """Save one summary row per seed so failures can be inspected later."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "steps",
        "episode_return",
        "terminated",
        "truncated",
        "success",
        "termination_reason",
        "ik_failures",
        "smoke_passed",
        "error",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check PandaContinuousEnv with reproducible random actions."
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    args = parser.parse_args()

    rows = run_batch(
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed_start=args.seed_start,
    )
    write_csv(rows, args.csv)

    passed_count = sum(bool(row["smoke_passed"]) for row in rows)
    success_count = sum(bool(row["success"]) for row in rows)
    print(f"Episodes passed robustness checks: {passed_count}/{len(rows)}")
    print(f"Task successes under random actions: {success_count}/{len(rows)}")
    print(f"CSV: {args.csv.resolve()}")

    if passed_count != len(rows):
        print("Random smoke test: FAILED")
        raise SystemExit(1)
    print("Random smoke test: PASSED")


if __name__ == "__main__":
    main()
