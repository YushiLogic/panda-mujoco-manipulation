"""Train, save, reload, and briefly evaluate SAC on the reach-only task.

The default 1,000-step run is a pipeline smoke test, not a convergence claim.
Increase ``--timesteps`` for a meaningful learning experiment.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from panda_mujoco.reach_env import PandaReachEnv


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = (
    REPOSITORY_ROOT / "outputs" / "day26" / "sac_reach" / "panda_sac_reach"
)
DEFAULT_EVALUATION_CSV = (
    REPOSITORY_ROOT / "outputs" / "day26" / "sac_reach" / "evaluation.csv"
)


def _run_episode(
    env: PandaReachEnv,
    *,
    seed: int,
    policy: Any | None,
) -> dict[str, object]:
    """Run one fixed-seed random or deterministic SAC reach episode."""

    observation, _reset_info = env.reset(seed=seed)
    env.action_space.seed(seed)
    total_reward = 0.0
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
        total_reward += float(reward)
        steps += 1

    return {
        "policy": "random" if policy is None else "sac",
        "seed": seed,
        "success": bool(info.get("reach_success", False)),
        "reason": str(info.get("reason", "unknown")),
        "steps": steps,
        "position_error_mm": 1000.0 * float(info["task_reach_error_m"]),
        "orientation_error_deg": float(
            np.rad2deg(info["task_orientation_error_rad"])
        ),
        "episode_return": total_reward,
    }


def _write_results(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "policy",
        "seed",
        "success",
        "reason",
        "steps",
        "position_error_mm",
        "orientation_error_deg",
        "episode_return",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000,
        help="training environment steps; default is only a smoke test",
    )
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="model save/load path without the .zip suffix",
    )
    parser.add_argument(
        "--evaluation-csv",
        type=Path,
        default=DEFAULT_EVALUATION_CSV,
    )
    args = parser.parse_args()

    if args.timesteps <= 0 or args.eval_episodes <= 0:
        parser.error("timesteps and eval-episodes must be positive")

    try:
        import torch
        from stable_baselines3 import SAC
    except ImportError as error:
        raise SystemExit(
            "SAC dependencies are missing. Install the project with: "
            'python -m pip install -e ".[rl]"'
        ) from error

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested, but torch.cuda.is_available() is False"
        )

    print(f"PyTorch: {torch.__version__}")
    print(f"Training device: {args.device}")
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    env = PandaReachEnv(max_episode_steps=args.max_episode_steps)
    try:
        learning_starts = min(500, max(10, args.timesteps // 2))
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=max(5_000, args.timesteps * 2),
            learning_starts=learning_starts,
            batch_size=128,
            policy_kwargs={"net_arch": [128, 128]},
            device=args.device,
            seed=args.seed,
            verbose=1,
        )
        print(f"Model device: {model.device}")
        print(f"Training steps: {args.timesteps}")
        model.learn(total_timesteps=args.timesteps, progress_bar=False)

        args.model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(args.model_path))
        print(f"Saved model: {args.model_path.with_suffix('.zip')}")

        # Reloading verifies that the saved artifact is usable for inference.
        del model
        loaded_model = SAC.load(
            str(args.model_path),
            env=env,
            device=args.device,
        )
        print(f"Reloaded model device: {loaded_model.device}")

        rows: list[dict[str, object]] = []
        first_eval_seed = args.seed + 100_000
        for eval_seed in range(first_eval_seed, first_eval_seed + args.eval_episodes):
            rows.append(_run_episode(env, seed=eval_seed, policy=None))
            rows.append(_run_episode(env, seed=eval_seed, policy=loaded_model))

        _write_results(args.evaluation_csv, rows)
        for row in rows:
            print(
                f"{row['policy']:>6} | seed={row['seed']} "
                f"| success={row['success']} | reason={row['reason']} "
                f"| steps={row['steps']:>3} "
                f"| pos={row['position_error_mm']:.2f} mm "
                f"| rot={row['orientation_error_deg']:.2f} deg "
                f"| return={row['episode_return']:.3f}"
            )
        print(f"Evaluation CSV: {args.evaluation_csv}")
        print(
            f"Note: this evaluation covers {args.eval_episodes} seeds only; "
            "it does not by itself establish convergence or broad "
            "generalization."
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
