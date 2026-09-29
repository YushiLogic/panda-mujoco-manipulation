"""Compare continuous random actions with the existing scripted pick expert.

The two policies use different control interfaces. This evaluator adapts each
to the same per-episode ``EpisodeResult`` schema and evaluates them on matching
cube-reset seeds; it does not replace or modify either controller.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import numpy as np

from panda_mujoco.continuous_env import PandaContinuousEnv
from panda_mujoco.evaluation import (
    EpisodeResult,
    summarize_episode_results,
)
from panda_mujoco.pick_controller import PickState, run_scripted_pick
from panda_mujoco.simulation import PandaScene


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV_PATH = (
    REPOSITORY_ROOT / "results" / "day26" / "policy_evaluation.csv"
)
ARM_DOF = 7
JOINT_LIMIT_TOLERANCE_RAD = 1e-6
RESULT_COLUMNS = [
    "policy_name",
    "seed",
    "evaluation_valid",
    "success",
    "outcome",
    "reason",
    "failed_stage",
    "reset_sim_time_s",
    "task_sim_time_s",
    "total_sim_time_s",
    "wall_time_s",
    "physics_steps",
    "episode_return",
    "error",
]


def _validate_continuous_state(
    env: PandaContinuousEnv,
    observation: np.ndarray,
    reward: float,
) -> None:
    """Reject numerical, space, or arm-limit corruption in an episode."""

    if not np.all(np.isfinite(observation)):
        raise AssertionError("observation contains NaN or infinity")
    if not env.observation_space.contains(observation):
        raise AssertionError("observation is outside observation_space")
    if not np.isfinite(reward):
        raise AssertionError("reward is NaN or infinity")

    scene = env.task.scene
    arm_qpos = scene.data.qpos[:ARM_DOF]
    arm_ranges = scene.model.jnt_range[:ARM_DOF]
    if not np.all(np.isfinite(arm_qpos)):
        raise AssertionError("arm joint position contains NaN or infinity")
    if np.any(arm_qpos < arm_ranges[:, 0] - JOINT_LIMIT_TOLERANCE_RAD):
        raise AssertionError("arm joint position is below its joint limit")
    if np.any(arm_qpos > arm_ranges[:, 1] + JOINT_LIMIT_TOLERANCE_RAD):
        raise AssertionError("arm joint position is above its joint limit")


def run_random_episode(seed: int, max_steps: int = 100) -> EpisodeResult:
    """Run random actions in the continuous environment for one reset seed."""

    started_at = time.perf_counter()
    env: PandaContinuousEnv | None = None
    reset_sim_time_s = 0.0
    task_start_time_s = 0.0
    task_sim_time_s = 0.0
    physics_steps = 0
    episode_return = 0.0
    success = False
    reason = "evaluation_error"
    error = "episode did not complete"
    evaluation_valid = False

    try:
        env = PandaContinuousEnv(max_episode_steps=max_steps)
        # The reset seed controls the cube pose. The action-space seed controls
        # sampled actions; using the same integer keeps the whole rollout
        # repeatable while the reset seed remains paired with the expert.
        env.action_space.seed(seed)
        observation, reset_info = env.reset(seed=seed)
        reset_sim_time_s = float(reset_info.get("settle_duration", 0.0))
        task_start_time_s = float(env.task.scene.data.time)
        _validate_continuous_state(env, observation, reward=0.0)

        terminated = False
        truncated = False
        info: dict[str, object] = {}

        for _ in range(max_steps):
            action = env.action_space.sample()
            if not np.all(np.isfinite(action)):
                raise AssertionError("sampled action contains NaN or infinity")
            if not env.action_space.contains(action):
                raise AssertionError("sampled action is outside action_space")

            observation, reward, terminated, truncated, info = env.step(action)
            _validate_continuous_state(env, observation, reward)
            episode_return += float(reward)
            if terminated or truncated:
                break

        scene = env.task.scene
        task_sim_time_s = max(
            0.0,
            float(scene.data.time) - task_start_time_s,
        )
        timestep = float(scene.model.opt.timestep)
        physics_steps = int(round(task_sim_time_s / timestep))

        if not (terminated or truncated):
            raise AssertionError("episode did not end at its configured step limit")

        success = bool(info.get("success", False))
        reason = (
            "success"
            if success
            else str(info.get("termination_reason", "unknown"))
        )
        if reason == "none":
            reason = str(info.get("reason", "unknown"))
        if reason in {"", "none"}:
            reason = "unknown"

        evaluation_valid = True
        error = ""
    except Exception as exc:
        if env is not None:
            scene = env.task.scene
            task_sim_time_s = max(
                0.0,
                float(scene.data.time) - task_start_time_s,
            )
            timestep = float(scene.model.opt.timestep)
            physics_steps = int(round(task_sim_time_s / timestep))
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if env is not None:
            env.close()

    return EpisodeResult(
        policy_name="random_continuous",
        seed=seed,
        evaluation_valid=evaluation_valid,
        success=success,
        reason=reason,
        failed_stage=None,
        reset_sim_time_s=reset_sim_time_s,
        task_sim_time_s=task_sim_time_s,
        wall_time_s=time.perf_counter() - started_at,
        physics_steps=physics_steps,
        episode_return=episode_return,
        error=error,
    )


def run_scripted_expert_episode(seed: int) -> EpisodeResult:
    """Run the existing state-machine expert on a fresh Panda scene."""

    started_at = time.perf_counter()
    scene: PandaScene | None = None
    reset_sim_time_s = 0.0
    task_sim_time_s = 0.0
    physics_steps = 0
    success = False
    reason = "evaluation_error"
    failed_stage: str | None = None
    error = "episode did not complete"
    evaluation_valid = False

    try:
        scene = PandaScene()
        pick_result = run_scripted_pick(scene, seed=seed)

        reset_sim_time_s = sum(
            record.simulation_duration
            for record in pick_result.records
            if record.state is PickState.RESET
        )
        task_sim_time_s = sum(
            record.simulation_duration
            for record in pick_result.records
            if record.state is not PickState.RESET
        )
        physics_steps = int(
            round(task_sim_time_s / float(scene.model.opt.timestep))
        )

        if not (
            np.all(np.isfinite(scene.data.qpos))
            and np.all(np.isfinite(scene.data.qvel))
            and np.all(np.isfinite(scene.data.ctrl))
        ):
            raise AssertionError("final MuJoCo state contains NaN or infinity")

        success = bool(pick_result.success)
        reason = "success" if success else pick_result.reason
        if reason in {"", "none"}:
            reason = "unknown"
        failed_stage = (
            None
            if pick_result.failed_stage is None
            else pick_result.failed_stage.value
        )
        evaluation_valid = True
        error = ""
    except Exception as exc:
        if scene is not None:
            task_sim_time_s = float(scene.data.time)
            timestep = float(scene.model.opt.timestep)
            physics_steps = int(
                round(task_sim_time_s / timestep)
            )
        error = f"{type(exc).__name__}: {exc}"

    return EpisodeResult(
        policy_name="scripted_expert",
        seed=seed,
        evaluation_valid=evaluation_valid,
        success=success,
        reason=reason,
        failed_stage=failed_stage,
        reset_sim_time_s=reset_sim_time_s,
        task_sim_time_s=task_sim_time_s,
        wall_time_s=time.perf_counter() - started_at,
        physics_steps=physics_steps,
        episode_return=None,
        error=error,
    )


def evaluate_policies(
    episodes: int,
    seed_start: int,
    random_max_steps: int,
) -> list[EpisodeResult]:
    """Evaluate both policies on the same consecutive reset seeds."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if random_max_steps <= 0:
        raise ValueError("random_max_steps must be positive")

    results: list[EpisodeResult] = []
    for seed in range(seed_start, seed_start + episodes):
        results.append(
            run_random_episode(seed=seed, max_steps=random_max_steps)
        )
        results.append(run_scripted_expert_episode(seed=seed))
    return results


def write_results_csv(
    results: list[EpisodeResult],
    output_path: Path,
) -> None:
    """Write one row per policy/seed pair."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(result.to_record() for result in results)


def print_summary(results: list[EpisodeResult]) -> None:
    """Print the comparable task metrics and policy-specific diagnostics."""

    print("Day 26 policy evaluation")
    for summary in summarize_episode_results(results):
        mean_success_time = summary["mean_success_task_sim_time_s"]
        mean_return = summary["mean_episode_return"]
        print(
            f"\n{summary['policy_name']} | "
            f"success={summary['successes']}/{summary['episodes']} "
            f"({summary['success_rate']:.1%}) | "
            f"valid={summary['valid_episodes']}/{summary['episodes']} "
            f"({summary['valid_rate']:.1%})"
        )
        print(f"failure reasons: {summary['failures_by_reason']}")
        if mean_success_time is not None:
            print(
                "mean successful task simulation time: "
                f"{mean_success_time:.3f} s"
            )
        print(f"mean wall time per episode: {summary['mean_wall_time_s']:.3f} s")
        if mean_return is not None:
            print(f"mean episode return: {mean_return:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare random continuous actions with the scripted pick expert."
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--random-max-steps", type=int, default=500)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    args = parser.parse_args()

    results = evaluate_policies(
        episodes=args.episodes,
        seed_start=args.seed_start,
        random_max_steps=args.random_max_steps,
    )
    print_summary(results)
    write_results_csv(results, args.csv)
    print(f"\nCSV: {args.csv.resolve()}")

    if any(not result.evaluation_valid for result in results):
        print("Evaluation validity: FAILED")
        raise SystemExit(1)
    print("Evaluation validity: PASSED")


if __name__ == "__main__":
    main()
