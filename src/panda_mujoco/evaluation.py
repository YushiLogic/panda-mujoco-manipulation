"""Shared result schema for policy-evaluation episodes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from statistics import fmean
from collections import defaultdict
from collections.abc import Iterable


@dataclass(frozen=True)
class EpisodeResult:
    """A normalized, CSV-friendly summary of one policy episode.

    ``physics_steps`` and ``task_sim_time_s`` describe policy execution only;
    ``reset_sim_time_s`` records scene preparation separately. ``episode_return``
    is optional because not every controller (for example, the scripted pick
    state machine) runs through a reward-producing Gymnasium environment.
    """

    policy_name: str
    seed: int
    evaluation_valid: bool
    success: bool
    reason: str
    failed_stage: str | None
    reset_sim_time_s: float
    task_sim_time_s: float
    wall_time_s: float
    physics_steps: int
    episode_return: float | None = None
    error: str = ""

    def __post_init__(self) -> None:
        """Reject malformed records before they enter aggregate statistics."""

        if not isinstance(self.policy_name, str) or not self.policy_name.strip():
            raise ValueError("policy_name must be a non-empty string")

        if (
            not isinstance(self.seed, Integral)
            or isinstance(self.seed, bool)
        ):
            raise ValueError("seed must be an integer")

        if not isinstance(self.evaluation_valid, bool):
            raise ValueError("evaluation_valid must be a bool")
        if not isinstance(self.success, bool):
            raise ValueError("success must be a bool")

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")
        if self.failed_stage is not None and (
            not isinstance(self.failed_stage, str)
            or not self.failed_stage.strip()
        ):
            raise ValueError("failed_stage must be a non-empty string or None")
        if not isinstance(self.error, str):
            raise ValueError("error must be a string")

        for field_name in (
            "reset_sim_time_s",
            "task_sim_time_s",
            "wall_time_s",
        ):
            value = getattr(self, field_name)
            if (
                not isinstance(value, Real)
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{field_name} must be finite")
            if value < 0.0:
                raise ValueError(f"{field_name} must be non-negative")

        if (
            not isinstance(self.physics_steps, Integral)
            or isinstance(self.physics_steps, bool)
            or self.physics_steps < 0
        ):
            raise ValueError("physics_steps must be a non-negative integer")

        if self.episode_return is not None:
            if (
                not isinstance(self.episode_return, Real)
                or isinstance(self.episode_return, bool)
                or not math.isfinite(float(self.episode_return))
            ):
                raise ValueError("episode_return must be finite or None")

        if self.success and not self.evaluation_valid:
            raise ValueError("an invalid evaluation cannot be successful")
        if self.evaluation_valid and self.error:
            raise ValueError("a valid evaluation must not contain an error")
        if not self.evaluation_valid and not self.error.strip():
            raise ValueError("an invalid evaluation must include an error")
        if self.success and self.reason != "success":
            raise ValueError("a successful episode must use reason='success'")
        if not self.success and self.reason == "success":
            raise ValueError("a failed episode cannot use reason='success'")

    @property
    def outcome(self) -> str:
        """Return a common high-level outcome label."""

        if not self.evaluation_valid:
            return "invalid"
        if self.success:
            return "success"
        if self.reason in {"time_limit", "timeout"}:
            return "timeout"
        return "failure"

    def to_record(self) -> dict[str, object]:
        """Return one flat record suitable for a CSV row or JSON output."""

        return {
            "policy_name": self.policy_name,
            "seed": int(self.seed),
            "evaluation_valid": self.evaluation_valid,
            "success": self.success,
            "outcome": self.outcome,
            "reason": self.reason,
            "failed_stage": self.failed_stage,
            "reset_sim_time_s": float(self.reset_sim_time_s),
            "task_sim_time_s": float(self.task_sim_time_s),
            "total_sim_time_s": float(
                self.reset_sim_time_s + self.task_sim_time_s
            ),
            "wall_time_s": float(self.wall_time_s),
            "physics_steps": int(self.physics_steps),
            "episode_return": (
                None
                if self.episode_return is None
                else float(self.episode_return)
            ),
            "error": self.error,
        }


def summarize_episode_results(
    results: Iterable[EpisodeResult],
) -> list[dict[str, object]]:
    """Aggregate per-episode outcomes without hiding invalid evaluations.

    ``success_rate`` uses every attempted episode as its denominator, so an
    invalid evaluation cannot disappear from the reported task success rate.
    ``valid_rate`` is reported separately to diagnose evaluation reliability.
    """

    grouped: dict[str, list[EpisodeResult]] = defaultdict(list)
    for result in results:
        grouped[result.policy_name].append(result)

    if not grouped:
        raise ValueError("at least one episode result is required")

    summaries: list[dict[str, object]] = []
    for policy_name in sorted(grouped):
        policy_results = grouped[policy_name]
        episode_count = len(policy_results)
        valid_results = [
            result for result in policy_results if result.evaluation_valid
        ]
        successful_results = [
            result for result in valid_results if result.success
        ]
        invalid_count = episode_count - len(valid_results)
        failures_by_reason: dict[str, int] = {}
        for result in valid_results:
            if not result.success:
                failures_by_reason[result.reason] = (
                    failures_by_reason.get(result.reason, 0) + 1
                )

        returns = [
            float(result.episode_return)
            for result in valid_results
            if result.episode_return is not None
        ]
        summaries.append(
            {
                "policy_name": policy_name,
                "episodes": episode_count,
                "valid_episodes": len(valid_results),
                "invalid_episodes": invalid_count,
                "valid_rate": len(valid_results) / episode_count,
                "successes": len(successful_results),
                "success_rate": len(successful_results) / episode_count,
                "failures_by_reason": failures_by_reason,
                "mean_success_task_sim_time_s": (
                    fmean(
                        result.task_sim_time_s
                        for result in successful_results
                    )
                    if successful_results
                    else None
                ),
                "mean_wall_time_s": fmean(
                    result.wall_time_s for result in policy_results
                ),
                "mean_episode_return": (
                    fmean(returns) if returns else None
                ),
            }
        )

    return summaries
