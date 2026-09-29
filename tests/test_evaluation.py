"""Tests for the shared policy-evaluation result schema."""

import pytest

from examples.day26.evaluate_policy import (
    run_random_episode,
    run_scripted_expert_episode,
)
from panda_mujoco.evaluation import (
    EpisodeResult,
    summarize_episode_results,
)


def make_result(**overrides) -> EpisodeResult:
    values = {
        "policy_name": "scripted_expert",
        "seed": 42,
        "evaluation_valid": True,
        "success": True,
        "reason": "success",
        "failed_stage": None,
        "reset_sim_time_s": 1.5,
        "task_sim_time_s": 10.0,
        "wall_time_s": 0.8,
        "physics_steps": 5000,
    }
    values.update(overrides)
    return EpisodeResult(**values)


def test_success_result_is_flattened_with_common_outcome():
    result = make_result()

    row = result.to_record()

    assert result.outcome == "success"
    assert row["policy_name"] == "scripted_expert"
    assert row["seed"] == 42
    assert row["success"] is True
    assert row["outcome"] == "success"
    assert row["episode_return"] is None


def test_timeout_is_distinguished_from_task_failure():
    result = make_result(
        policy_name="random_continuous",
        success=False,
        reason="time_limit",
        episode_return=-0.25,
    )

    assert result.outcome == "timeout"
    assert result.to_record()["episode_return"] == -0.25


def test_failure_reason_and_stage_are_preserved():
    result = make_result(
        success=False,
        reason="no_bilateral_contact",
        failed_stage="VERIFY_CONTACT",
    )

    row = result.to_record()

    assert result.outcome == "failure"
    assert row["reason"] == "no_bilateral_contact"
    assert row["failed_stage"] == "VERIFY_CONTACT"


def test_invalid_evaluation_is_not_counted_as_task_failure():
    result = make_result(
        evaluation_valid=False,
        success=False,
        reason="evaluation_error",
        error="AssertionError: observation contains NaN",
    )

    assert result.outcome == "invalid"
    assert result.to_record()["evaluation_valid"] is False


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("reset_sim_time_s", float("nan"), "reset_sim_time_s must be finite"),
        ("task_sim_time_s", -0.1, "task_sim_time_s must be non-negative"),
        ("physics_steps", -1, "physics_steps must be a non-negative integer"),
        ("episode_return", float("inf"), "episode_return must be finite or None"),
    ],
)
def test_invalid_numeric_fields_are_rejected(field, value, message):
    with pytest.raises(ValueError, match=message):
        make_result(**{field: value})


def test_invalid_evaluation_requires_an_error_description():
    with pytest.raises(ValueError, match="must include an error"):
        make_result(
            evaluation_valid=False,
            success=False,
            reason="evaluation_error",
        )


def test_summary_reports_task_success_and_evaluation_validity_separately():
    results = [
        make_result(seed=1),
        make_result(
            seed=2,
            success=False,
            reason="time_limit",
        ),
        make_result(
            seed=3,
            evaluation_valid=False,
            success=False,
            reason="evaluation_error",
            error="RuntimeError: simulation failed",
        ),
    ]

    summary = summarize_episode_results(results)[0]

    assert summary["episodes"] == 3
    assert summary["valid_episodes"] == 2
    assert summary["invalid_episodes"] == 1
    assert summary["valid_rate"] == pytest.approx(2 / 3)
    assert summary["successes"] == 1
    assert summary["success_rate"] == pytest.approx(1 / 3)
    assert summary["failures_by_reason"] == {"time_limit": 1}


def test_summary_averages_reward_only_when_a_policy_has_reward_values():
    results = [
        make_result(policy_name="random", episode_return=0.2),
        make_result(
            policy_name="random",
            seed=43,
            episode_return=0.4,
        ),
        make_result(policy_name="expert", episode_return=None),
    ]

    summaries = {
        row["policy_name"]: row
        for row in summarize_episode_results(results)
    }

    assert summaries["random"]["mean_episode_return"] == pytest.approx(0.3)
    assert summaries["expert"]["mean_episode_return"] is None


def test_summary_requires_at_least_one_episode():
    with pytest.raises(ValueError, match="at least one episode"):
        summarize_episode_results([])


def test_random_policy_runner_records_a_time_limit_as_valid_timeout():
    result = run_random_episode(seed=42, max_steps=1)

    assert result.evaluation_valid is True
    assert result.success is False
    assert result.outcome == "timeout"
    assert result.reason == "time_limit"
    assert result.physics_steps == 10
    assert result.episode_return is not None


def test_scripted_expert_runner_records_success_without_reward():
    result = run_scripted_expert_episode(seed=42)

    assert result.evaluation_valid is True
    assert result.success is True
    assert result.outcome == "success"
    assert result.reason == "success"
    assert result.failed_stage is None
    assert result.episode_return is None
    assert result.reset_sim_time_s > 0.0
    assert result.task_sim_time_s > 0.0
