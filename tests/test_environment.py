"""Day 25 integration checks for environment robustness and failure handling."""

import math

import pytest
from gymnasium.utils.env_checker import check_env

from panda_mujoco.continuous_env import PandaContinuousEnv
from examples.day25.failure_scenarios import (
    run_contact_loss_during_lift,
    run_no_contact,
    run_single_sided_contact,
    run_timeout,
    run_unreachable_target,
)
from examples.day25.random_smoke import run_episode


def test_gymnasium_checker_accepts_continuous_environment():
    """Keep Gymnasium's reset/step API contract in the regression suite."""

    env = PandaContinuousEnv()
    try:
        # Rendering is intentionally skipped because CI runs without a display.
        check_env(env, skip_render_check=True)
    finally:
        env.close()


@pytest.mark.parametrize(
    "scenario_runner",
    [
        pytest.param(run_unreachable_target, id="unreachable-target"),
        pytest.param(run_no_contact, id="no-bilateral-contact"),
        pytest.param(run_single_sided_contact, id="single-sided-contact"),
        pytest.param(run_contact_loss_during_lift, id="contact-loss-during-lift"),
        pytest.param(run_timeout, id="time-limit"),
    ],
)
def test_failure_scenario_reports_expected_safe_outcome(scenario_runner):
    """Faults must be diagnosed correctly and must not advance unsafe stages."""

    scenario = scenario_runner()

    assert scenario.passed, (
        f"{scenario.name}: observed {scenario.observed}; "
        f"expected {scenario.expected}"
    )


def test_random_smoke_timeout_is_robustness_pass_not_task_success():
    """A normal time limit is an episode outcome, not an environment crash."""

    row = run_episode(seed=42, max_steps=1)

    assert row["smoke_passed"] is True
    assert row["steps"] == 1
    assert row["terminated"] is False
    assert row["truncated"] is True
    assert row["success"] is False
    assert row["termination_reason"] == "time_limit"
    assert row["error"] == ""


def test_repeating_random_smoke_seed_repeats_episode_summary():
    """Seeding both reset and action sampling should reproduce the rollout."""

    first = run_episode(seed=42, max_steps=3)
    second = run_episode(seed=42, max_steps=3)

    assert first["smoke_passed"] is True
    assert second["smoke_passed"] is True
    for key in (
        "steps",
        "terminated",
        "truncated",
        "success",
        "termination_reason",
        "ik_failures",
    ):
        assert first[key] == second[key]
    assert math.isclose(
        float(first["episode_return"]),
        float(second["episode_return"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    )
