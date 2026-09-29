"""Tests for episode termination decisions."""

import pytest

from panda_mujoco.termination import decide_termination


def test_episode_continues_before_step_limit() -> None:
    result = decide_termination(
        success=False,
        failure_reason=None,
        episode_steps=4,
        max_episode_steps=5,
    )

    assert result.terminated is False
    assert result.truncated is False
    assert result.success is False
    assert result.reason == "none"


def test_success_at_step_limit_is_terminated() -> None:
    """成功恰好发生在最后一步，仍然算任务成功结束。"""

    result = decide_termination(
        success=True,
        failure_reason=None,
        episode_steps=5,
        max_episode_steps=5,
    )

    assert result.terminated is True
    assert result.truncated is False
    assert result.success is True
    assert result.reason == "success"


def test_unrecoverable_failure_is_terminated() -> None:
    result = decide_termination(
        success=False,
        failure_reason="cube_dropped",
        episode_steps=3,
        max_episode_steps=5,
    )

    assert result.terminated is True
    assert result.truncated is False
    assert result.success is False
    assert result.reason == "cube_dropped"


def test_reaching_step_limit_is_truncated() -> None:
    result = decide_termination(
        success=False,
        failure_reason=None,
        episode_steps=5,
        max_episode_steps=5,
    )

    assert result.terminated is False
    assert result.truncated is True
    assert result.success is False
    assert result.reason == "time_limit"


@pytest.mark.parametrize(
    ("episode_steps", "max_episode_steps"),
    [
        (-1, 5),
        (1, 0),
        (1.5, 5),
    ],
)
def test_invalid_step_counts_are_rejected(
    episode_steps: int,
    max_episode_steps: int,
) -> None:
    with pytest.raises(ValueError):
        decide_termination(
            success=False,
            failure_reason=None,
            episode_steps=episode_steps,
            max_episode_steps=max_episode_steps,
        )


def test_success_and_failure_cannot_both_be_set() -> None:
    with pytest.raises(ValueError, match="cannot both"):
        decide_termination(
            success=True,
            failure_reason="cube_dropped",
            episode_steps=3,
            max_episode_steps=5,
        )