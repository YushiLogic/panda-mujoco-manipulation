"""Episode termination decisions for the Panda manipulation task."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TerminationDecision:
    """Describe whether an episode continues, succeeds, or stops."""

    terminated: bool
    truncated: bool
    success: bool
    reason: str


def decide_termination(
    *,
    success: bool,
    failure_reason: str | None,
    episode_steps: int,
    max_episode_steps: int,
) -> TerminationDecision:
    """Choose whether this transition ends or times out the episode.

    Success and explicitly unrecoverable failure are task endings.
    Reaching the step limit without either one is a time-limit truncation.
    """

    if not isinstance(success, bool):
        raise TypeError("success must be a Boolean value")

    if failure_reason is not None:
        if not isinstance(failure_reason, str) or not failure_reason.strip():
            raise ValueError("failure_reason must be None or a non-empty string")

    if (
        not isinstance(episode_steps, int)
        or isinstance(episode_steps, bool)
        or episode_steps < 0
    ):
        raise ValueError("episode_steps must be a non-negative integer")

    if (
        not isinstance(max_episode_steps, int)
        or isinstance(max_episode_steps, bool)
        or max_episode_steps <= 0
    ):
        raise ValueError("max_episode_steps must be a positive integer")

    if success and failure_reason is not None:
        raise ValueError("success and failure_reason cannot both be set")

    # 成功是任务自然结束；即使刚好达到步数上限，也算成功结束。
    if success:
        return TerminationDecision(
            terminated=True,
            truncated=False,
            success=True,
            reason="success",
        )

    # 只有明确判定为不可恢复的失败，才在这里结束回合。
    if failure_reason is not None:
        return TerminationDecision(
            terminated=True,
            truncated=False,
            success=False,
            reason=failure_reason,
        )

    # 达到步数上限属于外部时间限制，不代表任务自然结束。
    if episode_steps >= max_episode_steps:
        return TerminationDecision(
            terminated=False,
            truncated=True,
            success=False,
            reason="time_limit",
        )

    return TerminationDecision(
        terminated=False,
        truncated=False,
        success=False,
        reason="none",
    )