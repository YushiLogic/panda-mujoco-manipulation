"""Pure reward functions for the Panda manipulation environment."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class RewardBreakdown:
    """Store the traceable contributions of one reward calculation."""

    reach_progress: float
    orientation_progress: float
    contact_bonus: float
    lift_progress: float
    success_bonus: float
    action_penalty: float

    def __post_init__(self) -> None:
        """Ensure that every reward component is finite."""

        components = np.asarray(
            [
                self.reach_progress,
                self.orientation_progress,
                self.contact_bonus,
                self.lift_progress,
                self.success_bonus,
                self.action_penalty,
            ],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(components)):
            raise ValueError("reward components must be finite")

    @property
    def total(self) -> float:
        """Return the scalar reward used by Gymnasium."""

        return float(
            self.reach_progress
            + self.orientation_progress
            + self.contact_bonus
            + self.lift_progress
            + self.success_bonus
            + self.action_penalty
        )

    def as_dict(self) -> dict[str, float]:
        """Return all components in a form suitable for info."""

        return {
            "reach_progress": float(self.reach_progress),
            "orientation_progress": float(
                self.orientation_progress
            ),
            "contact_bonus": float(self.contact_bonus),
            "lift_progress": float(self.lift_progress),
            "success_bonus": float(self.success_bonus),
            "action_penalty": float(self.action_penalty),
            "total": self.total,
        }

def error_progress_reward(
    previous_error: float,
    current_error: float,
    scale: float,
) -> float:
    """Reward a reduction in a non-negative error value.

    Examples
    --------
    If the distance error changes from 0.030 m to 0.029 m,
    with a scale of 0.002 m:

        reward = (0.030 - 0.029) / 0.002 = 0.5

    The result is clipped to the interval [-1, 1].
    """

    values = np.asarray(
        [previous_error, current_error, scale],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(values)):
        raise ValueError("reward inputs must be finite")

    if previous_error < 0.0 or current_error < 0.0:
        raise ValueError("error values must be non-negative")

    if scale <= 0.0:
        raise ValueError("scale must be positive")

    raw_reward = (previous_error - current_error) / scale

    return float(np.clip(raw_reward, -1.0, 1.0))

def increase_progress_reward(
    previous_value: float,
    current_value: float,
    scale: float,
) -> float:
    """Reward an increase in a quantity such as cube lift height."""

    values = np.asarray(
        [previous_value, current_value, scale],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(values)):
        raise ValueError("reward inputs must be finite")

    if scale <= 0.0:
        raise ValueError("scale must be positive")

    raw_reward = (current_value - previous_value) / scale

    return float(np.clip(raw_reward, -1.0, 1.0))

def condition_bonus(condition: bool) -> float:
    """Convert a Boolean task condition into a binary reward.

    Examples
    --------
    Persistent bilateral contact:
        False -> 0.0
        True  -> 1.0

    Task success:
        False -> 0.0
        True  -> 1.0
    """

    if not isinstance(condition, (bool, np.bool_)):
        raise TypeError("condition must be a Boolean value")

    return 1.0 if condition else 0.0


def action_effort_penalty(action: np.ndarray) -> float:
    """Penalize large normalized actions.

    The normalized action is expected to use approximately the range
    [-1, 1]. Values outside that range are clipped because the action
    mapping also clips commands before applying them.

    The returned penalty lies in [-1, 0]:

        all zeros ->  0.0
        all ones  -> -1.0
    """

    action_array = np.asarray(action, dtype=np.float64)

    if action_array.ndim != 1 or action_array.size == 0:
        raise ValueError("action must be a non-empty one-dimensional array")

    if not np.all(np.isfinite(action_array)):
        raise ValueError("action must contain only finite values")

    clipped_action = np.clip(action_array, -1.0, 1.0)

    mean_squared_action = np.mean(np.square(clipped_action))

    return -float(mean_squared_action)

@dataclass(frozen=True)
class RewardWeights:
    """权重用于调节各奖励分量对总分的影响。"""

    reach_progress: float = 1.0
    orientation_progress: float = 0.25
    contact_bonus: float = 0.5
    lift_progress: float = 1.0
    success_bonus: float = 5.0
    action_penalty: float = 0.01

    def __post_init__(self) -> None:
        """权重必须是有限的非负数。"""

        values = np.asarray(
            [
                self.reach_progress,
                self.orientation_progress,
                self.contact_bonus,
                self.lift_progress,
                self.success_bonus,
                self.action_penalty,
            ],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(values)):
            raise ValueError("reward weights must be finite")

        if np.any(values < 0.0):
            raise ValueError("reward weights must be non-negative")


def compute_reward(
    *,
    previous_reach_error_m: float,
    current_reach_error_m: float,
    previous_orientation_error_rad: float,
    current_orientation_error_rad: float,
    previous_persistent_contact: bool,
    persistent_contact: bool,
    previous_lift_height_m: float,
    current_lift_height_m: float,
    success: bool,
    action: np.ndarray,
    weights: RewardWeights = RewardWeights(),
) -> RewardBreakdown:
    """根据一次动作前后的状态变化，计算奖励明细。"""

    reach = weights.reach_progress * error_progress_reward(
        previous_reach_error_m,
        current_reach_error_m,
        scale=0.002,
    )

    orientation = weights.orientation_progress * error_progress_reward(
        previous_orientation_error_rad,
        current_orientation_error_rad,
        scale=np.deg2rad(2.0),
    )

    # 只奖励“刚刚形成”持续双侧接触的那一步，避免每步重复领奖。
    new_persistent_contact = (
        persistent_contact and not previous_persistent_contact
    )
    contact = weights.contact_bonus * condition_bonus(
        new_persistent_contact
    )

    lift = weights.lift_progress * increase_progress_reward(
        previous_lift_height_m,
        current_lift_height_m,
        scale=0.002,
    )

    # 成功后环境应结束本回合，因此成功奖励只应发放一次。
    success_reward = weights.success_bonus * condition_bonus(success)

    effort = weights.action_penalty * action_effort_penalty(action)

    return RewardBreakdown(
        reach_progress=reach,
        orientation_progress=orientation,
        contact_bonus=contact,
        lift_progress=lift,
        success_bonus=success_reward,
        action_penalty=effort,
    )