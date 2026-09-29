"""Tests for the pure reward functions."""

import numpy as np
import pytest

from panda_mujoco.rewards import (
    RewardWeights,
    action_effort_penalty,
    compute_reward,
    condition_bonus,
    error_progress_reward,
    increase_progress_reward,
)

def test_error_reduction_produces_positive_reward() -> None:
    """距离误差减小1 mm，尺度为2 mm，应获得+0.5奖励。"""

    reward = error_progress_reward(
        previous_error=0.030,
        current_error=0.029,
        scale=0.002,
    )

    assert reward == pytest.approx(0.5)


def test_error_increase_produces_negative_reward() -> None:
    """距离误差增大，说明动作让机械臂远离了目标。"""

    reward = error_progress_reward(
        previous_error=0.030,
        current_error=0.031,
        scale=0.002,
    )

    assert reward == pytest.approx(-0.5)


def test_unchanged_error_produces_zero_reward() -> None:
    """误差不变时，没有进步，也没有退步。"""

    reward = error_progress_reward(
        previous_error=0.030,
        current_error=0.030,
        scale=0.002,
    )

    assert reward == pytest.approx(0.0)


def test_large_error_change_is_clipped() -> None:
    """奖励应被限制在[-1, 1]，避免单步奖励过大。"""

    positive_reward = error_progress_reward(
        previous_error=0.030,
        current_error=0.020,
        scale=0.002,
    )

    negative_reward = error_progress_reward(
        previous_error=0.030,
        current_error=0.040,
        scale=0.002,
    )

    assert positive_reward == pytest.approx(1.0)
    assert negative_reward == pytest.approx(-1.0)


def test_lift_progress_produces_positive_reward() -> None:
    """方块上升时，抬升进度奖励应为正数。"""

    reward = increase_progress_reward(
        previous_value=0.010,
        current_value=0.011,
        scale=0.002,
    )

    assert reward == pytest.approx(0.5)


def test_falling_cube_produces_negative_lift_reward() -> None:
    """方块下落时，抬升进度奖励应为负数。"""

    reward = increase_progress_reward(
        previous_value=0.011,
        current_value=0.010,
        scale=0.002,
    )

    assert reward == pytest.approx(-0.5)


@pytest.mark.parametrize(
    "invalid_value",
    [np.nan, np.inf, -np.inf],
)
def test_non_finite_reward_input_is_rejected(
    invalid_value: float,
) -> None:
    """NaN和无穷大不能进入奖励计算。"""

    with pytest.raises(ValueError, match="finite"):
        error_progress_reward(
            previous_error=invalid_value,
            current_error=0.01,
            scale=0.002,
        )


@pytest.mark.parametrize("invalid_scale", [0.0, -0.001])
def test_non_positive_scale_is_rejected(
    invalid_scale: float,
) -> None:
    """奖励尺度必须大于零。"""

    with pytest.raises(ValueError, match="positive"):
        increase_progress_reward(
            previous_value=0.01,
            current_value=0.02,
            scale=invalid_scale,
        )
@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        (False, 0.0),
        (True, 1.0),
    ],
)
def test_condition_bonus_is_binary(
    condition: bool,
    expected: float,
) -> None:
    """布尔条件只能产生0或1两种奖励。"""

    assert condition_bonus(condition) == expected


def test_condition_bonus_rejects_non_boolean_input() -> None:
    """避免误把数字、字符串等对象当作任务状态。"""

    with pytest.raises(TypeError, match="Boolean"):
        condition_bonus(1)


def test_zero_action_has_no_effort_penalty() -> None:
    """不改变目标的零动作不产生动作幅值惩罚。"""

    action = np.zeros(7)

    assert action_effort_penalty(action) == pytest.approx(0.0)


def test_single_full_axis_has_one_seventh_penalty() -> None:
    """七维动作中只有一个维度为1，平均平方为1/7。"""

    action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    assert action_effort_penalty(action) == pytest.approx(-1.0 / 7.0)


def test_full_action_has_maximum_effort_penalty() -> None:
    """所有动作维度达到最大值时，惩罚应为-1。"""

    action = np.ones(7)

    assert action_effort_penalty(action) == pytest.approx(-1.0)


def test_action_penalty_uses_clipped_action() -> None:
    """超过动作范围的输入按照执行时的裁剪值计算。"""

    action = np.full(7, 2.0)

    assert action_effort_penalty(action) == pytest.approx(-1.0)


@pytest.mark.parametrize(
    "invalid_action",
    [
        np.array([[0.0, 0.0]]),
        np.array([]),
        np.array([0.0, np.nan]),
        np.array([0.0, np.inf]),
    ],
)

def test_invalid_action_penalty_input_is_rejected(
    invalid_action: np.ndarray,
) -> None:
    """动作必须是一维、非空并且全部有限。"""

    with pytest.raises(ValueError):
        action_effort_penalty(invalid_action)

def _default_reward_inputs() -> dict:
    """提供一组没有接触、没有抬升的默认状态变化。"""

    return {
        "previous_reach_error_m": 0.030,
        "current_reach_error_m": 0.029,
        "previous_orientation_error_rad": 0.0,
        "current_orientation_error_rad": 0.0,
        "previous_persistent_contact": False,
        "persistent_contact": False,
        "previous_lift_height_m": 0.0,
        "current_lift_height_m": 0.0,
        "success": False,
        "action": np.zeros(7),
    }


def test_compute_reward_combines_progress_components() -> None:
    """统一函数应把输入状态变化转换为奖励明细。"""

    result = compute_reward(**_default_reward_inputs())

    assert result.reach_progress == pytest.approx(0.5)
    assert result.orientation_progress == pytest.approx(0.0)
    assert result.contact_bonus == pytest.approx(0.0)
    assert result.lift_progress == pytest.approx(0.0)
    assert result.success_bonus == pytest.approx(0.0)
    assert result.action_penalty == pytest.approx(0.0)
    assert result.total == pytest.approx(0.5)


def test_contact_bonus_is_paid_only_when_persistent_contact_starts() -> None:
    """持续接触的后续时间步不应重复获得接触奖励。"""

    first_step = _default_reward_inputs()
    first_step["persistent_contact"] = True
    first_result = compute_reward(**first_step)

    next_step = _default_reward_inputs()
    next_step["previous_persistent_contact"] = True
    next_step["persistent_contact"] = True
    next_result = compute_reward(**next_step)

    assert first_result.contact_bonus == pytest.approx(0.5)
    assert next_result.contact_bonus == pytest.approx(0.0)


def test_success_bonus_uses_explicit_success_flag() -> None:
    """成功奖励由明确的成功布尔值触发。"""

    inputs = _default_reward_inputs()
    inputs["success"] = True

    result = compute_reward(**inputs)

    assert result.success_bonus == pytest.approx(5.0)


def test_reward_weights_must_be_non_negative() -> None:
    """奖励权重不能是负数。"""

    with pytest.raises(ValueError, match="non-negative"):
        RewardWeights(reach_progress=-1.0)
