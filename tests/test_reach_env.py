import numpy as np
import pytest

from panda_mujoco.reach_env import PandaReachEnv


def test_reach_env_reset_exposes_six_axis_actions_and_finite_observation():
    env = PandaReachEnv(max_episode_steps=5)
    try:
        observation, info = env.reset(seed=42)

        assert env.action_space.shape == (6,)
        assert observation.shape == (29,)
        assert np.isfinite(observation).all()
        assert env.observation_space.contains(observation)
        assert info["task"] == "reach"
        assert info["gripper_held_open"] is True
        assert info["reach_position_error_m"] > 0.0
    finally:
        env.close()


def test_zero_action_keeps_reach_episode_valid_and_gripper_open():
    env = PandaReachEnv(max_episode_steps=5)
    try:
        env.reset(seed=42)
        observation, reward, terminated, truncated, info = env.step(
            np.zeros(6, dtype=np.float32)
        )

        assert env.observation_space.contains(observation)
        assert np.isfinite(reward)
        assert not terminated
        assert not truncated
        assert info["success"] is False
        assert info["gripper_held_open"] is True
    finally:
        env.close()


@pytest.mark.parametrize(
    "action",
    [
        np.zeros(5, dtype=np.float32),
        np.full(6, np.nan, dtype=np.float32),
        np.full(6, 1.1, dtype=np.float32),
    ],
)
def test_invalid_reach_actions_are_rejected(action):
    env = PandaReachEnv(max_episode_steps=5)
    try:
        env.reset(seed=42)
        with pytest.raises(ValueError):
            env.step(action)
    finally:
        env.close()


@pytest.mark.parametrize(
    "position_tolerance,orientation_tolerance",
    [
        (0.0, 0.1),
        (0.01, -0.1),
        (float("inf"), 0.1),
    ],
)
def test_invalid_reach_tolerances_are_rejected(
    position_tolerance,
    orientation_tolerance,
):
    with pytest.raises(ValueError, match="finite and positive"):
        PandaReachEnv(
            position_tolerance_m=position_tolerance,
            orientation_tolerance_rad=orientation_tolerance,
        )
