import numpy as np
import pytest

from panda_mujoco.gym_env import PandaScriptedPickEnv


def test_reset_is_reproducible_and_observation_is_in_space():
    env = PandaScriptedPickEnv()
    try:
        first, info = env.reset(seed=42)
        second, _ = env.reset(seed=42)

        assert info["seed"] == 42
        assert env.observation_space.contains(first)
        assert np.array_equal(first, second)
    finally:
        env.close()


def test_wait_step_does_not_end_episode():
    env = PandaScriptedPickEnv()
    try:
        env.reset(seed=42)
        observation, reward, terminated, truncated, info = env.step(0)

        assert env.observation_space.contains(observation)
        assert reward == 0.0
        assert terminated is False
        assert truncated is False
        assert info["action"] == "WAIT"
        assert info["episode_steps"] == 1
    finally:
        env.close()


def test_action_outside_space_is_rejected():
    env = PandaScriptedPickEnv()
    try:
        env.reset(seed=42)

        with pytest.raises(ValueError, match="invalid action"):
            env.step(2)
    finally:
        env.close()
def test_scripted_pick_reports_successful_termination():
    env = PandaScriptedPickEnv()
    try:
        env.reset(seed=42)
        observation, reward, terminated, truncated, info = env.step(1)

        assert env.observation_space.contains(observation)
        assert reward == 1.0
        assert terminated is True
        assert truncated is False
        assert info["success"] is True
        assert info["final_state"] == "DONE"
        assert info["lift_height"] > 0.05
    finally:
        env.close()
def test_waiting_until_time_limit_truncates_episode():
    env = PandaScriptedPickEnv()
    try:
        env.reset(seed=42)

        for _ in range(4):
            _, reward, terminated, truncated, _ = env.step(0)
            assert reward == 0.0
            assert terminated is False
            assert truncated is False

        _, reward, terminated, truncated, info = env.step(0)
        assert reward == 0.0
        assert terminated is False
        assert truncated is True
        assert info["episode_steps"] == 5
        assert info["reason"] == "time_limit"
    finally:
        env.close()