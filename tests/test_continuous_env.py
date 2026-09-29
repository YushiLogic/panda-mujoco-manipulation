import numpy as np
import pytest

from panda_mujoco.continuous_env import PandaContinuousEnv
from panda_mujoco.kinematics import get_ee_position
from scipy.spatial.transform import Rotation

def test_zero_action_100_steps_is_stable():
    env = PandaContinuousEnv()
    env.reset(seed=42)
    scene = env.task.scene

    start_position = get_ee_position(scene).copy()
    start_target = env.target_position.copy()
    start_time = scene.data.time
    zero_action = np.zeros(7, dtype=np.float32)

    for _ in range(100):
        observation, _, _, _, info = env.step(zero_action)

    drift_m = np.linalg.norm(get_ee_position(scene) - start_position)

    assert info["episode_steps"] == 100
    assert np.array_equal(env.target_position, start_target)
    assert drift_m < 1e-4  # 允许小于 0.1 mm 的数值/仿真误差
    assert np.all(np.isfinite(observation))
    assert scene.data.time - start_time == pytest.approx(2.0)

    env.close()
def test_zero_action_holds_previous_target_while_arm_catches_up():
    env = PandaContinuousEnv()
    env.reset(seed=42)
    start_target = env.target_position.copy()

    move_x = np.zeros(7, dtype=np.float32)
    move_x[0] = 1.0
    _, _, _, _, info = env.step(move_x)

    assert info["ik_success"]
    assert np.allclose(
        env.target_position - start_target,
        [0.002, 0.0, 0.0],
    )

    held_target = env.target_position.copy()
    initial_error = info["tracking_error_m"]

    for _ in range(10):
        _, _, _, _, info = env.step(np.zeros(7, dtype=np.float32))

    assert np.array_equal(env.target_position, held_target)
    assert info["tracking_error_m"] < initial_error
    assert info["episode_steps"] == 11

    env.close()
def test_rotation_and_gripper_actions_update_targets():
    env = PandaContinuousEnv()
    env.reset(seed=42)

    start_position = env.target_position.copy()
    start_rotation = env.target_rotation.copy()
    start_width = env.target_width

    action = np.zeros(7, dtype=np.float32)
    action[5] = 1.0   # 绕世界 z 轴增加 2°
    action[6] = -1.0  # 夹爪目标宽度减少 2 mm

    _, _, _, _, info = env.step(action)

    delta_rotation = env.target_rotation @ start_rotation.T
    rotation_vector_deg = np.rad2deg(
        Rotation.from_matrix(delta_rotation).as_rotvec()
    )

    assert info["ik_success"]
    assert np.allclose(env.target_position, start_position)
    assert np.allclose(rotation_vector_deg, [0.0, 0.0, 2.0])
    assert env.target_width == pytest.approx(start_width - 0.002)

    env.close()

def test_rotation_and_gripper_errors_decrease_while_holding():
    env = PandaContinuousEnv()
    env.reset(seed=42)

    action = np.zeros(7, dtype=np.float32)
    action[5] = 1.0   # 姿态目标 +2°
    action[6] = -1.0  # 夹爪目标闭合 2 mm
    _, _, _, _, before = env.step(action)

    zero_action = np.zeros(7, dtype=np.float32)
    for _ in range(50):
        _, _, _, _, after = env.step(zero_action)

    assert after["orientation_error_rad"] < before["orientation_error_rad"]
    assert after["gripper_width_error_m"] < before["gripper_width_error_m"]
    assert after["tracking_error_m"] < before["tracking_error_m"]

    env.close()

def test_step_returns_reward_breakdown_in_info():
    env = PandaContinuousEnv()
    env.reset(seed=42)

    action = np.zeros(7, dtype=np.float32)
    action[2] = -1.0

    _, reward, terminated, truncated, info = env.step(action)

    breakdown = info["reward_breakdown"]
    component_sum = sum(
        value
        for name, value in breakdown.items()
        if name != "total"
    )

    assert reward == pytest.approx(breakdown["total"])
    assert reward == pytest.approx(component_sum)
    assert breakdown["action_penalty"] == pytest.approx(-0.01 / 7.0)
    assert terminated is False
    assert truncated is False

    env.close()


def test_step_limit_truncates_episode_and_blocks_more_steps():
    env = PandaContinuousEnv(max_episode_steps=1)
    env.reset(seed=42)

    action = np.zeros(7, dtype=np.float32)
    _, _, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is True
    assert info["termination_reason"] == "time_limit"
    assert info["reason"] == "time_limit"

    with pytest.raises(RuntimeError, match="episode has ended"):
        env.step(action)

    env.close()