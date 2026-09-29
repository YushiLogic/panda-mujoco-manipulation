"""A reach-only Gymnasium task built on the project's Panda controller.

The agent controls small Cartesian position/orientation increments. The
existing MuJoCo simulation, IK planner, joint controller, and bias
compensation remain responsible for executing those commands.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from panda_mujoco.action_mapping import ACTION_DIM
from panda_mujoco.continuous_env import PandaContinuousEnv
from panda_mujoco.grasp_task import generate_grasp_targets
from panda_mujoco.kinematics import get_ee_pose
from panda_mujoco.pick_controller import get_cube_yaw
from panda_mujoco.rewards import action_effort_penalty, error_progress_reward
from panda_mujoco.rotations import rotation_error


REACH_ACTION_DIM = 6
REACH_ACTION_INDICES = slice(0, REACH_ACTION_DIM)
POSITION_ERROR_SCALE_M = 0.002
ORIENTATION_ERROR_SCALE_RAD = np.deg2rad(2.0)
ORIENTATION_REWARD_WEIGHT = 0.25
ACTION_PENALTY_WEIGHT = 0.01
SUCCESS_BONUS = 5.0


class PandaReachEnv(gym.Env):
    """Learn to move the Panda end-effector to the cube's grasp pose.

    This environment deliberately stops at reaching the grasp pose; it does
    not close the fingers or attempt to lift the cube. Actions are six
    normalized values in ``[-1, 1]`` for world-frame XYZ and rotation-vector
    increments. The gripper remains at its reset width.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        frame_skip: int = 10,
        max_episode_steps: int = 500,
        position_tolerance_m: float = 0.01,
        orientation_tolerance_rad: float = np.deg2rad(5.0),
    ) -> None:
        super().__init__()

        tolerances = np.asarray(
            [position_tolerance_m, orientation_tolerance_rad],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(tolerances)) or np.any(tolerances <= 0.0):
            raise ValueError("reach tolerances must be finite and positive")

        self.base_env = PandaContinuousEnv(
            frame_skip=frame_skip,
            max_episode_steps=max_episode_steps,
        )
        self.position_tolerance_m = float(position_tolerance_m)
        self.orientation_tolerance_rad = float(orientation_tolerance_rad)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(REACH_ACTION_DIM,),
            dtype=np.float32,
        )
        # Keep the same 29-value state used by the existing continuous task.
        self.observation_space = self.base_env.observation_space
        self._episode_done = True

    def _measure_errors(self) -> tuple[float, float]:
        """Measure EE position/orientation error against the cube grasp pose."""

        scene = self.base_env.task.scene
        cube_position = np.asarray(
            scene.data.body("cube").xpos,
            dtype=np.float64,
        ).copy()
        targets = generate_grasp_targets(
            cube_position,
            cube_yaw=get_cube_yaw(scene),
        )
        ee_position, ee_rotation = get_ee_pose(scene)
        position_error = float(
            np.linalg.norm(targets.grasp_position - ee_position)
        )
        orientation_error_rad = float(
            np.linalg.norm(rotation_error(targets.rotation, ee_rotation))
        )
        return position_error, orientation_error_rad

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        observation, info = self.base_env.reset(seed=seed, options=options)
        self._episode_done = False
        position_error, orientation_error_rad = self._measure_errors()
        info = dict(info)
        info.update(
            {
                "task": "reach",
                "reach_position_error_m": position_error,
                "reach_orientation_error_rad": orientation_error_rad,
                "gripper_held_open": True,
            }
        )
        return observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        if self._episode_done:
            raise RuntimeError("call reset() before step() or after episode end")

        safe_action = np.asarray(action, dtype=np.float64)
        if safe_action.shape != (REACH_ACTION_DIM,):
            raise ValueError("reach action must have shape (6,)")
        if not np.all(np.isfinite(safe_action)):
            raise ValueError("reach action must contain only finite values")
        if np.any(safe_action < -1.0) or np.any(safe_action > 1.0):
            raise ValueError("reach action values must be in [-1, 1]")

        previous_position_error, previous_orientation_error = (
            self._measure_errors()
        )

        # Reuse the project's seven-dimensional action mapping while fixing
        # the seventh (gripper) command to zero, i.e. hold the reset width.
        full_action = np.zeros(ACTION_DIM, dtype=np.float32)
        full_action[REACH_ACTION_INDICES] = safe_action.astype(np.float32)
        observation, _pick_reward, base_terminated, base_truncated, base_info = (
            self.base_env.step(full_action)
        )

        position_error = float(base_info["task_reach_error_m"])
        orientation_error_rad = float(
            base_info["task_orientation_error_rad"]
        )
        reached = (
            position_error <= self.position_tolerance_m
            and orientation_error_rad <= self.orientation_tolerance_rad
        )

        # This task is specifically about reaching. A rare underlying pick
        # termination is preserved as an episode end, but is not mislabeled
        # as a reach success unless both reach tolerances were met.
        terminated = bool(reached or base_terminated)
        truncated = bool(base_truncated and not terminated)
        if reached:
            reason = "reach_success"
        elif base_terminated:
            reason = str(base_info.get("termination_reason", "task_terminated"))
        elif truncated:
            reason = "time_limit"
        else:
            reason = "none"

        reach_progress = error_progress_reward(
            previous_position_error,
            position_error,
            POSITION_ERROR_SCALE_M,
        )
        orientation_progress = ORIENTATION_REWARD_WEIGHT * error_progress_reward(
            previous_orientation_error,
            orientation_error_rad,
            ORIENTATION_ERROR_SCALE_RAD,
        )
        action_penalty = ACTION_PENALTY_WEIGHT * action_effort_penalty(
            safe_action
        )
        success_bonus = SUCCESS_BONUS if reached else 0.0
        reward_parts = {
            "reach_progress": reach_progress,
            "orientation_progress": orientation_progress,
            "success_bonus": success_bonus,
            "action_penalty": action_penalty,
        }
        reward = float(sum(reward_parts.values()))

        self._episode_done = bool(terminated or truncated)
        info = dict(base_info)
        info.update(
            {
                "task": "reach",
                "success": bool(reached),
                "reach_success": bool(reached),
                "reason": reason,
                "task_reach_error_m": position_error,
                "task_orientation_error_rad": orientation_error_rad,
                "position_tolerance_m": self.position_tolerance_m,
                "orientation_tolerance_rad": self.orientation_tolerance_rad,
                "gripper_held_open": True,
                "reward_breakdown": {**reward_parts, "total": reward},
            }
        )
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self.base_env.close()
