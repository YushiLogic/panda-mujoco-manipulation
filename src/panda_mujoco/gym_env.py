"""Gymnasium adapter for the existing scripted pick task."""

import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from panda_mujoco.panda_pick_task import OBSERVATION_SIZE, PandaPickTask
from panda_mujoco.simulation import PandaScene


class PandaScriptedPickEnv(gym.Env):
    """Gymnasium interface for the existing high-level pick actions."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()

        if render_mode not in (None, "human"):
            raise ValueError(f"unsupported render_mode: {render_mode}")

        self.render_mode = render_mode
        self._viewer = None
        self.task = PandaPickTask()

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float64,
        )
        self.action_space = spaces.Discrete(2)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)

        if options:
            raise ValueError("reset options are not supported yet")

        # 未指定seed时，也从Gymnasium管理的随机数生成器取一个，
        # 避免方块随机化脱离环境的随机数管理。
        task_seed = seed
        if task_seed is None:
            task_seed = int(self.np_random.integers(0, 2**32))

        observation, info = self.task.reset(seed=task_seed)

        if not self.observation_space.contains(observation):
            raise RuntimeError("reset observation is outside observation_space")

        return observation, info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action}")

        if self.render_mode == "human" and self.task.has_reset:
            self.render()

        callback = (
            self._render_physics_step
            if self.render_mode == "human"
            else None
        )
        observation, reward, terminated, truncated, info = self.task.step(
            int(action),
            step_callback=callback,
        )

        if not self.observation_space.contains(observation):
            raise RuntimeError("step observation is outside observation_space")

        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        if self.render_mode != "human":
            return

        if self._viewer is None:
            import mujoco.viewer

            scene = self.task.scene
            self._viewer = mujoco.viewer.launch_passive(
                scene.model, scene.data
            )
            self._viewer.cam.lookat[:] = [0.38, 0.0, 0.30]
            self._viewer.cam.distance = 1.5
            self._viewer.cam.azimuth = 135.0
            self._viewer.cam.elevation = -25.0

        if self._viewer.is_running():
            self._viewer.sync()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def _render_physics_step(self, scene: PandaScene) -> None:
        if self._viewer is None or not self._viewer.is_running():
            return

        self._viewer.sync()
        time.sleep(scene.model.opt.timestep)
