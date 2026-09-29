"""7 维增量动作的 Panda 环境。"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from panda_mujoco.contact import GraspMonitor
from panda_mujoco.gripper import GRIPPER_OPEN_WIDTH, get_gripper_width
from panda_mujoco.kinematics import get_ee_pose
from panda_mujoco.panda_pick_task import OBSERVATION_SIZE, PandaPickTask
import mujoco
from panda_mujoco.rotations import rotation_error

from panda_mujoco.grasp_task import generate_grasp_targets
from panda_mujoco.pick_controller import get_cube_yaw
from panda_mujoco.rewards import compute_reward
from panda_mujoco.termination import decide_termination

from panda_mujoco.action_mapping import (
    clip_normalized_action,
    map_action_to_targets,
)
from panda_mujoco.arm_control import (
    apply_arm_bias_compensation,
    command_arm_joint_positions,
)
from panda_mujoco.gripper import command_gripper
from panda_mujoco.motion import plan_pose_target
from panda_mujoco.panda_pick_task import get_pick_observation

class PandaContinuousEnv(gym.Env):
    """每个动作指定一次小幅末端/夹爪目标变化。"""

    metadata = {"render_modes": []}  # 先做无窗口版本，之后加入动画

    def __init__(
        self,
        frame_skip: int = 10,
        max_episode_steps: int = 500,
    ) -> None:
        super().__init__()

        if not isinstance(frame_skip, int) or frame_skip <= 0:
            raise ValueError("frame_skip must be a positive integer")

        if (
            not isinstance(max_episode_steps, int)
            or isinstance(max_episode_steps, bool)
            or max_episode_steps <= 0
        ):
            raise ValueError("max_episode_steps must be a positive integer")

        self.task = PandaPickTask()
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.episode_steps = 0

        self.grasp_monitor: GraspMonitor | None = None
        self.grasp_status = None
        self.ever_lifted = False
        self.terminated = False
        self.truncated = False
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float64,
        )

        self.target_position: np.ndarray | None = None
        self.target_rotation: np.ndarray | None = None
        self.target_width: float | None = None
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        if options:
            raise ValueError("reset options are not supported yet")

        task_seed = seed
        if task_seed is None:
            task_seed = int(self.np_random.integers(0, 2**32))

        observation, info = self.task.reset(seed=task_seed)
        scene = self.task.scene
        self.grasp_monitor = GraspMonitor(
        scene,
        reference_cube_height=float(
            scene.data.body("cube").xpos[2]
        ),
    )
        self.grasp_status = self.grasp_monitor.update(scene)

        self.ever_lifted = False
        self.terminated = False
        self.truncated = False

        self.target_position, self.target_rotation = get_ee_pose(scene)
        self.target_width = float(
            np.clip(get_gripper_width(scene), 0.0, GRIPPER_OPEN_WIDTH)
        )
        self.episode_steps = 0

        return observation, info

    def _get_task_errors(
        self,
        scene,
    ) -> tuple[float, float]:
        """计算末端到方块抓取目标的位置和姿态误差。"""

        cube_position = np.asarray(
            scene.data.body("cube").xpos,
            dtype=np.float64,
        ).copy()
        cube_yaw = get_cube_yaw(scene)

        grasp_targets = generate_grasp_targets(
            cube_position,
            cube_yaw=cube_yaw,
        )

        ee_position, ee_rotation = get_ee_pose(scene)

        reach_error_m = float(
            np.linalg.norm(
                grasp_targets.grasp_position - ee_position
            )
        )
        orientation_error_rad = float(
            np.linalg.norm(
                rotation_error(
                    grasp_targets.rotation,
                    ee_rotation,
                )
            )
        )

        return reach_error_m, orientation_error_rad

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        if self.target_position is None:
            raise RuntimeError("call reset() before step()")

        if self.grasp_monitor is None or self.grasp_status is None:
            raise RuntimeError("grasp monitor is not initialized")

        if self.terminated or self.truncated:
            raise RuntimeError("episode has ended; call reset() before step()")

        scene = self.task.scene
        monitor = self.grasp_monitor

        # 保存动作前的任务误差和抓取状态，用来计算“这一步进步了多少”。
        previous_reach_error_m, previous_orientation_error_rad = (
            self._get_task_errors(scene)
        )
        previous_status = self.grasp_status

        safe_action = clip_normalized_action(action)

        next_position, next_rotation, next_width = map_action_to_targets(
            safe_action,
            self.target_position,
            self.target_rotation,
            self.target_width,
        )
        plan = plan_pose_target(scene, next_position, next_rotation)

        if plan.success:
            command_arm_joint_positions(scene, plan.joint_target)
            command_gripper(scene, next_width)

            self.target_position = next_position
            self.target_rotation = next_rotation
            self.target_width = next_width

        # 每个物理步都更新监测器，使它能够累计连续接触时间。
        for _ in range(self.frame_skip):
            apply_arm_bias_compensation(scene)
            mujoco.mj_step(scene.model, scene.data)
            self.grasp_status = monitor.update(scene)

            # 记住方块是否曾经达到“已抬起”的高度，之后才能识别掉落。
            if self.grasp_status.cube_lifted:
                self.ever_lifted = True

        self.episode_steps += 1
        status = self.grasp_status

        current_reach_error_m, current_orientation_error_rad = (
            self._get_task_errors(scene)
        )

        dropped = (
            self.ever_lifted
            and status.contact_snapshot.cube_on_floor
        )
        failure_reason = "cube_dropped" if dropped else None

        decision = decide_termination(
            success=bool(status.grasp_success),
            failure_reason=failure_reason,
            episode_steps=self.episode_steps,
            max_episode_steps=self.max_episode_steps,
        )
        self.terminated = decision.terminated
        self.truncated = decision.truncated

        reward_breakdown = compute_reward(
            previous_reach_error_m=previous_reach_error_m,
            current_reach_error_m=current_reach_error_m,
            previous_orientation_error_rad=(
                previous_orientation_error_rad
            ),
            current_orientation_error_rad=(
                current_orientation_error_rad
            ),
            previous_persistent_contact=(
                previous_status.persistent_contact
            ),
            persistent_contact=status.persistent_contact,
            previous_lift_height_m=previous_status.lift_height,
            current_lift_height_m=status.lift_height,
            success=decision.success,
            action=safe_action,
        )

        observation = get_pick_observation(scene)
        if not self.observation_space.contains(observation):
            raise RuntimeError("step observation is outside observation_space")

        actual_position, actual_rotation = get_ee_pose(scene)

        info = {
            "episode_steps": self.episode_steps,
            "ik_success": plan.success,
            "ik_reason": plan.reason,
            "reason": (
                decision.reason
                if decision.reason != "none"
                else plan.reason
            ),
            "termination_reason": decision.reason,
            "success": decision.success,
            "dropped": dropped,
            "grasp_reason": status.reason,
            "persistent_contact": status.persistent_contact,
            "lift_height_m": status.lift_height,
            "clipped_action": safe_action.copy(),
            "target_position": self.target_position.copy(),
            "tracking_error_m": float(
                np.linalg.norm(self.target_position - actual_position)
            ),
            "task_reach_error_m": current_reach_error_m,
            "task_orientation_error_rad": current_orientation_error_rad,
            "control_period_s": float(
                self.frame_skip * scene.model.opt.timestep
            ),
            "orientation_error_rad": float(
                np.linalg.norm(
                    rotation_error(self.target_rotation, actual_rotation)
                )
            ),
            "gripper_width_error_m": float(
                abs(self.target_width - get_gripper_width(scene))
            ),
            "reward_breakdown": reward_breakdown.as_dict(),
        }

        return (
            observation,
            reward_breakdown.total,
            decision.terminated,
            decision.truncated,
            info,
        )