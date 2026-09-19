"""Day 20：Panda抓取任务的轻量接口。

本模块把此前分散的方块随机化、状态读取和Day19脚本抓取状态机，
组织成 ``reset(seed)`` / ``step(action)`` 形式的任务边界。
"""

from collections.abc import Callable
from enum import IntEnum

import mujoco
import numpy as np

from panda_mujoco.arm_control import (
    apply_arm_bias_compensation,
)
from panda_mujoco.cube_reset import (
    get_cube_joint_addresses,
    reset_cube,
    sample_cube_pose,
    settle_cube,
)
from panda_mujoco.gripper import get_gripper_width
from panda_mujoco.kinematics import get_ee_position
from panda_mujoco.pick_controller import get_cube_yaw
from panda_mujoco.pick_controller import (
    PickResult,
    PickState,
    run_scripted_pick,
)
from panda_mujoco.simulation import PandaScene


# 观察向量布局。使用具名slice，避免其他代码到处硬编码魔法下标。
ARM_POSITION_SLICE = slice(0, 7)
ARM_VELOCITY_SLICE = slice(7, 14)
GRIPPER_WIDTH_INDEX = 14
EE_POSITION_SLICE = slice(15, 18)
CUBE_POSITION_SLICE = slice(18, 21)
CUBE_YAW_SIN_COS_SLICE = slice(21, 23)
CUBE_LINEAR_VELOCITY_SLICE = slice(23, 26)
CUBE_ANGULAR_VELOCITY_SLICE = slice(26, 29)

OBSERVATION_SIZE = 29

PhysicsStepCallback = Callable[[PandaScene], None]


class PickAction(IntEnum):
    """Day20轻量任务接口使用的高层离散动作。"""

    WAIT = 0
    RUN_SCRIPTED_PICK = 1


def classify_pick_failure(
    result: PickResult,
) -> str:
    """把详细失败原因归入适合批量统计的类别。"""

    if result.success:
        return "none"

    if result.reason in {
        "ik_failed",
        "non_finite_state",
        "joint_limit_violation",
        "position_tolerance",
        "orientation_tolerance",
        "prepared_scene_not_finite",
    }:
        return "planning_or_motion_failure"

    if result.failed_stage in {
        PickState.CLOSE_GRIPPER,
        PickState.VERIFY_CONTACT,
    }:
        return "contact_failure"

    if result.failed_stage is PickState.LIFT:
        return "lift_failure"

    if result.failed_stage is PickState.CHECK_SUCCESS:
        return "stability_failure"

    return "other_failure"


class PandaPickTask:
    """管理Panda抓取episode的轻量任务接口。

    ``reset(seed)`` 建立一个完整、可复现且落稳的episode初始状态；
    ``step(action)`` 可以等待，也可以运行Day19的完整脚本抓取策略。
    """

    def __init__(
        self,
        scene: PandaScene | None = None,
        *,
        max_episode_steps: int = 5,
        wait_physics_steps: int = 10,
    ) -> None:
        if (
            not isinstance(
                max_episode_steps,
                (int, np.integer),
            )
            or isinstance(max_episode_steps, bool)
            or max_episode_steps <= 0
        ):
            raise ValueError(
                "max_episode_steps must be a positive integer"
            )

        if (
            not isinstance(
                wait_physics_steps,
                (int, np.integer),
            )
            or isinstance(wait_physics_steps, bool)
            or wait_physics_steps <= 0
        ):
            raise ValueError(
                "wait_physics_steps must be a positive integer"
            )

        self.scene = (
            PandaScene() if scene is None else scene
        )
        self.max_episode_steps = int(
            max_episode_steps
        )
        self.wait_physics_steps = int(
            wait_physics_steps
        )

        self.seed: int | None = None
        self.episode_steps = 0
        self.terminated = False
        self.truncated = False
        self.has_reset = False

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """开始一个按seed可复现的新episode。

        reset会彻底清除上一回合的MuJoCo运行时状态，恢复机器人home，
        重新采样并落稳方块，然后返回初始观察和诊断信息。
        """

        if (
            seed is not None
            and (
                not isinstance(seed, (int, np.integer))
                or isinstance(seed, bool)
            )
        ):
            raise ValueError(
                "seed must be an integer or None"
            )

        normalized_seed = (
            None if seed is None else int(seed)
        )

        # mj_resetData清除time、qacc、qfrc_applied等完整运行时状态；
        # reset_to_home再写入本项目定义的安全home位姿和控制量。
        mujoco.mj_resetData(
            self.scene.model,
            self.scene.data,
        )
        self.scene.reset_to_home()

        sampled_pose = sample_cube_pose(
            normalized_seed
        )
        reset_cube(
            self.scene,
            sampled_pose,
        )
        settle_result = settle_cube(self.scene)

        if not settle_result.settled:
            raise RuntimeError(
                "cube failed to settle during task reset"
            )

        self.seed = normalized_seed
        self.episode_steps = 0
        self.terminated = False
        self.truncated = False
        self.has_reset = True

        observation = get_pick_observation(
            self.scene
        )
        cube_yaw = get_cube_yaw(self.scene)

        info: dict[str, object] = {
            "seed": self.seed,
            "episode_steps": self.episode_steps,
            "task_state": "READY",
            "cube_position": (
                settle_result.final_position.copy()
            ),
            "cube_yaw": cube_yaw,
            "settle_duration": (
                settle_result.simulation_duration
            ),
        }

        return observation, info

    def step(
        self,
        action: PickAction | int,
        *,
        step_callback: PhysicsStepCallback | None = None,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, object],
    ]:
        """执行一个上层任务动作。

        ``WAIT`` 保持已有控制目标并推进固定数量的MuJoCo物理步；
        ``RUN_SCRIPTED_PICK`` 复用当前episode并运行Day19状态机。
        """

        if not self.has_reset:
            raise RuntimeError(
                "call reset() before step()"
            )

        if self.terminated or self.truncated:
            raise RuntimeError(
                "episode has ended; call reset() before step()"
            )

        if (
            step_callback is not None
            and not callable(step_callback)
        ):
            raise ValueError(
                "step_callback must be callable or None"
            )

        if isinstance(action, bool):
            raise ValueError(
                "action must be a valid PickAction"
            )

        try:
            normalized_action = PickAction(action)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "action must be a valid PickAction"
            ) from error

        if normalized_action is PickAction.RUN_SCRIPTED_PICK:
            self.episode_steps += 1

            pick_result = run_scripted_pick(
                self.scene,
                seed=self.seed,
                reset_scene=False,
                step_callback=step_callback,
            )

            # RUN_SCRIPTED_PICK一定会得到明确的成功或失败结果，
            # 因此属于terminated，而不是外部时间限制导致的truncated。
            self.terminated = True
            self.truncated = False

            observation = get_pick_observation(
                self.scene
            )
            reward = (
                1.0 if pick_result.success else 0.0
            )

            info: dict[str, object] = {
                "seed": self.seed,
                "episode_steps": self.episode_steps,
                "action": normalized_action.name,
                "task_state": "TERMINATED",
                "success": pick_result.success,
                "final_state": (
                    pick_result.final_state.value
                ),
                "failed_stage": (
                    None
                    if pick_result.failed_stage is None
                    else pick_result.failed_stage.value
                ),
                "reason": pick_result.reason,
                "failure_category": (
                    classify_pick_failure(pick_result)
                ),
                "lift_height": (
                    pick_result.final_status.lift_height
                ),
                "hold_duration": (
                    pick_result.hold_duration
                ),
                "gripper_width": (
                    pick_result.final_status.gripper_width
                ),
                "stage_trace": tuple(
                    record.state.value
                    for record in pick_result.records
                ),
            }

            return (
                observation,
                reward,
                self.terminated,
                self.truncated,
                info,
            )

        # 一次task.step(WAIT)包含多个底层mj_step。仿真时间会推进，
        # 但上层episode_steps只增加1。
        for _ in range(self.wait_physics_steps):
            apply_arm_bias_compensation(
                self.scene
            )
            mujoco.mj_step(
                self.scene.model,
                self.scene.data,
            )
            if step_callback is not None:
                step_callback(self.scene)

        self.episode_steps += 1

        if (
            self.episode_steps
            >= self.max_episode_steps
        ):
            self.truncated = True
            reason = "time_limit"
            task_state = "TRUNCATED"
        else:
            reason = "none"
            task_state = "RUNNING"

        observation = get_pick_observation(
            self.scene
        )

        reward = 0.0
        info: dict[str, object] = {
            "seed": self.seed,
            "episode_steps": self.episode_steps,
            "action": normalized_action.name,
            "task_state": task_state,
            "success": False,
            "reason": reason,
        }

        return (
            observation,
            reward,
            self.terminated,
            self.truncated,
            info,
        )


def get_pick_observation(
    scene: PandaScene,
) -> np.ndarray:
    """读取当前场景并返回形状为 ``(29,)`` 的任务观察向量。

    布局依次为：

    - 7维机械臂关节位置，单位rad；
    - 7维机械臂关节速度，单位rad/s；
    - 1维夹爪实际总开口，单位m；
    - 3维末端世界位置，单位m；
    - 3维方块世界位置，单位m；
    - 2维方块yaw的sin/cos，无量纲；
    - 3维方块线速度，单位m/s；
    - 3维方块角速度，单位rad/s。

    返回值由新数组组成，修改它不会反向修改MuJoCo的 ``MjData``。
    """

    observation = np.empty(
        OBSERVATION_SIZE,
        dtype=float,
    )

    # Panda手臂的七个标量关节位于qpos和qvel的前七项。
    observation[ARM_POSITION_SLICE] = (
        scene.data.qpos[:7]
    )
    observation[ARM_VELOCITY_SLICE] = (
        scene.data.qvel[:7]
    )

    # 读取实际手指位置之和，而不是夹爪ctrl命令。
    observation[GRIPPER_WIDTH_INDEX] = (
        get_gripper_width(scene)
    )

    observation[EE_POSITION_SLICE] = (
        get_ee_position(scene)
    )
    observation[CUBE_POSITION_SLICE] = (
        scene.data.body("cube").xpos
    )

    cube_yaw = get_cube_yaw(scene)
    observation[CUBE_YAW_SIN_COS_SLICE] = (
        np.sin(cube_yaw),
        np.cos(cube_yaw),
    )

    # cube是free joint，其qvel连续占用6项：前三项线速度，
    # 后三项角速度。通过模型地址读取，避免硬编码下标9。
    _, cube_qvel_address = get_cube_joint_addresses(
        scene.model
    )
    cube_velocity = scene.data.qvel[
        cube_qvel_address:cube_qvel_address + 6
    ]

    observation[CUBE_LINEAR_VELOCITY_SLICE] = (
        cube_velocity[:3]
    )
    observation[CUBE_ANGULAR_VELOCITY_SLICE] = (
        cube_velocity[3:]
    )

    if observation.shape != (OBSERVATION_SIZE,):
        raise RuntimeError(
            "pick observation has an unexpected shape"
        )

    if not np.all(np.isfinite(observation)):
        raise RuntimeError(
            "pick observation contains NaN or Inf"
        )

    return observation
