"""Day 19：用有限状态机组织一次完整的脚本抓取。

本模块不替代前面已经完成的运动、夹爪和监测代码，而是负责决定：
当前应该执行哪个阶段、什么条件下进入下一阶段，以及失败时在哪里停止。
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
import math

import mujoco
import numpy as np

from panda_mujoco.arm_control import apply_arm_bias_compensation
from panda_mujoco.contact import GraspMonitor, GraspStatus
from panda_mujoco.cube_reset import reset_cube, sample_cube_pose, settle_cube
from panda_mujoco.grasp_task import GraspTargets, generate_grasp_targets
from panda_mujoco.gripper import close_gripper, open_gripper
from panda_mujoco.motion import MotionResult, move_end_effector_to
from panda_mujoco.simulation import PandaScene


class PickState(str, Enum):
    """脚本抓取状态机中的全部状态。"""

    RESET = "RESET"
    MOVE_ABOVE = "MOVE_ABOVE"
    APPROACH = "APPROACH"
    CLOSE_GRIPPER = "CLOSE_GRIPPER"
    VERIFY_CONTACT = "VERIFY_CONTACT"
    LIFT = "LIFT"
    CHECK_SUCCESS = "CHECK_SUCCESS"
    DONE = "DONE"
    FAILED = "FAILED"


@dataclass(frozen=True)
class PickStageRecord:
    """一个状态执行结束时留下的可追踪记录。"""

    state: PickState
    success: bool
    reason: str
    simulation_duration: float


@dataclass(frozen=True)
class PickResult:
    """一次完整抓取流程的最终结果。"""

    success: bool
    final_state: PickState
    reason: str
    failed_stage: PickState | None
    records: tuple[PickStageRecord, ...]
    targets: GraspTargets
    final_status: GraspStatus
    hold_duration: float


StepCallback = Callable[[PandaScene], None]


def get_cube_yaw(scene: PandaScene) -> float:
    """从方块的实际旋转矩阵读取世界坐标系中的 yaw。"""

    rotation = scene.data.body("cube").xmat.reshape(3, 3)
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def _step_once(
    scene: PandaScene,
    *,
    monitor: GraspMonitor | None = None,
    step_callback: StepCallback | None = None,
) -> GraspStatus | None:
    """保持手臂补偿并推进一个物理步，可同时更新监测器和画面。"""

    apply_arm_bias_compensation(scene)
    mujoco.mj_step(scene.model, scene.data)

    status = None if monitor is None else monitor.update(scene)

    if step_callback is not None:
        step_callback(scene)

    return status


def _run_motion(
    scene: PandaScene,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    *,
    command_count: int,
    monitor: GraspMonitor | None = None,
    step_callback: StepCallback | None = None,
) -> MotionResult:
    """执行运动原语，并在需要时把监测器接入每一个物理步。"""

    def combined_callback(callback_scene: PandaScene) -> None:
        if monitor is not None:
            monitor.update(callback_scene)
        if step_callback is not None:
            step_callback(callback_scene)

    return move_end_effector_to(
        scene,
        target_position,
        target_rotation,
        command_count=command_count,
        physics_steps_per_command=10,
        settle_steps=250,
        step_callback=combined_callback,
    )


def run_scripted_pick(
    scene: PandaScene,
    *,
    seed: int = 42,
    reset_scene: bool = True,
    close_timeout: float = 1.0,
    hold_duration: float = 0.5,
    step_callback: StepCallback | None = None,
) -> PickResult:
    """执行 RESET 到 DONE/FAILED 的完整固定策略抓取。

    这是一个同步状态机：函数一次执行完一个阶段，再依据该阶段结果决定
    是否进入下一阶段。它仍然是真实动力学控制，因为运动过程通过 ctrl 和
    ``mj_step`` 完成，而不是直接改写机械臂 qpos。
    """

    timestep = float(scene.model.opt.timestep)
    if close_timeout <= 0.0 or not np.isfinite(close_timeout):
        raise ValueError("close_timeout must be positive and finite")
    if hold_duration <= 0.0 or not np.isfinite(hold_duration):
        raise ValueError("hold_duration must be positive and finite")

    if not isinstance(reset_scene, bool):
        raise ValueError(
            "reset_scene must be a bool"
        )

    records: list[PickStageRecord] = []

    # ---------- RESET或检查已经由任务接口准备好的场景 ----------
    if reset_scene:
        scene.reset_to_home()
        sampled_pose = sample_cube_pose(seed)
        reset_cube(scene, sampled_pose)
        settle_result = settle_cube(scene)

        reset_success = settle_result.settled
        reset_reason = (
            "none"
            if reset_success
            else "cube_not_settled"
        )
        reset_duration = (
            settle_result.simulation_duration
        )
        cube_position = (
            settle_result.final_position.copy()
        )
    else:
        # PandaPickTask.reset(seed)已经完成了home、方块随机化与落稳。
        # 这里直接读取当前实际状态，避免step()悄悄开启第二个episode。
        cube_position = (
            scene.data.body("cube").xpos.copy()
        )
        reset_success = bool(
            np.all(np.isfinite(scene.data.qpos))
            and np.all(np.isfinite(scene.data.qvel))
            and np.all(np.isfinite(scene.data.ctrl))
            and np.all(np.isfinite(cube_position))
        )
        reset_reason = (
            "none"
            if reset_success
            else "prepared_scene_not_finite"
        )
        reset_duration = 0.0

    cube_yaw = get_cube_yaw(scene)
    targets = generate_grasp_targets(cube_position, cube_yaw=cube_yaw)
    monitor = GraspMonitor(scene)
    final_status = monitor.update(scene)

    records.append(
        PickStageRecord(
            state=PickState.RESET,
            success=reset_success,
            reason=reset_reason,
            simulation_duration=reset_duration,
        )
    )

    def failed(stage: PickState, reason: str) -> PickResult:
        return PickResult(
            success=False,
            final_state=PickState.FAILED,
            reason=reason,
            failed_stage=stage,
            records=tuple(records),
            targets=targets,
            final_status=monitor.update(scene),
            hold_duration=0.0,
        )

    if not reset_success:
        return failed(
            PickState.RESET,
            reset_reason,
        )

    open_gripper(scene)

    # ---------- MOVE_ABOVE：移动到方块正上方 ----------
    result = _run_motion(
        scene,
        targets.pregrasp_position,
        targets.rotation,
        command_count=201,
        step_callback=step_callback,
    )
    records.append(
        PickStageRecord(PickState.MOVE_ABOVE, result.success, result.reason, result.simulation_duration)
    )
    if not result.success:
        return failed(PickState.MOVE_ABOVE, result.reason)

    # ---------- APPROACH：保持抓取姿态，下降到方块中心附近 ----------
    result = _run_motion(
        scene,
        targets.grasp_position,
        targets.rotation,
        command_count=101,
        step_callback=step_callback,
    )
    records.append(
        PickStageRecord(PickState.APPROACH, result.success, result.reason, result.simulation_duration)
    )
    if not result.success:
        return failed(PickState.APPROACH, result.reason)

    # ---------- CLOSE_GRIPPER：发闭合命令并等待形成持续双侧接触 ----------
    close_gripper(scene)
    close_start_time = float(scene.data.time)
    close_steps = math.ceil(close_timeout / timestep)
    final_status = monitor.update(scene)

    for _ in range(close_steps):
        status = _step_once(
            scene,
            monitor=monitor,
            step_callback=step_callback,
        )
        assert status is not None
        final_status = status
        if status.grasp_candidate:
            break

    close_elapsed = float(scene.data.time) - close_start_time
    records.append(
        PickStageRecord(
            PickState.CLOSE_GRIPPER,
            final_status.grasp_candidate,
            "none" if final_status.grasp_candidate else final_status.reason,
            close_elapsed,
        )
    )

    # VERIFY_CONTACT 是单独的决策状态：它不再执行动作，只读取监测结果。
    records.append(
        PickStageRecord(
            PickState.VERIFY_CONTACT,
            final_status.grasp_candidate,
            "none" if final_status.grasp_candidate else final_status.reason,
            0.0,
        )
    )
    if not final_status.grasp_candidate:
        return failed(PickState.VERIFY_CONTACT, final_status.reason)

    # ---------- LIFT：闭合命令保持不变，同时把末端抬高 ----------
    result = _run_motion(
        scene,
        targets.lift_position,
        targets.rotation,
        command_count=101,
        monitor=monitor,
        step_callback=step_callback,
    )
    final_status = monitor.update(scene)
    lift_success = result.success and final_status.cube_lifted
    lift_reason = result.reason if not result.success else final_status.reason
    if lift_success:
        lift_reason = "none"

    records.append(
        PickStageRecord(PickState.LIFT, lift_success, lift_reason, result.simulation_duration)
    )
    if not lift_success:
        return failed(PickState.LIFT, lift_reason)

    # ---------- CHECK_SUCCESS：在目标高度连续稳定保持至少 hold_duration ----------
    required_hold_steps = math.ceil(hold_duration / timestep)
    consecutive_success_steps = 0
    check_start_time = float(scene.data.time)

    # 最多给两倍目标保持时间；只有连续成功帧才累计。
    for _ in range(required_hold_steps * 2):
        status = _step_once(
            scene,
            monitor=monitor,
            step_callback=step_callback,
        )
        assert status is not None
        final_status = status

        if status.grasp_success:
            consecutive_success_steps += 1
        else:
            consecutive_success_steps = 0

        if consecutive_success_steps >= required_hold_steps:
            break

    actual_hold_duration = consecutive_success_steps * timestep
    success = consecutive_success_steps >= required_hold_steps
    reason = "none" if success else final_status.reason
    records.append(
        PickStageRecord(
            PickState.CHECK_SUCCESS,
            success,
            reason,
            float(scene.data.time) - check_start_time,
        )
    )

    if not success:
        return PickResult(
            success=False,
            final_state=PickState.FAILED,
            reason=reason,
            failed_stage=PickState.CHECK_SUCCESS,
            records=tuple(records),
            targets=targets,
            final_status=final_status,
            hold_duration=actual_hold_duration,
        )

    records.append(PickStageRecord(PickState.DONE, True, "none", 0.0))
    return PickResult(
        success=True,
        final_state=PickState.DONE,
        reason="none",
        failed_stage=None,
        records=tuple(records),
        targets=targets,
        final_status=final_status,
        hold_duration=actual_hold_duration,
    )
