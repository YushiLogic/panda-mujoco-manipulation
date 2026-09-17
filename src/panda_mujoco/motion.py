"""Panda 机械臂运动规划与执行工具。"""
from collections.abc import Callable
from panda_mujoco.ik import solve_pose_ik
import mujoco
import numpy as np
from dataclasses import dataclass
from panda_mujoco.gripper import get_gripper_width
from panda_mujoco.kinematics import (
    ARM_JOINT_NAMES,
    get_ee_pose,
)
from panda_mujoco.rotations import rotation_error
from panda_mujoco.arm_control import (
    apply_arm_bias_compensation,
    command_arm_joint_positions,
)
from panda_mujoco.simulation import PandaScene

ARM_JOINT_COUNT = 7

@dataclass
class PosePlan:
    """一次末端目标位姿规划的结果。"""

    success: bool
    reason: str
    joint_target: np.ndarray
    ik_iterations: int
    position_error_norm: float
    orientation_error_norm: float
@dataclass
class MotionResult:
    """一次末端运动原语的执行结果。"""

    success: bool
    reason: str
    simulation_duration: float
    ik_iterations: int

    target_joint_positions: np.ndarray
    final_joint_positions: np.ndarray

    final_position: np.ndarray
    final_rotation: np.ndarray

    position_error_norm: float
    orientation_error_norm: float
    maximum_joint_error: float

    joint_limit_violation: bool
    gripper_width: float
def plan_pose_target(
    control_scene: PandaScene,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    *,
    damping: float = 0.01,
    position_tolerance: float = 1e-4,
    orientation_tolerance: float = 1e-3,
    max_iterations: int = 100,
    max_joint_step: float = 0.1,
) -> PosePlan:
    """从控制场景的当前状态规划一个末端目标位姿。

    本函数创建独立的规划场景，并在规划场景中运行IK。
    它不会修改control_scene的qpos、qvel、ctrl或仿真时间。
    """

    # 创建独立规划场景。它拥有自己的MjData，
    # 因此IK修改它的qpos不会让控制场景中的机器人瞬移。
    planning_scene = PandaScene(
        control_scene.scene_path
    )

    # 让规划从控制场景当前的实际状态开始。
    # 这里复制整个qpos，其中包括机械臂、夹爪和方块。
    planning_scene.data.qpos[:] = (
        control_scene.data.qpos.copy()
    )

    # 规划场景只进行运动学计算，不需要继承实际速度。
    planning_scene.data.qvel[:] = 0.0

    # qpos被修改后，需要重新计算末端位置、旋转矩阵等派生量。
    mujoco.mj_forward(
        planning_scene.model,
        planning_scene.data,
    )

    # solve_pose_ik只会修改planning_scene，
    # 不会修改control_scene。
    ik_result = solve_pose_ik(
        planning_scene,
        target_position,
        target_rotation,
        damping=damping,
        position_tolerance=position_tolerance,
        orientation_tolerance=(
            orientation_tolerance
        ),
        max_iterations=max_iterations,
        max_joint_step=max_joint_step,
    )

    if ik_result.success:
        reason = "none"
    else:
        reason = "ik_failed"

    # IK结束后，规划场景前7个qpos就是求出的目标关节角。
    joint_target = (
        planning_scene.data.qpos[:ARM_JOINT_COUNT]
        .copy()
    )

    return PosePlan(
        success=ik_result.success,
        reason=reason,
        joint_target=joint_target,
        ik_iterations=ik_result.iterations,
        position_error_norm=(
            ik_result.final_position_error_norm
        ),
        orientation_error_norm=(
            ik_result.final_orientation_error_norm
        ),
    )

def linear_joint_trajectory(
    start_positions: np.ndarray,
    target_positions: np.ndarray,
    command_count: int,
) -> np.ndarray:
    """生成包含起点和终点的关节空间线性轨迹。

    Args:
        start_positions:
            七个机械臂关节的起始角度，形状为 (7,)，单位为弧度。

        target_positions:
            七个机械臂关节的目标角度，形状为 (7,)，单位为弧度。

        command_count:
            轨迹中包含的关节命令数量。必须至少为2，因为轨迹需要
            同时包含起点和终点。

    Returns:
        形状为 (command_count, 7) 的关节轨迹。
        每一行表示一个时刻的七个关节目标。
    """

    # 将列表等输入统一转换成浮点 NumPy 数组。
    # copy() 保证后面的操作不会修改调用者传入的原始数组。
    start_positions = np.asarray(
        start_positions,
        dtype=float,
    ).copy()

    target_positions = np.asarray(
        target_positions,
        dtype=float,
    ).copy()

    # Panda 手臂具有7个关节，因此必须恰好传入7个关节角。
    if start_positions.shape != (ARM_JOINT_COUNT,):
        raise ValueError(
            "start_positions must have shape (7,)"
        )

    if target_positions.shape != (ARM_JOINT_COUNT,):
        raise ValueError(
            "target_positions must have shape (7,)"
        )

    # NaN和Inf不能作为有效的关节角。
    if not np.all(np.isfinite(start_positions)):
        raise ValueError(
            "start_positions must contain finite values"
        )

    if not np.all(np.isfinite(target_positions)):
        raise ValueError(
            "target_positions must contain finite values"
        )

    # bool 是 int 的子类，因此需要显式排除 True 和 False。
    if (
        not isinstance(command_count, (int, np.integer))
        or isinstance(command_count, bool)
        or command_count < 2
    ):
        raise ValueError(
            "command_count must be an integer of at least 2"
        )

    # np.linspace 可以在两个同形状数组之间进行逐元素线性插值。
    # 第一行等于 start_positions，最后一行等于 target_positions。
    trajectory = np.linspace(
        start_positions,
        target_positions,
        num=int(command_count),
    )

    return trajectory

def execute_joint_trajectory(
    scene: PandaScene,
    joint_trajectory: np.ndarray,
    *,
    physics_steps_per_command: int = 10,
    settle_steps: int = 0,
    step_callback: (
        Callable[[PandaScene], None] | None
    ) = None,
) -> float:
    """通过位置执行器执行一条关节轨迹。

    Args:
        scene:
            执行动力学仿真的 Panda 场景。

        joint_trajectory:
            形状为 (N, 7) 的关节轨迹。每一行是一组完整的
            七关节位置命令。

        physics_steps_per_command:
            每一组关节命令保持多少个 MuJoCo 物理时间步。
        step_callback:
        可选的物理步回调函数。每完成一个 mj_step() 后调用一次。
        可用于刷新可视化窗口；None 表示不调用。
        settle_steps:
            轨迹命令全部发送完毕后，继续保持最后一组目标的
            物理步数。使用0表示不增加稳定阶段。
    Returns:
        执行这条轨迹所经过的仿真时间，单位为秒。
    """

    # 使用副本，防止函数内部意外修改调用者的数据。
    joint_trajectory = np.asarray(
        joint_trajectory,
        dtype=float,
    ).copy()

    # 轨迹必须是二维数组，而且每一行必须有7个关节角。
    if (
        joint_trajectory.ndim != 2
        or joint_trajectory.shape[1] != ARM_JOINT_COUNT
        or len(joint_trajectory) < 2
    ):
        raise ValueError(
            "joint_trajectory must have shape (N, 7) "
            "with N >= 2"
        )

    # 不允许把NaN或Inf写入执行器控制量。
    if not np.all(np.isfinite(joint_trajectory)):
        raise ValueError(
            "joint_trajectory must contain finite values"
        )

    # 每个命令至少需要执行一个物理步。
    if (
        not isinstance(
            physics_steps_per_command,
            (int, np.integer),
        )
        or isinstance(physics_steps_per_command, bool)
        or physics_steps_per_command < 1
    ):
        raise ValueError(
            "physics_steps_per_command must be "
            "a positive integer"
        )
    if (
        not isinstance(
            settle_steps,
            (int, np.integer),
        )
        or isinstance(settle_steps, bool)
        or settle_steps < 0
    ):
        raise ValueError(
            "settle_steps must be a non-negative integer"
        )
    start_time = float(scene.data.time)

    # 逐行读取轨迹。joint_command每次都是形状为(7,)的一行。
    for joint_command in joint_trajectory:
        # 这里只写ctrl，不会直接改变qpos。
        command_arm_joint_positions(
            scene,
            joint_command,
        )

        # 让执行器在多个物理步内跟随当前目标。
        for _ in range(physics_steps_per_command):
            # qfrc_bias会随姿态和速度变化，所以每一步都要更新。
            apply_arm_bias_compensation(scene)

            mujoco.mj_step(
                scene.model,
                scene.data,
            )
            if step_callback is not None:
                step_callback(scene)

    # ---------- 最终稳定阶段 ----------

    final_joint_command = joint_trajectory[-1]

    command_arm_joint_positions(
        scene,
        final_joint_command,
    )

    for _ in range(settle_steps):
        apply_arm_bias_compensation(scene)

        mujoco.mj_step(
            scene.model,
            scene.data,
        )

        if step_callback is not None:
            step_callback(scene)

    end_time = float(scene.data.time)

    return end_time - start_time

def _has_arm_joint_limit_violation(
    scene: PandaScene,
    *,
    tolerance: float = 1e-6,
) -> bool:
    """检查七个机械臂关节是否超出MJCF限位。"""

    for joint_name in ARM_JOINT_NAMES:
        joint_id = scene.model.joint(
            joint_name
        ).id

        qpos_address = (
            scene.model.jnt_qposadr[joint_id]
        )

        joint_position = float(
            scene.data.qpos[qpos_address]
        )

        lower_limit, upper_limit = (
            scene.model.jnt_range[joint_id]
        )

        if (
            joint_position < lower_limit - tolerance
            or joint_position > upper_limit + tolerance
        ):
            return True

    return False

def move_end_effector_to(
    control_scene: PandaScene,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    *,
    command_count: int = 201,
    physics_steps_per_command: int = 10,
    settle_steps: int = 250,
    position_tolerance: float = 0.002,
    orientation_tolerance: float = np.deg2rad(1.0),
    step_callback: (
        Callable[[PandaScene], None] | None
    ) = None,
) -> MotionResult:
    """规划并执行一次末端位姿运动。

    默认执行验收标准：
    - 位置误差不超过2毫米；
    - 姿态误差不超过1度；
    - 状态中不出现NaN或Inf；
    - 机械臂关节不超限。
    """

    target_position = np.asarray(
        target_position,
        dtype=float,
    ).copy()

    target_rotation = np.asarray(
        target_rotation,
        dtype=float,
    ).copy()

    if (
        not np.isfinite(position_tolerance)
        or position_tolerance <= 0.0
    ):
        raise ValueError(
            "position_tolerance must be positive and finite"
        )

    if (
        not np.isfinite(orientation_tolerance)
        or orientation_tolerance <= 0.0
    ):
        raise ValueError(
            "orientation_tolerance must be "
            "positive and finite"
        )

    # ---------- 第一阶段：独立场景中的IK规划 ----------

    plan = plan_pose_target(
        control_scene,
        target_position,
        target_rotation,
    )

    # 如果IK失败，就不能继续执行关节轨迹。
    if not plan.success:
        final_position, final_rotation = (
            get_ee_pose(control_scene)
        )

        return MotionResult(
            success=False,
            reason=plan.reason,
            simulation_duration=0.0,
            ik_iterations=plan.ik_iterations,
            target_joint_positions=(
                plan.joint_target.copy()
            ),
            final_joint_positions=(
                control_scene.data.qpos[:7].copy()
            ),
            final_position=final_position,
            final_rotation=final_rotation,
            position_error_norm=float(
                np.linalg.norm(
                    target_position - final_position
                )
            ),
            orientation_error_norm=float(
                np.linalg.norm(
                    rotation_error(
                        target_rotation,
                        final_rotation,
                    )
                )
            ),
            maximum_joint_error=float("inf"),
            joint_limit_violation=(
                _has_arm_joint_limit_violation(
                    control_scene
                )
            ),
            gripper_width=get_gripper_width(
                control_scene
            ),
        )

    # ---------- 第二阶段：生成关节空间轨迹 ----------

    start_joint_positions = (
        control_scene.data.qpos[:7].copy()
    )

    joint_trajectory = linear_joint_trajectory(
        start_joint_positions,
        plan.joint_target,
        command_count=command_count,
    )

    # ---------- 第三阶段：动力学执行 ----------

    simulation_duration = (
        execute_joint_trajectory(
            control_scene,
            joint_trajectory,
            physics_steps_per_command=(
                physics_steps_per_command
            ),
            settle_steps=settle_steps,
            step_callback=step_callback,
        )
    )

    # ---------- 第四阶段：读取真实执行结果 ----------

    final_position, final_rotation = get_ee_pose(
        control_scene
    )

    final_joint_positions = (
        control_scene.data.qpos[:7].copy()
    )

    position_error_norm = float(
        np.linalg.norm(
            target_position - final_position
        )
    )

    orientation_error_norm = float(
        np.linalg.norm(
            rotation_error(
                target_rotation,
                final_rotation,
            )
        )
    )

    maximum_joint_error = float(
        np.max(
            np.abs(
                plan.joint_target
                - final_joint_positions
            )
        )
    )

    joint_limit_violation = (
        _has_arm_joint_limit_violation(
            control_scene
        )
    )

    state_is_finite = bool(
        np.all(np.isfinite(control_scene.data.qpos))
        and np.all(
            np.isfinite(control_scene.data.qvel)
        )
        and np.all(
            np.isfinite(control_scene.data.ctrl)
        )
        and np.isfinite(position_error_norm)
        and np.isfinite(orientation_error_norm)
    )

    # ---------- 第五阶段：确定成功状态和原因 ----------

    if not state_is_finite:
        success = False
        reason = "non_finite_state"

    elif joint_limit_violation:
        success = False
        reason = "joint_limit_violation"

    elif position_error_norm > position_tolerance:
        success = False
        reason = "position_tolerance"

    elif orientation_error_norm > orientation_tolerance:
        success = False
        reason = "orientation_tolerance"

    else:
        success = True
        reason = "none"

    return MotionResult(
        success=success,
        reason=reason,
        simulation_duration=simulation_duration,
        ik_iterations=plan.ik_iterations,
        target_joint_positions=(
            plan.joint_target.copy()
        ),
        final_joint_positions=(
            final_joint_positions
        ),
        final_position=final_position,
        final_rotation=final_rotation,
        position_error_norm=position_error_norm,
        orientation_error_norm=(
            orientation_error_norm
        ),
        maximum_joint_error=maximum_joint_error,
        joint_limit_violation=(
            joint_limit_violation
        ),
        gripper_width=get_gripper_width(
            control_scene
        ),
    )
