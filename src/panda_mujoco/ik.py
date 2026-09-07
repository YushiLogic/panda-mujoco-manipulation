"""Panda 数值逆运动学工具。"""

from dataclasses import dataclass

import mujoco
import numpy as np

from panda_mujoco.kinematics import (
    ARM_JOINT_NAMES,
    get_ee_jacobian,
    get_ee_position,
)
from panda_mujoco.simulation import PandaScene


@dataclass
class PositionIKResult:
    """一次位置逆运动学求解的结果。"""

    success: bool
    iterations: int
    final_position: np.ndarray
    final_error: np.ndarray
    final_error_norm: float


def _get_arm_joint_metadata(
    scene: PandaScene,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """取得手臂关节的 qpos 地址和上下限。"""

    model = scene.model

    joint_ids = np.array(
        [
            model.joint(joint_name).id
            for joint_name in ARM_JOINT_NAMES
        ],
        dtype=int,
    )

    qpos_addresses = np.array(
        [
            model.jnt_qposadr[joint_id]
            for joint_id in joint_ids
        ],
        dtype=int,
    )

    lower_limits = model.jnt_range[joint_ids, 0].copy()
    upper_limits = model.jnt_range[joint_ids, 1].copy()

    return qpos_addresses, lower_limits, upper_limits


def damped_least_squares(
    jacobian: np.ndarray,
    task_error: np.ndarray,
    damping: float = 0.05,
) -> np.ndarray:
    """计算一次阻尼最小二乘关节更新。

    Args:
        jacobian:
            当前姿态下的任务 Jacobian，形状为 (m, n)。

        task_error:
            希望减小的任务误差，形状为 (m,)。

        damping:
            正的阻尼系数。阻尼越大，关节更新越保守。

    Returns:
        关节更新量，形状为 (n,)。
    """

    # 转为浮点 NumPy 数组，让函数也能接受普通列表。
    jacobian = np.asarray(jacobian, dtype=float)
    task_error = np.asarray(task_error, dtype=float)

    if jacobian.ndim != 2:
        raise ValueError("jacobian must be a 2D array")

    task_dimension = jacobian.shape[0]

    if task_error.shape != (task_dimension,):
        raise ValueError(
            "task_error shape must match "
            "the number of Jacobian rows"
        )

    if not np.all(np.isfinite(jacobian)):
        raise ValueError("jacobian must contain finite values")

    if not np.all(np.isfinite(task_error)):
        raise ValueError("task_error must contain finite values")

    if not np.isfinite(damping) or damping <= 0.0:
        raise ValueError("damping must be positive and finite")

    identity = np.eye(task_dimension)

    regularized_matrix = (
        jacobian @ jacobian.T
        + damping**2 * identity
    )

    # 解下面的线性方程：
    #
    # (J J.T + lambda² I) y = task_error
    #
    # 使用 solve 比显式计算矩阵逆更稳定。
    task_solution = np.linalg.solve(
        regularized_matrix,
        task_error,
    )

    # delta_q = J.T y
    delta_q = jacobian.T @ task_solution

    return delta_q


def solve_position_ik(
    scene: PandaScene,
    target_position: np.ndarray,
    *,
    damping: float = 0.05,
    tolerance: float = 1e-4,
    max_iterations: int = 50,
    max_joint_step: float = 0.1,
) -> PositionIKResult:
    """使用迭代 DLS 求解 Panda 的末端目标位置。

    本函数会直接修改 scene.data.qpos，然后调用 mj_forward
    更新正运动学，但不会推进仿真时间。

    Args:
        scene:
            需要进行 IK 求解的 Panda 场景。

        target_position:
            末端在 world frame 下的目标位置，形状为 (3,)，单位为米。

        damping:
            DLS 阻尼系数。

        tolerance:
            成功判定的位置误差，单位为米。

        max_iterations:
            允许执行的最大关节更新次数。

        max_joint_step:
            每次迭代中单个关节允许变化的最大值，单位为弧度。

    Returns:
        包含成功状态、迭代次数和最终误差的 PositionIKResult。
    """

    target_position = np.asarray(
        target_position,
        dtype=float,
    ).copy()

    # ---------- 输入检查 ----------

    if target_position.shape != (3,):
        raise ValueError(
            "target_position must have shape (3,)"
        )

    if not np.all(np.isfinite(target_position)):
        raise ValueError(
            "target_position must contain finite values"
        )

    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(
            "tolerance must be positive and finite"
        )

    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError(
            "max_iterations must be a positive integer"
        )

    if (
        not np.isfinite(max_joint_step)
        or max_joint_step <= 0.0
    ):
        raise ValueError(
            "max_joint_step must be positive and finite"
        )

    # damped_least_squares() 只有在需要更新时才会被调用，
    # 因此这里提前检查 damping。
    if not np.isfinite(damping) or damping <= 0.0:
        raise ValueError(
            "damping must be positive and finite"
        )

    qpos_addresses, lower_limits, upper_limits = (
        _get_arm_joint_metadata(scene)
    )

    updates_used = 0

    # ---------- 迭代 IK 主循环 ----------

    for _ in range(max_iterations):
        current_position = get_ee_position(scene)

        position_error = (
            target_position - current_position
        )
        error_norm = float(
            np.linalg.norm(position_error)
        )

        # 先检查是否已经到达目标。
        if error_norm < tolerance:
            break

        # Jacobian 依赖当前姿态，因此每一轮都必须重新计算。
        linear_jacobian, _ = get_ee_jacobian(scene)

        raw_delta_q = damped_least_squares(
            linear_jacobian,
            position_error,
            damping=damping,
        )

        # 限制一次迭代的最大关节变化。
        delta_q = np.clip(
            raw_delta_q,
            -max_joint_step,
            max_joint_step,
        )

        current_joint_positions = (
            scene.data.qpos[qpos_addresses].copy()
        )

        candidate_joint_positions = (
            current_joint_positions + delta_q
        )

        # 候选关节角不能越过 MJCF 中定义的关节限位。
        limited_joint_positions = np.clip(
            candidate_joint_positions,
            lower_limits,
            upper_limits,
        )

        scene.data.qpos[qpos_addresses] = (
            limited_joint_positions
        )

        # qpos 改变后重新计算末端位置和其他派生量。
        mujoco.mj_forward(
            scene.model,
            scene.data,
        )

        updates_used += 1

    # ---------- 汇总最终结果 ----------

    final_position = get_ee_position(scene)
    final_error = target_position - final_position
    final_error_norm = float(
        np.linalg.norm(final_error)
    )

    success = final_error_norm < tolerance

    return PositionIKResult(
        success=success,
        iterations=updates_used,
        final_position=final_position,
        final_error=final_error,
        final_error_norm=final_error_norm,
    )
