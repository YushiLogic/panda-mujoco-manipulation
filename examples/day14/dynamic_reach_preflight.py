"""用前 5 个固定目标预检 IK 结果能否通过动力学真正执行。

这个脚本故意把一次任务拆成两个场景：

1. planning_scene：只负责运行位置 IK，求出目标关节角；
2. control_scene：从 home 出发，通过执行器和 MuJoCo 动力学跟踪目标。

这样可以避免把“直接改 qpos 得到的数学结果”误认为“机器人已经运动到位”。
"""

from dataclasses import dataclass
from time import perf_counter

import mujoco
import numpy as np

from panda_mujoco.arm_control import (
    apply_arm_bias_compensation,
    command_arm_joint_positions,
)
from panda_mujoco.ik import solve_position_ik
from panda_mujoco.joint_trajectory import JointPath
from panda_mujoco.kinematics import (
    ARM_JOINT_NAMES,
    get_ee_position,
)
from panda_mujoco.simulation import PandaScene

from target_sampling_probe import (
    RANDOM_SEED,
    TARGET_COUNT,
    generate_target_positions,
)


PREFLIGHT_COUNT = 5

# 当前模型的物理步长为 0.002 s，即 500 Hz。
# 每 10 个物理步更新一次关节目标，相当于 50 Hz 控制频率。
CONTROL_EVERY = 10

# 100 个控制拍 × 0.02 s = 2 s 运动时间。
MOVE_TICKS = 100

# 到达最终目标后继续保持 500 个物理步，即 1 s。
SETTLE_STEPS = 500

# Day14 的批量动力学验收标准：末端误差不超过 2 cm。
TRACKING_TOLERANCE_M = 0.02


@dataclass
class DynamicCaseResult:
    """保存一个目标点从 IK 到动力学执行结束的评测结果。"""

    case_id: int
    target_position: np.ndarray
    ik_success: bool
    success: bool
    failure_reason: str
    ik_iterations: int
    ik_error_m: float
    tracking_error_m: float
    max_joint_error_rad: float
    joint_velocity_norm_rad_s: float
    all_finite: bool
    joint_limit_violation: bool
    simulation_time_s: float
    wall_time_s: float


def get_arm_joint_metadata(
    scene: PandaScene,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回手臂关节的 qpos 地址、下限和上限。"""

    joint_ids = np.array(
        [
            scene.model.joint(name).id
            for name in ARM_JOINT_NAMES
        ],
        dtype=int,
    )

    qpos_addresses = np.array(
        [
            scene.model.jnt_qposadr[joint_id]
            for joint_id in joint_ids
        ],
        dtype=int,
    )

    lower_limits = scene.model.jnt_range[
        joint_ids,
        0,
    ].copy()

    upper_limits = scene.model.jnt_range[
        joint_ids,
        1,
    ].copy()

    return qpos_addresses, lower_limits, upper_limits


def state_is_finite(scene: PandaScene) -> bool:
    """检查主要仿真数组中是否出现 NaN 或 Inf。"""

    return bool(
        np.all(np.isfinite(scene.data.qpos))
        and np.all(np.isfinite(scene.data.qvel))
        and np.all(np.isfinite(scene.data.ctrl))
        and np.all(np.isfinite(scene.data.qfrc_applied))
    )


def arm_violates_limits(
    scene: PandaScene,
    qpos_addresses: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
) -> bool:
    """检查当前七个手臂关节是否越过 MJCF 限位。"""

    joint_positions = scene.data.qpos[
        qpos_addresses
    ]

    # 给浮点计算和动力学求解器留出极小的数值容差。
    tolerance = 1e-6

    return bool(
        np.any(
            joint_positions
            < lower_limits - tolerance
        )
        or np.any(
            joint_positions
            > upper_limits + tolerance
        )
    )


def step_dynamics(scene: PandaScene) -> None:
    """执行一次带偏置力补偿的物理仿真步。"""

    # qfrc_bias 会随 qpos 和 qvel 变化，因此必须每一步更新。
    apply_arm_bias_compensation(scene)
    mujoco.mj_step(scene.model, scene.data)


def evaluate_target(
    case_id: int,
    target_position: np.ndarray,
) -> DynamicCaseResult:
    """对一个目标执行 IK 规划、关节轨迹和动力学跟踪。"""

    wall_start = perf_counter()

    # planning_scene 只用于计算 IK，不推进仿真时间。
    planning_scene = PandaScene()

    ik_result = solve_position_ik(
        planning_scene,
        target_position,
        damping=0.05,
        tolerance=1e-4,
        max_iterations=100,
        max_joint_step=0.1,
    )

    planning_addresses, _, _ = get_arm_joint_metadata(
        planning_scene
    )

    target_joint_positions = planning_scene.data.qpos[
        planning_addresses
    ].copy()

    # control_scene 是真正执行运动的机器人，始终从 home 开始。
    control_scene = PandaScene()

    (
        control_addresses,
        lower_limits,
        upper_limits,
    ) = get_arm_joint_metadata(control_scene)

    initial_joint_positions = control_scene.data.qpos[
        control_addresses
    ].copy()

    all_finite = bool(
        state_is_finite(planning_scene)
        and np.all(np.isfinite(target_joint_positions))
        and np.isfinite(ik_result.final_error_norm)
    )

    joint_limit_violation = arm_violates_limits(
        planning_scene,
        planning_addresses,
        lower_limits,
        upper_limits,
    )

    # IK 失败时不应该盲目让机器人执行一个无效目标。
    if ik_result.success and all_finite and not joint_limit_violation:
        path = JointPath(
            goals=[
                initial_joint_positions,
                target_joint_positions,
            ],
            ticks_per_move=MOVE_TICKS,
        )

        # 用 2 秒的插值轨迹平滑移动到目标关节角。
        while path.finished() is False:
            command_arm_joint_positions(
                control_scene,
                path.target(),
            )

            for _ in range(CONTROL_EVERY):
                step_dynamics(control_scene)

                all_finite = bool(
                    all_finite
                    and state_is_finite(control_scene)
                )

                joint_limit_violation = bool(
                    joint_limit_violation
                    or arm_violates_limits(
                        control_scene,
                        control_addresses,
                        lower_limits,
                        upper_limits,
                    )
                )

            path.advance()

        # 明确发送最终目标，再保持 1 秒等待系统稳定。
        command_arm_joint_positions(
            control_scene,
            target_joint_positions,
        )

        for _ in range(SETTLE_STEPS):
            step_dynamics(control_scene)

            all_finite = bool(
                all_finite
                and state_is_finite(control_scene)
            )

            joint_limit_violation = bool(
                joint_limit_violation
                or arm_violates_limits(
                    control_scene,
                    control_addresses,
                    lower_limits,
                    upper_limits,
                )
            )

    actual_joint_positions = control_scene.data.qpos[
        control_addresses
    ].copy()

    actual_position = get_ee_position(control_scene)

    tracking_error_m = float(
        np.linalg.norm(
            target_position - actual_position
        )
    )

    max_joint_error_rad = float(
        np.max(
            np.abs(
                target_joint_positions
                - actual_joint_positions
            )
        )
    )

    joint_velocity_norm_rad_s = float(
        np.linalg.norm(
            control_scene.data.qvel[
                control_addresses
            ]
        )
    )

    all_finite = bool(
        all_finite
        and state_is_finite(control_scene)
        and np.isfinite(tracking_error_m)
        and np.isfinite(max_joint_error_rad)
        and np.isfinite(joint_velocity_norm_rad_s)
    )

    success = bool(
        ik_result.success
        and all_finite
        and not joint_limit_violation
        and tracking_error_m
        <= TRACKING_TOLERANCE_M
    )

    if not all_finite:
        failure_reason = "non_finite_state"
    elif joint_limit_violation:
        failure_reason = "joint_limit_violation"
    elif not ik_result.success:
        failure_reason = "ik_not_converged"
    elif tracking_error_m > TRACKING_TOLERANCE_M:
        failure_reason = "tracking_error_too_large"
    else:
        failure_reason = "none"

    wall_time_s = perf_counter() - wall_start

    return DynamicCaseResult(
        case_id=case_id,
        target_position=target_position.copy(),
        ik_success=ik_result.success,
        success=success,
        failure_reason=failure_reason,
        ik_iterations=ik_result.iterations,
        ik_error_m=ik_result.final_error_norm,
        tracking_error_m=tracking_error_m,
        max_joint_error_rad=max_joint_error_rad,
        joint_velocity_norm_rad_s=(
            joint_velocity_norm_rad_s
        ),
        all_finite=all_finite,
        joint_limit_violation=joint_limit_violation,
        simulation_time_s=float(control_scene.data.time),
        wall_time_s=wall_time_s,
    )


def main() -> None:
    home_scene = PandaScene()
    home_position = get_ee_position(home_scene)

    targets = generate_target_positions(
        home_position,
        target_count=TARGET_COUNT,
        seed=RANDOM_SEED,
    )

    results = []

    print("Day 14 dynamic reach preflight")
    print()

    for case_id, target_position in enumerate(
        targets[:PREFLIGHT_COUNT]
    ):
        result = evaluate_target(
            case_id,
            target_position,
        )

        results.append(result)

        print(
            f"case {result.case_id:02d}"
            f" | success={result.success}"
            f" | IK iterations={result.ik_iterations:2d}"
            f" | IK error={result.ik_error_m * 1000:.4f} mm"
            f" | tracking error="
            f"{result.tracking_error_m * 1000:.4f} mm"
            f" | sim={result.simulation_time_s:.3f} s"
            f" | wall={result.wall_time_s:.4f} s"
            f" | reason={result.failure_reason}"
        )

    success_count = sum(
        result.success
        for result in results
    )

    maximum_tracking_error_m = max(
        result.tracking_error_m
        for result in results
    )

    print()
    print("Dynamic preflight summary")
    print(
        f"success count: "
        f"{success_count}/{PREFLIGHT_COUNT}"
    )
    print(
        f"maximum tracking error: "
        f"{maximum_tracking_error_m * 1000:.6f} mm"
    )

    assert success_count == PREFLIGHT_COUNT
    assert all(result.all_finite for result in results)
    assert not any(
        result.joint_limit_violation
        for result in results
    )

    print()
    print("Dynamic reach preflight: PASSED")


if __name__ == "__main__":
    main()
