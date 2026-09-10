"""通过动力学执行连续笛卡尔路点，并记录末端轨迹误差。"""

import mujoco
import numpy as np

from panda_mujoco.arm_control import (
    apply_arm_bias_compensation,
    command_arm_joint_positions,
)
from panda_mujoco.cartesian_trajectory import (
    linear_position_waypoints,
    plan_position_waypoints,
)
from panda_mujoco.kinematics import get_ee_position
from panda_mujoco.simulation import PROJECT_ROOT, PandaScene


TARGET_OFFSET = np.array(
    [0.020, -0.010, 0.015]
)

WAYPOINT_COUNT = 6

# 每两个笛卡尔路点之间使用20个控制拍。
TICKS_PER_SEGMENT = 20

# 物理频率500 Hz，每10步更新一次控制目标，即50 Hz。
CONTROL_EVERY = 10

# 完成轨迹后保持最终目标1秒。
SETTLE_STEPS = 500

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "day12"
DATA_PATH = OUTPUT_DIR / "cartesian_trajectory_data.npz"


def compensated_step(
    scene: PandaScene,
) -> None:
    """执行一次带理想偏置力补偿的物理步。"""

    apply_arm_bias_compensation(scene)

    mujoco.mj_step(
        scene.model,
        scene.data,
    )


def main() -> None:
    planning_scene = PandaScene()
    control_scene = PandaScene()

    # 纯运动学参考场景：用于计算“如果关节完全等于命令值，
    # 末端应该在哪里”。它不执行动力学。
    reference_scene = PandaScene()

    start_position = get_ee_position(
        control_scene
    )

    target_position = (
        start_position + TARGET_OFFSET
    )

    cartesian_waypoints = linear_position_waypoints(
        start_position,
        target_position,
        waypoint_count=WAYPOINT_COUNT,
    )

    waypoint_plan = plan_position_waypoints(
        planning_scene,
        cartesian_waypoints,
        damping=0.05,
        tolerance=1e-4,
        max_iterations=50,
        max_joint_step=0.1,
    )

    joint_targets = waypoint_plan.joint_targets

    time_log = []
    desired_position_log = []
    reference_position_log = []
    actual_position_log = []

    # ---------- 执行5段连续轨迹 ----------

    for segment_index in range(
        WAYPOINT_COUNT - 1
    ):
        joint_start = joint_targets[
            segment_index
        ]
        joint_end = joint_targets[
            segment_index + 1
        ]

        position_start = cartesian_waypoints[
            segment_index
        ]
        position_end = cartesian_waypoints[
            segment_index + 1
        ]

        for tick in range(
            1,
            TICKS_PER_SEGMENT + 1,
        ):
            interpolation = (
                tick / TICKS_PER_SEGMENT
            )

            # 关节命令在相邻IK解之间线性插值。
            joint_command = (
                joint_start
                + interpolation
                * (joint_end - joint_start)
            )

            # 同一时刻，理想的笛卡尔目标位于直线上。
            desired_position = (
                position_start
                + interpolation
                * (position_end - position_start)
            )

            # 在独立参考场景中直接设置关节命令，只计算正运动学。
            # 这一步用于分离“关节插值的几何偏差”和“动力学滞后”。
            reference_scene.data.qpos[:7] = (
                joint_command
            )

            mujoco.mj_forward(
                reference_scene.model,
                reference_scene.data,
            )

            reference_position = get_ee_position(
                reference_scene
            )

            command_arm_joint_positions(
                control_scene,
                joint_command,
            )

            # 一个50 Hz控制拍包含10个500 Hz物理步。
            for _ in range(CONTROL_EVERY):
                compensated_step(control_scene)

            actual_position = get_ee_position(
                control_scene
            )

            time_log.append(
                control_scene.data.time
            )
            desired_position_log.append(
                desired_position.copy()
            )
            reference_position_log.append(
                reference_position
            )
            actual_position_log.append(
                actual_position
            )

    time_log = np.asarray(time_log)

    desired_position_log = np.stack(
        desired_position_log,
        axis=0,
    )

    reference_position_log = np.stack(
        reference_position_log,
        axis=0,
    )

    actual_position_log = np.stack(
        actual_position_log,
        axis=0,
    )

    # 几何误差：关节空间插值产生的运动学轨迹与理想直线之差。
    geometric_error_vectors = (
        desired_position_log
        - reference_position_log
    )

    # 动力学误差：实际末端没有完全跟上关节命令造成的误差。
    dynamic_error_vectors = (
        reference_position_log
        - actual_position_log
    )

    # 总误差：理想直线目标与实际末端位置之差。
    total_error_vectors = (
        desired_position_log
        - actual_position_log
    )

    geometric_error_norms = np.linalg.norm(
        geometric_error_vectors,
        axis=1,
    )

    dynamic_error_norms = np.linalg.norm(
        dynamic_error_vectors,
        axis=1,
    )

    total_error_norms = np.linalg.norm(
        total_error_vectors,
        axis=1,
    )

    maximum_tracking_error = float(
        np.max(total_error_norms)
    )

    mean_tracking_error = float(
        np.mean(total_error_norms)
    )

    end_of_motion_error = float(
        total_error_norms[-1]
    )

    maximum_geometric_error = float(
        np.max(geometric_error_norms)
    )

    mean_geometric_error = float(
        np.mean(geometric_error_norms)
    )

    maximum_dynamic_error = float(
        np.max(dynamic_error_norms)
    )

    mean_dynamic_error = float(
        np.mean(dynamic_error_norms)
    )

    # ---------- 最终目标保持1秒 ----------

    command_arm_joint_positions(
        control_scene,
        joint_targets[-1],
    )

    for _ in range(SETTLE_STEPS):
        compensated_step(control_scene)

    settled_position = get_ee_position(
        control_scene
    )

    settled_final_error = float(
        np.linalg.norm(
            target_position - settled_position
        )
    )

    final_joint_error = float(
        np.max(
            np.abs(
                joint_targets[-1]
                - control_scene.data.qpos[:7]
            )
        )
    )

    # 保存绘图所需的结构化数据。NPZ 能保留数组形状和浮点精度，
    # 绘图脚本无需重新运行仿真，也不需要从终端文本中解析数字。
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        DATA_PATH,
        time=time_log,
        start_position=start_position,
        target_position=target_position,
        cartesian_waypoints=cartesian_waypoints,
        joint_targets=joint_targets,
        desired_position=desired_position_log,
        reference_position=reference_position_log,
        actual_position=actual_position_log,
        geometric_error_norm=geometric_error_norms,
        dynamic_error_norm=dynamic_error_norms,
        total_error_norm=total_error_norms,
        settled_position=settled_position,
        settled_final_error=settled_final_error,
    )

    np.set_printoptions(
        precision=9,
        suppress=True,
    )

    print("Cartesian trajectory tracking")
    print()

    # 每20个控制拍正好完成一段，打印各路点的实际结果。
    for segment_index in range(
        WAYPOINT_COUNT - 1
    ):
        sample_index = (
            (segment_index + 1)
            * TICKS_PER_SEGMENT
            - 1
        )

        print(
            f"waypoint {segment_index + 1} | "
            f"time={time_log[sample_index]:.3f} s | "
            f"desired="
            f"{desired_position_log[sample_index]} | "
            f"reference="
            f"{reference_position_log[sample_index]} | "
            f"actual="
            f"{actual_position_log[sample_index]} | "
            f"geometric="
            f"{geometric_error_norms[sample_index]:.6e} m | "
            f"dynamic="
            f"{dynamic_error_norms[sample_index]:.6e} m | "
            f"total="
            f"{total_error_norms[sample_index]:.6e} m"
        )

    print()
    print(
        "Maximum geometric error:      "
        f"{maximum_geometric_error:.6e} m"
    )
    print(
        "Mean geometric error:         "
        f"{mean_geometric_error:.6e} m"
    )
    print(
        "Maximum dynamic error:        "
        f"{maximum_dynamic_error:.6e} m"
    )
    print(
        "Mean dynamic error:           "
        f"{mean_dynamic_error:.6e} m"
    )
    print(
        "Maximum moving tracking error: "
        f"{maximum_tracking_error:.6e} m"
    )
    print(
        "Mean moving tracking error:    "
        f"{mean_tracking_error:.6e} m"
    )
    print(
        "End-of-motion error:           "
        f"{end_of_motion_error:.6e} m"
    )
    print(
        "Settled final error:           "
        f"{settled_final_error:.6e} m"
    )
    print(
        "Final maximum joint error:     "
        f"{final_joint_error:.6e} rad"
    )
    print(
        "Total simulation time:         "
        f"{control_scene.data.time:.3f} s"
    )
    print(f"Trajectory data saved:         {DATA_PATH}")

    assert time_log.shape == (
        (WAYPOINT_COUNT - 1)
        * TICKS_PER_SEGMENT,
    )

    assert desired_position_log.shape == (
        len(time_log),
        3,
    )

    assert reference_position_log.shape == (
        len(time_log),
        3,
    )

    assert actual_position_log.shape == (
        len(time_log),
        3,
    )

    assert np.all(
        np.isfinite(actual_position_log)
    )

    # 误差向量应满足：总误差 = 几何误差 + 动力学误差。
    np.testing.assert_allclose(
        total_error_vectors,
        geometric_error_vectors
        + dynamic_error_vectors,
        atol=1e-12,
    )

    assert maximum_geometric_error < 1e-4
    assert maximum_dynamic_error < 0.002

    # 运动过程中允许存在约1 mm的动态跟踪滞后。
    assert maximum_tracking_error < 0.002

    # 稳定后应重新接近IK误差范围。
    assert settled_final_error < 1e-4
    assert final_joint_error < 1e-5

    print()
    print("Cartesian trajectory tracking: PASSED")


if __name__ == "__main__":
    main()
