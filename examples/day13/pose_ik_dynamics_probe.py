"""通过 MuJoCo 动力学执行 6D IK 求出的关节目标。"""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from panda_mujoco.arm_control import (
    apply_arm_bias_compensation,
    command_arm_joint_positions,
)
from panda_mujoco.ik import solve_pose_ik
from panda_mujoco.joint_trajectory import JointPath
from panda_mujoco.kinematics import get_ee_pose
from panda_mujoco.rotations import rotation_error
from panda_mujoco.simulation import PandaScene


TARGET_ANGLE_DEGREES = 15.0

# MuJoCo 物理步长为 0.002 s，每10步更新一次关节目标。
# 因此控制频率为 50 Hz。
CONTROL_EVERY = 10

# 200个控制拍，每拍0.02 s，总运动时间为4 s。
MOVE_TICKS = 200

# 最终目标保持500个物理步，即1 s。
SETTLE_STEPS = 500


def step_with_bias_compensation(
    scene: PandaScene,
) -> None:
    """执行一个带理想偏置力补偿的物理步。"""

    apply_arm_bias_compensation(scene)
    mujoco.mj_step(scene.model, scene.data)


def main() -> None:
    # planning_scene 只负责运动学求解。
    # control_scene 负责执行器和动力学仿真。
    planning_scene = PandaScene()
    control_scene = PandaScene()

    initial_position, initial_rotation = get_ee_pose(
        planning_scene
    )

    target_position = initial_position.copy()
    target_rotation = (
        Rotation.from_euler(
            "z",
            TARGET_ANGLE_DEGREES,
            degrees=True,
        ).as_matrix()
        @ initial_rotation
    )

    # ---------- 第一阶段：运动学规划 ----------

    ik_result = solve_pose_ik(
        planning_scene,
        target_position,
        target_rotation,
        damping=0.01,
        position_tolerance=1e-4,
        orientation_tolerance=1e-3,
        max_iterations=100,
        max_joint_step=0.1,
    )

    assert ik_result.success

    target_joint_positions = (
        planning_scene.data.qpos[:7].copy()
    )

    # ---------- 第二阶段：生成平滑关节轨迹 ----------

    initial_joint_positions = (
        control_scene.data.qpos[:7].copy()
    )

    path = JointPath(
        goals=[
            initial_joint_positions,
            target_joint_positions,
        ],
        ticks_per_move=MOVE_TICKS,
    )

    # ---------- 第三阶段：动力学执行 ----------

    while not path.finished():
        command_arm_joint_positions(
            control_scene,
            path.target(),
        )

        for _ in range(CONTROL_EVERY):
            step_with_bias_compensation(control_scene)

        path.advance()

    # 明确写入最终目标，并保持1秒等待稳态。
    command_arm_joint_positions(
        control_scene,
        target_joint_positions,
    )

    for _ in range(SETTLE_STEPS):
        step_with_bias_compensation(control_scene)

    # ---------- 第四阶段：测量实际位姿 ----------

    actual_position, actual_rotation = get_ee_pose(
        control_scene
    )

    actual_joint_positions = (
        control_scene.data.qpos[:7].copy()
    )

    joint_error = (
        target_joint_positions - actual_joint_positions
    )
    position_error = target_position - actual_position
    orientation_error = rotation_error(
        target_rotation,
        actual_rotation,
    )

    maximum_joint_error = float(
        np.max(np.abs(joint_error))
    )
    position_error_norm = float(
        np.linalg.norm(position_error)
    )
    orientation_error_norm = float(
        np.linalg.norm(orientation_error)
    )
    joint_velocity_norm = float(
        np.linalg.norm(control_scene.data.qvel[:7])
    )

    np.set_printoptions(precision=9, suppress=True)

    print("IK target joint positions:")
    print(target_joint_positions)

    print()
    print("Actual joint positions:")
    print(actual_joint_positions)

    print()
    print("Target end-effector position:")
    print(target_position)

    print()
    print("Actual end-effector position:")
    print(actual_position)

    print()
    print("Position error:")
    print(position_error)

    print()
    print("Orientation error vector:")
    print(orientation_error)

    print()
    print(f"IK updates:              {ik_result.iterations}")
    print(f"Simulation time:         {control_scene.data.time:.3f} s")
    print(f"Maximum joint error:     {maximum_joint_error:.6e} rad")
    print(f"Position error:          {position_error_norm:.6e} m")
    print(
        "Orientation error:       "
        f"{np.rad2deg(orientation_error_norm):.6e} degrees"
    )
    print(f"Joint velocity norm:     {joint_velocity_norm:.6e} rad/s")

    assert maximum_joint_error < 1e-5
    assert position_error_norm < 1e-4
    assert orientation_error_norm < 1e-3
    assert joint_velocity_norm < 1e-3

    print()
    print("6D IK tracking through dynamics: PASSED")


if __name__ == "__main__":
    main()
