"""让 Panda 通过执行器跟踪 IK 求出的关节目标。"""

import mujoco
import numpy as np

from panda_mujoco.ik import solve_position_ik
from panda_mujoco.joint_trajectory import JointPath
from panda_mujoco.kinematics import get_ee_position
from panda_mujoco.simulation import PandaScene
from panda_mujoco.arm_control import (
    apply_arm_bias_compensation,
    command_arm_joint_positions,
)


TARGET_OFFSET = np.array(
    [0.020, -0.010, 0.015]
)

# 500 Hz 物理仿真，每10步更新一次控制目标，即50 Hz控制频率。
CONTROL_EVERY = 10

# 100个控制拍，每拍0.02秒，所以运动持续2秒。
MOVE_TICKS = 100

# 最终目标保持500个物理步，即1秒。
SETTLE_STEPS = 500

def step_with_bias_compensation(
    scene: PandaScene,
) -> None:
    """补偿 MuJoCo 当前计算出的关节偏置力。"""

    # qfrc_bias 包括重力、科里奥利力和离心力等偏置项。
    # 将它作为外部广义力施加，相当于使用理想模型进行补偿。
    apply_arm_bias_compensation(scene)

    mujoco.mj_step(
        scene.model,
        scene.data,
    )
def main() -> None:
    # ---------- 第一阶段：计算目标关节角 ----------

    planning_scene = PandaScene()
    control_scene = PandaScene()

    initial_position = get_ee_position(control_scene)
    target_position = initial_position + TARGET_OFFSET

    ik_result = solve_position_ik(
        planning_scene,
        target_position,
        damping=0.05,
        tolerance=1e-4,
        max_iterations=50,
        max_joint_step=0.1,
    )

    assert ik_result.success

    target_joint_positions = (
        planning_scene.data.qpos[:7].copy()
    )

    # ---------- 第二阶段：创建关节插值轨迹 ----------

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

    # ---------- 第三阶段：执行动力学运动 ----------

    while not path.finished():
        # 当前控制拍应该发送的插值关节目标。
        current_joint_target = path.target()

        # 写入执行器目标，不直接修改实际 qpos。
        command_arm_joint_positions(
            control_scene,
            current_joint_target,
        )

        # 一个控制拍内执行10次物理仿真。
        for _ in range(CONTROL_EVERY):
            step_with_bias_compensation(control_scene)

        path.advance()

    # JointPath完成后，显式发送最终关节目标。
    command_arm_joint_positions(
        control_scene,
        target_joint_positions,
    )

    # 保持目标1秒，让实际关节稳定下来。
    for _ in range(SETTLE_STEPS):
        step_with_bias_compensation(control_scene)

    # ---------- 第四阶段：测量实际结果 ----------

    actual_joint_positions = (
        control_scene.data.qpos[:7].copy()
    )

    actual_ee_position = get_ee_position(
        control_scene
    )

    joint_error = (
        target_joint_positions
        - actual_joint_positions
    )

    ee_error = (
        target_position
        - actual_ee_position
    )

    max_joint_error = float(
        np.max(np.abs(joint_error))
    )

    ee_error_norm = float(
        np.linalg.norm(ee_error)
    )

    joint_velocity_norm = float(
        np.linalg.norm(
            control_scene.data.qvel[:7]
        )
    )

    np.set_printoptions(
        precision=9,
        suppress=True,
    )

    print("IK target joint positions:")
    print(target_joint_positions)

    print()
    print("Final actuator commands:")
    print(control_scene.data.ctrl[:7])

    print()
    print("Actual joint positions:")
    print(actual_joint_positions)

    print()
    print("Joint tracking error:")
    print(joint_error)

    print()
    print("Target end-effector position:")
    print(target_position)

    print()
    print("Actual end-effector position:")
    print(actual_ee_position)

    print()
    print("End-effector position error:")
    print(ee_error)

    print()
    print(
        f"Simulation time:       "
        f"{control_scene.data.time:.3f} s"
    )
    print(
        f"Maximum joint error:   "
        f"{max_joint_error:.6e} rad"
    )
    print(
        f"End-effector error:    "
        f"{ee_error_norm:.6e} m"
    )
    print(
        f"Joint velocity norm:   "
        f"{joint_velocity_norm:.6e} rad/s"
    )

    print()
    print("Final bias forces:")
    print(control_scene.data.qfrc_bias[:7])

    print()
    print("Final applied compensation:")
    print(control_scene.data.qfrc_applied[:7])

    # 当前模型的位置执行器存在一定稳态跟踪误差，
    # 因此这里先使用较宽松的验收标准。
    assert max_joint_error < 1e-5
    assert ee_error_norm < 1e-4
    assert joint_velocity_norm < 1e-3

    print()
    print("IK tracking with bias compensation: PASSED")


if __name__ == "__main__":
    main()