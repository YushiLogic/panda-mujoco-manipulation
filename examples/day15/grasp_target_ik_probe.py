"""Day 15：检查固定抓取目标的6D IK可达性。

本程序使用独立的planning scene依次求解：

    home -> pregrasp -> grasp -> lift

solve_pose_ik会修改planning scene中的qpos，但不会推进仿真时间。
因此这只是运动学预检，不代表机械臂已经通过动力学完成抓取。
"""

import mujoco
import numpy as np

from panda_mujoco.contact import ContactSensor
from panda_mujoco.grasp_task import generate_grasp_targets
from panda_mujoco.ik import solve_pose_ik
from panda_mujoco.simulation import PandaScene


# 代表方块在地面上落稳后的中心位置。
# 方块半高为0.02 m，因此中心高度约为0.02 m。
REPRESENTATIVE_CUBE_POSITION = np.array(
    [0.45, 0.0, 0.02],
    dtype=float,
)


def place_cube_for_kinematic_probe(
    scene: PandaScene,
    cube_position: np.ndarray,
) -> None:
    """在独立规划场景中设置代表性的方块位姿。

    这是Day15的固定运动学探针。Day16会把方块reset正式封装成
    可复现、可随机化的公共模块。
    """

    model = scene.model
    data = scene.data

    cube_joint_id = model.joint("cube_joint").id

    # free joint在qpos中占7格：xyz + wxyz。
    qpos_address = int(
        model.jnt_qposadr[cube_joint_id]
    )

    # free joint在qvel中占6格：线速度 + 角速度。
    dof_address = int(
        model.jnt_dofadr[cube_joint_id]
    )

    data.qpos[qpos_address:qpos_address + 3] = (
        cube_position
    )

    # 单位四元数wxyz，表示方块没有旋转。
    data.qpos[qpos_address + 3:qpos_address + 7] = (
        np.array([1.0, 0.0, 0.0, 0.0])
    )

    # 清除方块原有的线速度和角速度。
    data.qvel[dof_address:dof_address + 6] = 0.0

    # qpos改变后，重新计算body、site和contact等派生量。
    # mj_forward不会推进仿真时间。
    mujoco.mj_forward(model, data)


def main() -> None:
    """依次检查三个抓取目标的IK可达性。"""

    # 这个场景专门用于IK规划，不是动力学执行场景。
    planning_scene = PandaScene()

    place_cube_for_kinematic_probe(
        planning_scene,
        REPRESENTATIVE_CUBE_POSITION,
    )

    model = planning_scene.model
    data = planning_scene.data

    cube_position = (
        data.body("cube").xpos.copy()
    )

    targets = generate_grasp_targets(
        cube_position
    )

    print("Representative cube position:")
    print(cube_position)

    print("\nGenerated grasp targets")
    print("pregrasp:", targets.pregrasp_position)
    print("grasp:   ", targets.grasp_position)
    print("lift:    ", targets.lift_position)

    # Day5已经实现的接触读取器。
    contact_sensor = ContactSensor(model)

    target_sequence = [
        (
            "pregrasp",
            targets.pregrasp_position,
        ),
        (
            "grasp",
            targets.grasp_position,
        ),
        (
            "lift",
            targets.lift_position,
        ),
    ]

    print("\nSequential 6D IK preflight")

    for target_name, target_position in target_sequence:
        # 每次求解都从上一个目标的IK结果继续。
        # 因此顺序是home -> pregrasp -> grasp -> lift。
        result = solve_pose_ik(
            planning_scene,
            target_position,
            targets.rotation,
            position_tolerance=1e-4,
            orientation_tolerance=1e-3,
            max_iterations=200,
            max_joint_step=0.1,
        )

        left_contacts, right_contacts = (
            contact_sensor.finger_contacts(data)
        )

        position_error_mm = (
            result.final_position_error_norm * 1000.0
        )
        orientation_error_degrees = np.degrees(
            result.final_orientation_error_norm
        )

        print(
            f"{target_name:10s} | "
            f"success={result.success} | "
            f"iterations={result.iterations:3d} | "
            f"position error={position_error_mm:.6f} mm | "
            f"orientation error="
            f"{orientation_error_degrees:.6f} deg | "
            f"finger contacts="
            f"({left_contacts}, {right_contacts})"
        )

        # 三个目标都必须存在满足容差的6D IK解。
        assert result.success

        # 所有计算结果必须有限。
        assert np.all(
            np.isfinite(result.final_position)
        )
        assert np.all(
            np.isfinite(result.final_rotation)
        )
        assert np.all(
            np.isfinite(data.qpos)
        )

        # 夹爪始终保持home的张开命令。
        assert data.ctrl[7] == 255.0

        # pregrasp和grasp阶段尚未发送闭合命令，
        # 不应该提前产生手指与方块的接触。
        if target_name in {"pregrasp", "grasp"}:
            assert left_contacts == 0
            assert right_contacts == 0

    # IK只调用mj_forward，不调用mj_step，所以时间仍为0。
    assert data.time == 0.0

    print("\nFixed grasp target IK preflight: PASSED")


if __name__ == "__main__":
    main()