"""可视化演示从home移动到方块上方的pregrasp位姿。"""

import os
import sys
import time

# Windows下帮助MuJoCo找到conda环境的渲染DLL。
dll_directory = os.path.join(
    os.path.dirname(sys.executable),
    "Library",
    "bin",
)

if os.path.isdir(dll_directory):
    os.environ["PATH"] = (
        dll_directory
        + os.pathsep
        + os.environ["PATH"]
    )

import mujoco.viewer
import numpy as np

from panda_mujoco.cube_reset import (
    reset_cube,
    sample_cube_pose,
    settle_cube,
)
from panda_mujoco.grasp_task import (
    generate_grasp_targets,
)
from panda_mujoco.gripper import open_gripper
from panda_mujoco.motion import (
    move_end_effector_to,
)
from panda_mujoco.simulation import PandaScene


CUBE_SEED = 42
DISPLAY_PAUSE = 2.0


def wait_without_stepping(
    viewer,
    duration: float,
) -> bool:
    """刷新画面，但不推进动力学。"""

    end_time = time.perf_counter() + duration

    while time.perf_counter() < end_time:
        if not viewer.is_running():
            return False

        viewer.sync()
        time.sleep(0.01)

    return True


def make_visual_callback(viewer):
    """创建轨迹执行期间使用的画面刷新函数。"""

    def update_viewer(scene: PandaScene) -> None:
        if not viewer.is_running():
            return

        viewer.sync()
        time.sleep(scene.model.opt.timestep)

    return update_viewer


def main() -> None:
    scene = PandaScene()

    # ---------- 准备方块 ----------

    sampled_pose = sample_cube_pose(
        CUBE_SEED
    )

    reset_cube(
        scene,
        sampled_pose,
    )

    # 必须先让方块落稳，再用最终位置计算抓取目标。
    settle_result = settle_cube(scene)

    if not settle_result.settled:
        raise RuntimeError(
            "cube failed to settle"
        )

    cube_position = (
        settle_result.final_position.copy()
    )

    # xmat保存方块body相对于世界坐标系的实际旋转矩阵。
    # MuJoCo提供的是长度为9的一维数组，因此恢复成3×3矩阵。
    cube_rotation = (
        scene.data.body("cube")
        .xmat.reshape(3, 3)
        .copy()
    )

    # 对于保持直立、只绕世界z轴旋转的方块：
    #
    # R[0, 0] = cos(yaw)
    # R[1, 0] = sin(yaw)
    #
    # 因此可以使用atan2恢复实际yaw。
    cube_yaw = float(
        np.arctan2(
            cube_rotation[1, 0],
            cube_rotation[0, 0],
        )
    )

    grasp_targets = generate_grasp_targets(
        cube_position,
        cube_yaw=cube_yaw,
    )

    # Day17的MOVE_ABOVE阶段要求夹爪保持张开。
    open_gripper(scene)

    np.set_printoptions(
        precision=6,
        suppress=True,
    )

    print(
        "\nSettled cube yaw:",
        np.degrees(cube_yaw),
        "degrees",
    )

    print(
        "\nCube rotation:"
    )
    print(cube_rotation)
    # 方块局部+y轴在世界坐标系中的方向。
    cube_local_y = cube_rotation[:, 1]

    # 目标末端局部+y轴是夹爪张合方向。
    gripper_local_y = (
        grasp_targets.rotation[:, 1]
    )

    # 当前定义让夹爪局部+y与方块局部-y对齐。
    alignment = float(
        np.dot(
            gripper_local_y,
            -cube_local_y,
        )
    )

    print(
        "\nGripper/cube axis alignment:",
        alignment,
    )
    print("\nPregrasp target position:")
    print(grasp_targets.pregrasp_position)

    print("\nPregrasp target rotation:")
    print(grasp_targets.rotation)

    print(
        "\nThe cube has already settled. "
        "The animation will focus on the arm motion."
    )

    # ---------- 显示并执行MOVE_ABOVE ----------

    with mujoco.viewer.launch_passive(
        scene.model,
        scene.data,
    ) as viewer:
        viewer.cam.lookat[:] = [
            0.38,
            0.0,
            0.38,
        ]
        viewer.cam.distance = 1.6
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -25.0

        print(
            "\nStage 1: displaying the start pose."
        )

        if not wait_without_stepping(
            viewer,
            DISPLAY_PAUSE,
        ):
            return

        visual_callback = make_visual_callback(
            viewer
        )

        # ---------- MOVE_ABOVE ----------

        print(
            "Stage 2: MOVE_ABOVE."
        )

        pregrasp_result = move_end_effector_to(
            scene,
            grasp_targets.pregrasp_position,
            grasp_targets.rotation,
            command_count=201,
            physics_steps_per_command=10,
            settle_steps=250,
            step_callback=visual_callback,
        )

        print_motion_result(
            "MOVE_ABOVE",
            pregrasp_result,
        )

        if not pregrasp_result.success:
            raise RuntimeError(
                "MOVE_ABOVE failed: "
                f"{pregrasp_result.reason}"
            )

        if not wait_without_stepping(
            viewer,
            1.0,
        ):
            return

        # ---------- APPROACH ----------

        print(
            "\nStage 3: APPROACH."
        )

        approach_result = move_end_effector_to(
            scene,
            grasp_targets.grasp_position,
            grasp_targets.rotation,
            command_count=101,
            physics_steps_per_command=10,
            settle_steps=250,
            step_callback=visual_callback,
        )

        print_motion_result(
            "APPROACH",
            approach_result,
        )

        if not approach_result.success:
            raise RuntimeError(
                "APPROACH failed: "
                f"{approach_result.reason}"
            )

        print(
            "\nStage 4: displaying the "
            "final grasp pose."
        )

        wait_without_stepping(
            viewer,
            DISPLAY_PAUSE,
        )

    print(
        "\nMotion primitives visual demo: FINISHED"
    )
def print_motion_result(
    stage_name: str,
    result,
) -> None:
    """打印一次末端运动的关键验收结果。"""

    print(f"\n{stage_name} result")
    print("success:", result.success)
    print("reason:", result.reason)

    print(
        "simulation duration:",
        result.simulation_duration,
        "s",
    )

    print(
        "IK iterations:",
        result.ik_iterations,
    )

    print(
        "final position:",
        result.final_position,
    )

    print(
        "position error:",
        result.position_error_norm * 1000.0,
        "mm",
    )

    print(
        "orientation error:",
        np.degrees(
            result.orientation_error_norm
        ),
        "degrees",
    )

    print(
        "maximum joint error:",
        result.maximum_joint_error,
        "rad",
    )

    print(
        "joint limit violation:",
        result.joint_limit_violation,
    )

    print(
        "gripper width:",
        result.gripper_width,
        "m",
    )
if __name__ == "__main__":
    main()