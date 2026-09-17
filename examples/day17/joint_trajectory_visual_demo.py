"""可视化演示关节轨迹的动力学执行过程。"""

import os
import sys
import time

# Windows下确保MuJoCo能够找到conda环境中的渲染DLL。
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

from panda_mujoco.motion import (
    execute_joint_trajectory,
    linear_joint_trajectory,
)
from panda_mujoco.simulation import PandaScene


COMMAND_COUNT = 101
PHYSICS_STEPS_PER_COMMAND = 10
DISPLAY_PAUSE = 1.0
SETTLE_STEPS = 250

def wait_without_stepping(
    viewer,
    duration: float,
) -> bool:
    """暂停画面，但不推进MuJoCo仿真时间。"""

    end_time = time.perf_counter() + duration

    while time.perf_counter() < end_time:
        if not viewer.is_running():
            return False

        viewer.sync()
        time.sleep(0.01)

    return True


def make_visual_callback(viewer):
    """生成一个供轨迹执行函数调用的画面刷新函数。"""

    def update_viewer(scene: PandaScene) -> None:
        # 如果用户提前关闭窗口，就不再刷新画面。
        if not viewer.is_running():
            return

        # 将最新的仿真状态显示到窗口。
        viewer.sync()

        # MuJoCo无界面运行速度远快于现实。
        # 暂停一个物理时间步，让动画接近真实速度。
        time.sleep(scene.model.opt.timestep)

    return update_viewer


def main() -> None:
    """演示从home出发、转动joint1、再返回home。"""

    scene = PandaScene()

    # 读取机械臂当前的实际关节角，而不是手动重复home数值。
    home_positions = scene.data.qpos[:7].copy()

    # 创建一个简单安全的目标：只让joint1增加0.2 rad。
    target_positions = home_positions.copy()
    target_positions[0] += 0.2

    outward_trajectory = linear_joint_trajectory(
        home_positions,
        target_positions,
        command_count=COMMAND_COUNT,
    )

    np.set_printoptions(
        precision=6,
        suppress=True,
    )

    print("Home joint positions:")
    print(home_positions)

    print("\nTarget joint positions:")
    print(target_positions)

    print("\nOutward trajectory shape:")
    print(outward_trajectory.shape)

    expected_duration = (
        (
            COMMAND_COUNT
            * PHYSICS_STEPS_PER_COMMAND
            + SETTLE_STEPS
        )
        * scene.model.opt.timestep
    )
    print(
        "\nExpected duration for one motion:",
        expected_duration,
        "s",
    )

    with mujoco.viewer.launch_passive(
        scene.model,
        scene.data,
    ) as viewer:
        # 将相机对准机械臂。
        viewer.cam.lookat[:] = [
            0.35,
            0.0,
            0.45,
        ]
        viewer.cam.distance = 1.8
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -20.0

        visual_callback = make_visual_callback(
            viewer
        )

        print(
            "\nStage 1: displaying the home pose."
        )

        if not wait_without_stepping(
            viewer,
            DISPLAY_PAUSE,
        ):
            return

        print(
            "Stage 2: moving joint1 by +0.2 rad."
        )

        outward_duration = execute_joint_trajectory(
            scene,
            outward_trajectory,
            physics_steps_per_command=(
                PHYSICS_STEPS_PER_COMMAND
            ),
            settle_steps=SETTLE_STEPS,
            step_callback=visual_callback,
        )
        outward_actual = (
            scene.data.qpos[:7].copy()
        )

        outward_error = (
            target_positions - outward_actual
        )

        print(
            "Actual joint positions after "
            "outward motion:"
        )
        print(outward_actual)

        print(
            "Maximum outward joint error:",
            np.max(np.abs(outward_error)),
            "rad",
        )

        print(
            "Outward simulation duration:",
            outward_duration,
            "s",
        )

        if not wait_without_stepping(
            viewer,
            DISPLAY_PAUSE,
        ):
            return

        # 返回轨迹从当前实际关节角开始，而不是假设机械臂
        # 已经百分之百等于上一个目标。
        return_start = (
            scene.data.qpos[:7].copy()
        )

        return_trajectory = linear_joint_trajectory(
            return_start,
            home_positions,
            command_count=COMMAND_COUNT,
        )

        print(
            "\nStage 3: returning to home."
        )

        return_duration = execute_joint_trajectory(
            scene,
            return_trajectory,
            physics_steps_per_command=(
                PHYSICS_STEPS_PER_COMMAND
            ),
            settle_steps=SETTLE_STEPS,
            step_callback=visual_callback,
        )
        final_positions = (
            scene.data.qpos[:7].copy()
        )

        final_error = (
            home_positions - final_positions
        )

        print("Final joint positions:")
        print(final_positions)

        print(
            "Maximum home joint error:",
            np.max(np.abs(final_error)),
            "rad",
        )

        print(
            "Return simulation duration:",
            return_duration,
            "s",
        )

        print(
            "\nStage 4: displaying the final pose."
        )

        wait_without_stepping(
            viewer,
            DISPLAY_PAUSE,
        )

    print("\nJoint trajectory visual demo: FINISHED")


if __name__ == "__main__":
    main()