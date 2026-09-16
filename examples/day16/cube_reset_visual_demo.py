"""Day 16：可视化方块随机复位和落稳过程。"""

import os
import sys
import time

# 使用python.exe完整路径启动时，确保能找到conda环境的渲染DLL。
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

import mujoco
import mujoco.viewer
import numpy as np

from panda_mujoco.arm_control import (
    apply_arm_bias_compensation,
)
from panda_mujoco.cube_reset import (
    get_cube_joint_addresses,
    reset_cube,
    sample_cube_pose,
)
from panda_mujoco.simulation import PandaScene


DEMO_SEEDS = (42, 43, 44)
SPAWN_DISPLAY_DURATION = 1.0
SETTLE_DURATION = 1.5
FINAL_DISPLAY_DURATION = 1.0


def wait_without_stepping(
    viewer,
    duration: float,
) -> bool:
    """保持画面，但不推进MuJoCo动力学。"""

    end_wall_time = (
        time.perf_counter() + duration
    )

    while time.perf_counter() < end_wall_time:
        if not viewer.is_running():
            return False

        # sync只刷新显示，不推进仿真时间。
        viewer.sync()
        time.sleep(0.01)

    return True


def step_in_real_time(
    scene: PandaScene,
    viewer,
    duration: float,
) -> bool:
    """以接近真实时间的速度推进MuJoCo。"""

    end_simulation_time = (
        scene.data.time + duration
    )
    physics_timestep = (
        scene.model.opt.timestep
    )

    while (
        scene.data.time
        < end_simulation_time
    ):
        if not viewer.is_running():
            return False

        wall_step_start = time.perf_counter()

        # 补偿机械臂当前的重力和其他bias力，
        # 防止观察方块时机械臂自己明显下坠。
        apply_arm_bias_compensation(scene)

        # 真正推进一个物理时间步。
        mujoco.mj_step(
            scene.model,
            scene.data,
        )

        # 将最新状态送到显示窗口。
        viewer.sync()

        # MuJoCo无窗口运行通常远快于现实时间。
        # sleep让动画速度大致对应真实时间。
        elapsed = (
            time.perf_counter()
            - wall_step_start
        )
        remaining = physics_timestep - elapsed

        if remaining > 0.0:
            time.sleep(remaining)

    return True


def main() -> None:
    """依次演示三个seed的复位和落稳过程。"""

    scene = PandaScene()

    qpos_address, qvel_address = (
        get_cube_joint_addresses(
            scene.model
        )
    )

    with mujoco.viewer.launch_passive(
        scene.model,
        scene.data,
    ) as viewer:

        # 将相机对准方块所在的桌面区域。
        viewer.cam.lookat[:] = [
            0.45,
            0.0,
            0.10,
        ]
        viewer.cam.distance = 0.9
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -25.0

        for seed in DEMO_SEEDS:
            if not viewer.is_running():
                return

            # 每轮先让机械臂和整个场景回到home。
            scene.reset_to_home()

            pose = sample_cube_pose(seed)
            reset_cube(scene, pose)

            print(
                f"\nseed={seed} | "
                f"spawn position={pose.position} | "
                f"yaw={np.degrees(pose.yaw):.3f} deg"
            )

            print(
                "The cube is suspended. "
                "Physics is temporarily paused."
            )

            # 暂停动力学一秒，让我们看清生成位置。
            if not wait_without_stepping(
                viewer,
                SPAWN_DISPLAY_DURATION,
            ):
                return

            print("Physics started: cube falling.")

            # 推进动力学，观察自由落体和碰撞。
            if not step_in_real_time(
                scene,
                viewer,
                SETTLE_DURATION,
            ):
                return

            cube_qpos = scene.data.qpos[
                qpos_address:qpos_address + 7
            ].copy()

            cube_qvel = scene.data.qvel[
                qvel_address:qvel_address + 6
            ].copy()

            print(
                "settled position:",
                cube_qpos[:3],
            )
            print(
                "final 6D speed norm:",
                np.linalg.norm(cube_qvel),
            )

            # 落稳后再停留一秒，方便观察结果。
            if not wait_without_stepping(
                viewer,
                FINAL_DISPLAY_DURATION,
            ):
                return

    print("\nCube reset visual demo: FINISHED")


if __name__ == "__main__":
    main()