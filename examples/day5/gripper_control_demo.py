"""夹爪控制模块的可视化演示。

运动顺序：
完全张开 → 半开 → 完全闭合 → 再次张开
"""

import time

import mujoco
import mujoco.viewer

from panda_mujoco.gripper import (
    close_gripper,
    command_gripper,
    get_gripper_width,
    open_gripper,
)
from panda_mujoco.simulation import PandaScene


def step_for_duration(scene: PandaScene, viewer, duration: float) -> bool:
    """推进指定长度的仿真时间，同时刷新 viewer。

    Returns:
        viewer 仍在运行时返回 True；
        用户提前关闭窗口时返回 False。
    """

    end_time = scene.data.time + duration
    physics_dt = scene.model.opt.timestep

    while scene.data.time < end_time:
        if not viewer.is_running():
            return False

        wall_step_start = time.perf_counter()

        mujoco.mj_step(scene.model, scene.data)
        viewer.sync()

        # 控制显示速度，使仿真时间大致对应现实时间。
        elapsed = time.perf_counter() - wall_step_start
        remaining = physics_dt - elapsed

        if remaining > 0.0:
            time.sleep(remaining)

    return True


def print_gripper_state(scene: PandaScene, label: str) -> None:
    """打印夹爪命令和实际开口宽度。"""

    actuator_id = scene.model.actuator("actuator8").id
    ctrl = float(scene.data.ctrl[actuator_id])
    actual_width = get_gripper_width(scene)

    print(
        f"{label:<8} | "
        f"ctrl={ctrl:6.1f} | "
        f"actual_width={actual_width:.5f} m"
    )


def main() -> None:
    scene = PandaScene()

    with mujoco.viewer.launch_passive(
        scene.model,
        scene.data,
    ) as viewer:

        # PandaScene 的 home 状态本来就是完全张开。
        open_gripper(scene)

        if not step_for_duration(scene, viewer, 2.0):
            return

        print_gripper_state(scene, "OPEN")

        # 目标总开口宽度 0.04 m。
        command_gripper(scene, 0.04)

        if not step_for_duration(scene, viewer, 2.0):
            return

        print_gripper_state(scene, "HALF")

        close_gripper(scene)

        if not step_for_duration(scene, viewer, 2.0):
            return

        print_gripper_state(scene, "CLOSED")

        open_gripper(scene)

        if not step_for_duration(scene, viewer, 2.0):
            return

        print_gripper_state(scene, "OPEN")


if __name__ == "__main__":
    main()