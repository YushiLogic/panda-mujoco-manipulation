"""Day 18：可视化持续双侧接触的建立过程。"""

import math
import time

import mujoco
import mujoco.viewer

from contact_persistence_probe import (
    CONTACT_DURATION,
    SIMULATION_DURATION,
    place_cube_between_fingers,
)
from panda_mujoco.contact import (
    BilateralContactTracker,
    ContactSensor,
)
from panda_mujoco.gripper import close_gripper
from panda_mujoco.simulation import PandaScene


PHYSICS_STEPS_PER_RENDER = 10


def display_without_stepping(
    viewer,
    duration: float,
) -> None:
    """保持当前物理状态，只刷新画面。"""

    deadline = time.perf_counter() + duration

    while (
        viewer.is_running()
        and time.perf_counter() < deadline
    ):
        viewer.sync()
        time.sleep(1.0 / 60.0)


def main() -> None:
    scene = PandaScene()
    model = scene.model
    data = scene.data

    # 这个演示只观察夹爪接触过程，因此关闭重力。
    model.opt.gravity[:] = 0.0
    place_cube_between_fingers(scene)

    sensor = ContactSensor(model)

    required_steps = math.ceil(
        CONTACT_DURATION / model.opt.timestep
    )

    tracker = BilateralContactTracker(
        required_steps=required_steps,
    )

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:
        # 把相机对准夹爪中心。
        viewer.cam.lookat[:] = (
            data.body("ee_center_body").xpos
        )
        viewer.cam.distance = 0.45
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -20.0

        print("Stage 1: open gripper")
        print(sensor.snapshot(data))

        # 暂停1.5秒，让我们先看清初始状态。
        display_without_stepping(viewer, 1.5)

        if not viewer.is_running():
            print("Viewer closed before motion.")
            return

        print("\nStage 2: closing gripper")
        close_gripper(scene)

        total_steps = math.ceil(
            SIMULATION_DURATION
            / model.opt.timestep
        )

        start_simulation_time = float(data.time)
        start_wall_time = time.perf_counter()

        previous_bilateral = False
        previous_persistent = False

        for step_index in range(total_steps):
            if not viewer.is_running():
                print("Viewer closed during motion.")
                return

            mujoco.mj_step(model, data)

            snapshot = sensor.snapshot(data)
            persistent = tracker.update(snapshot)

            bilateral_changed = (
                snapshot.bilateral_contact
                != previous_bilateral
            )
            persistent_changed = (
                persistent
                != previous_persistent
            )

            if bilateral_changed or persistent_changed:
                print(
                    f"time={data.time:.3f} s | "
                    f"left={snapshot.left_cube_contacts} | "
                    f"right={snapshot.right_cube_contacts} | "
                    f"counter={tracker.consecutive_steps} | "
                    f"bilateral={snapshot.bilateral_contact} | "
                    f"persistent={persistent}"
                )

            previous_bilateral = (
                snapshot.bilateral_contact
            )
            previous_persistent = persistent

            # 不需要每个2 ms都刷新窗口。
            if (
                (step_index + 1)
                % PHYSICS_STEPS_PER_RENDER
                == 0
            ):
                viewer.sync()

                simulated_elapsed = (
                    float(data.time)
                    - start_simulation_time
                )
                wall_elapsed = (
                    time.perf_counter()
                    - start_wall_time
                )

                remaining = (
                    simulated_elapsed - wall_elapsed
                )

                if remaining > 0.0:
                    time.sleep(remaining)

        final_snapshot = sensor.snapshot(data)

        print("\nStage 3: persistent contact")
        print(final_snapshot)
        print(
            "consecutive steps:",
            tracker.consecutive_steps,
        )
        print(
            "persistent contact:",
            tracker.persistent_contact,
        )

        # 保持最终画面，观察手指停在方块两侧。
        display_without_stepping(viewer, 3.0)

    print("\nContact persistence visual demo: FINISHED")


if __name__ == "__main__":
    main()