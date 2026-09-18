"""Day 19：运行一次可视化的完整脚本抓取状态机。"""

import argparse
import os
import sys
import time

dll_directory = os.path.join(os.path.dirname(sys.executable), "Library", "bin")
if os.path.isdir(dll_directory):
    os.environ["PATH"] = dll_directory + os.pathsep + os.environ["PATH"]

import mujoco.viewer
import numpy as np

from panda_mujoco.pick_controller import PickResult, run_scripted_pick
from panda_mujoco.simulation import PandaScene


def print_result(result: PickResult) -> None:
    """按状态打印流程轨迹，便于定位成功或失败发生在哪一步。"""

    print("\nState-machine trace")
    for record in result.records:
        print(
            f"{record.state.value:<15} | success={str(record.success):<5} | "
            f"reason={record.reason:<24} | time={record.simulation_duration:.3f} s"
        )

    status = result.final_status
    print("\nFinal grasp measurements")
    print("result success:", result.success)
    print("final state:", result.final_state.value)
    print("reason:", result.reason)
    print("failed stage:", None if result.failed_stage is None else result.failed_stage.value)
    print("bilateral contact:", status.contact_snapshot.bilateral_contact)
    print("persistent contact:", status.persistent_contact)
    print("gripper width:", status.gripper_width, "m")
    print("cube lift height:", status.lift_height, "m")
    print("cube linear speed:", status.cube_linear_speed, "m/s")
    print("cube angular speed:", status.cube_angular_speed, "rad/s")
    print("continuous successful hold:", result.hold_duration, "s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    np.set_printoptions(precision=6, suppress=True)
    scene = PandaScene()

    if args.headless:
        result = run_scripted_pick(scene, seed=args.seed)
    else:
        with mujoco.viewer.launch_passive(scene.model, scene.data) as viewer:
            viewer.cam.lookat[:] = [0.38, 0.0, 0.30]
            viewer.cam.distance = 1.5
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -25.0

            def visual_callback(callback_scene: PandaScene) -> None:
                if viewer.is_running():
                    viewer.sync()
                    time.sleep(callback_scene.model.opt.timestep)

            print("Running RESET -> MOVE_ABOVE -> APPROACH -> CLOSE -> LIFT...")
            result = run_scripted_pick(
                scene,
                seed=args.seed,
                step_callback=visual_callback,
            )

            print_result(result)
            print("\nThe final state will remain visible for 3 seconds.")
            end_time = time.perf_counter() + 3.0
            while viewer.is_running() and time.perf_counter() < end_time:
                viewer.sync()
                time.sleep(0.01)

    if args.headless:
        print_result(result)

    if not result.success:
        raise RuntimeError(
            f"scripted pick failed at {result.failed_stage}: {result.reason}"
        )

    print("\nDay 19 scripted pick: PASSED")


if __name__ == "__main__":
    main()
