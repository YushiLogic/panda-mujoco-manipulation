import time

import mujoco.viewer
import numpy as np

from panda_mujoco.continuous_env import PandaContinuousEnv
from panda_mujoco.kinematics import get_ee_position
from scipy.spatial.transform import Rotation
from panda_mujoco.gripper import get_gripper_width
from panda_mujoco.kinematics import get_ee_pose, get_ee_position


def main():
    env = PandaContinuousEnv()
    env.reset(seed=42)
    scene = env.task.scene
    start_position = get_ee_position(scene).copy()
    start_rotation = get_ee_pose(scene)[1].copy()
    move_y = np.zeros(7, dtype=np.float32)
    move_y[1] = 1.0
    hold = np.zeros(7, dtype=np.float32)
    rotate_z = np.zeros(7, dtype=np.float32)
    rotate_z[5] = 1.0       # 世界 +Z 旋转，每次目标增加 2°

    close_gripper = np.zeros(7, dtype=np.float32)
    close_gripper[6] = -1.0  # 每次目标开口减少 2 mm

    with mujoco.viewer.launch_passive(scene.model, scene.data) as viewer:
        viewer.cam.lookat[:] = [0.4, 0.0, 0.4]
        viewer.cam.distance = 1.7
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20

        viewer.sync()
        time.sleep(1.0)

        stages = [
        ("MOVE +Y", move_y, 20),
        ("HOLD position", hold, 50),
        ("ROTATE +WORLD Z", rotate_z, 10),
        ("HOLD rotation", hold, 50),
        ("CLOSE GRIPPER", close_gripper, 20),
        ("HOLD gripper width", hold, 50),
    ]
        for name, action, count in stages:
            print("\n" + name)

            for _ in range(count):
                if not viewer.is_running():
                    return

                wall_start = time.perf_counter()
                _, _, _, _, info = env.step(action)
                viewer.sync()

                # 只控制动画播放节奏；仿真已由 env.step() 推进。
                remaining = info["control_period_s"] - (
                    time.perf_counter() - wall_start
                )
                if remaining > 0:
                    time.sleep(remaining)

            actual_position = get_ee_position(scene)
            print("IK success:", info["ik_success"])
            print(
                "target displacement mm:",
                (env.target_position - start_position) * 1000,
            )
            print(
                "actual displacement mm:",
                (actual_position - start_position) * 1000,
            )
            print("tracking error mm:", info["tracking_error_m"] * 1000)
            _, actual_rotation = get_ee_pose(scene)

            target_rotation_deg = np.rad2deg(
                Rotation.from_matrix(
                    env.target_rotation @ start_rotation.T
                ).as_rotvec()
            )
            actual_rotation_deg = np.rad2deg(
                Rotation.from_matrix(
                    actual_rotation @ start_rotation.T
                ).as_rotvec()
            )

            print("target rotation vector deg:", target_rotation_deg)
            print("actual rotation vector deg:", actual_rotation_deg)
            print("target gripper width m:", env.target_width)
            print("actual gripper width m:", get_gripper_width(scene))
            print("orientation error deg:", np.rad2deg(info["orientation_error_rad"]))
            print("gripper width error mm:", info["gripper_width_error_m"] * 1000)
        time.sleep(2.0)

    env.close()


if __name__ == "__main__":
    main()
