import numpy as np
from scipy.spatial.transform import Rotation

from panda_mujoco.action_mapping import map_action_to_targets
from panda_mujoco.kinematics import get_ee_pose
from panda_mujoco.simulation import PandaScene

scene = PandaScene()
start_position, start_rotation = get_ee_pose(scene)
axis_names = ("x", "y", "z", "Rx", "Ry", "Rz")

np.set_printoptions(precision=3, suppress=True)

for axis_index, axis_name in enumerate(axis_names):
    for sign in (1.0, -1.0):
        action = np.zeros(7, dtype=np.float32)
        action[axis_index] = sign

        next_position, next_rotation, next_width = map_action_to_targets(
            action, start_position, start_rotation, 0.04
        )

        position_change_mm = (next_position - start_position) * 1000
        world_rotation_change = next_rotation @ start_rotation.T
        rotation_change_deg = np.rad2deg(
            Rotation.from_matrix(world_rotation_change).as_rotvec()
        )

        print(
            f"{sign:+.0f}{axis_name:>2} | "
            f"position mm: {position_change_mm} | "
            f"rotation deg: {rotation_change_deg} | "
            f"width m: {next_width:.3f}"
        )