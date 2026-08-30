import mujoco
import mujoco.viewer
import time
def panda():
    model = mujoco.MjModel.from_xml_path(r"D:\Mujoco Project\model\franka_emika_panda\scene_with_cube.xml")
    data = mujoco.MjData(model)
    data.qpos[:9] = [1, 0, 0, -1.57079, 0, 3.0, -1.7853, 0.04, 0.04]   # 机器人
    data.qpos[9:16] = [0.35, 0.40, 0.05, 1.0, 0.0, 0.0, 0.0]            # 方块：xyz + 四元数
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
if __name__ == "__main__":
    panda()