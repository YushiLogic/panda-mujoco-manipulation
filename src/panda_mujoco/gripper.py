from panda_mujoco.simulation import PandaScene


GRIPPER_CLOSED_WIDTH = 0.0
GRIPPER_OPEN_WIDTH = 0.08

def width_to_ctrl(width: float) -> float:
    if width < GRIPPER_CLOSED_WIDTH or width > GRIPPER_OPEN_WIDTH:
        raise ValueError(
            f"Width must be in the range "
            f"[{GRIPPER_CLOSED_WIDTH}, {GRIPPER_OPEN_WIDTH}]"
        )

    ctrl = 255.0 * width / GRIPPER_OPEN_WIDTH
    return ctrl

def command_gripper(scene: PandaScene, width: float):
    ctrl = width_to_ctrl(width)
    actuator_id=scene.model.actuator("actuator8").id
    scene.data.ctrl[actuator_id] = ctrl

def open_gripper(scene: PandaScene) -> None:
    """发送完全张开的夹爪命令。"""

    command_gripper(scene, GRIPPER_OPEN_WIDTH)


def close_gripper(scene: PandaScene) -> None:
    """发送完全闭合的夹爪命令。"""

    command_gripper(scene, GRIPPER_CLOSED_WIDTH)

def get_gripper_width(scene: PandaScene) -> float:
    """读取两根手指的实际位置，返回当前总开口宽度（米）。"""

    left = float(
        scene.data.joint("finger_joint1").qpos[0]
    )

    right = float(
        scene.data.joint("finger_joint2").qpos[0]
    )

    width = left + right
    return width

if __name__ == "__main__":
    print(width_to_ctrl(0))
    print(width_to_ctrl(0.04))
    print(width_to_ctrl(0.08))