"""Day 18：用真实MuJoCo接触验证持续双侧接触计数。"""

import math

import mujoco
import numpy as np

from panda_mujoco.contact import (
    BilateralContactTracker,
    ContactSensor,
)
from panda_mujoco.gripper import close_gripper
from panda_mujoco.simulation import PandaScene


CONTACT_DURATION = 0.1
SIMULATION_DURATION = 1.0


def place_cube_between_fingers(
    scene: PandaScene,
    offset: float = 0.0,
) -> None:
    """把方块放在两指之间，并可沿实际开合轴偏移。"""

    offset = float(offset)
    if not np.isfinite(offset):
        raise ValueError("offset must be finite")

    model = scene.model
    data = scene.data

    cube_joint_id = model.joint("cube_joint").id
    cube_qpos_address = int(
        model.jnt_qposadr[cube_joint_id]
    )
    cube_qvel_address = int(
        model.jnt_dofadr[cube_joint_id]
    )

    ee_body = data.body("ee_center_body")
    finger_joint_id = model.joint("finger_joint1").id
    opening_axis = data.xaxis[finger_joint_id].copy()
    cube_position = (
        ee_body.xpos + offset * opening_axis
    )

    cube_quaternion = np.zeros(4)
    mujoco.mju_mat2Quat(
        cube_quaternion,
        ee_body.xmat,
    )

    data.qpos[
        cube_qpos_address:
        cube_qpos_address + 3
    ] = cube_position

    data.qpos[
        cube_qpos_address + 3:
        cube_qpos_address + 7
    ] = cube_quaternion

    data.qvel[
        cube_qvel_address:
        cube_qvel_address + 6
    ] = 0.0

    mujoco.mj_forward(model, data)


def main() -> None:
    scene = PandaScene()
    model = scene.model
    data = scene.data

    # 这是接触计数实验，不是真实抓取任务。
    # 暂时关闭重力，防止方块在夹爪闭合前掉走。
    model.opt.gravity[:] = 0.0

    place_cube_between_fingers(scene)

    sensor = ContactSensor(model)

    required_steps = math.ceil(
        CONTACT_DURATION / model.opt.timestep
    )

    tracker = BilateralContactTracker(
        required_steps=required_steps,
    )

    initial_snapshot = sensor.snapshot(data)

    print("Initial contact state")
    print(initial_snapshot)
    print(
        "required contact duration:",
        CONTACT_DURATION,
        "s",
    )
    print(
        "model timestep:",
        model.opt.timestep,
        "s",
    )
    print(
        "required consecutive steps:",
        required_steps,
    )

    # 这里只发送闭合命令；手指仍需要通过物理仿真逐渐运动。
    close_gripper(scene)

    total_steps = math.ceil(
        SIMULATION_DURATION / model.opt.timestep
    )

    previous_bilateral = False
    previous_persistent = False

    for step_index in range(total_steps):
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

        first_step = step_index == 0
        final_step = step_index == total_steps - 1

        # 不打印全部500步，只打印状态发生变化的时刻。
        if (
            first_step
            or bilateral_changed
            or persistent_changed
            or final_step
        ):
            print(
                f"time={data.time:.3f} s | "
                f"left={snapshot.left_cube_contacts:2d} | "
                f"right={snapshot.right_cube_contacts:2d} | "
                f"counter={tracker.consecutive_steps:3d} | "
                f"bilateral={snapshot.bilateral_contact} | "
                f"persistent={persistent}"
            )

        previous_bilateral = snapshot.bilateral_contact
        previous_persistent = persistent

    final_snapshot = sensor.snapshot(data)

    print("\nFinal result")
    print(final_snapshot)
    print(
        "consecutive bilateral steps:",
        tracker.consecutive_steps,
    )
    print(
        "persistent contact:",
        tracker.persistent_contact,
    )

    assert final_snapshot.bilateral_contact
    assert tracker.persistent_contact

    print("\nContact persistence probe: PASSED")


if __name__ == "__main__":
    main()
