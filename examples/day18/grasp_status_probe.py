"""Day 18：验证接触分类、持续计数与抓取状态语义。

这个探针故意把六类状态分开构造。前五类关闭重力并把方块放在夹爪
中间，用来隔离手指接触逻辑；最后一类恢复正常重力，让方块落在地面
上，确认方块-地面接触不会被误判为抓取。
"""

import mujoco

from contact_persistence_probe import (
    place_cube_between_fingers,
)
from panda_mujoco.contact import (
    GraspMonitor,
    GraspStatus,
)
from panda_mujoco.cube_reset import settle_cube
from panda_mujoco.gripper import close_gripper
from panda_mujoco.simulation import PandaScene


def print_status(
    label: str,
    status: GraspStatus,
) -> None:
    """用统一格式打印一种抓取状态。"""

    contact = status.contact_snapshot

    print(f"\n{label}")
    print(
        "contacts: "
        f"left={contact.left_cube_contacts}, "
        f"right={contact.right_cube_contacts}, "
        f"floor={contact.cube_floor_contacts}, "
        f"total={contact.total_contacts}"
    )
    print(
        "contact history: "
        f"{status.consecutive_bilateral_steps}/"
        f"{status.required_bilateral_steps} steps"
    )
    print(
        f"bilateral={contact.bilateral_contact} | "
        f"persistent={status.persistent_contact}"
    )
    print(
        f"gripper width={status.gripper_width:.6f} m | "
        f"width valid={status.width_valid}"
    )
    print(
        f"cube height={status.cube_height:.6f} m | "
        f"lift height={status.lift_height:.6f} m | "
        f"on floor={contact.cube_on_floor}"
    )
    print(
        f"linear speed={status.cube_linear_speed:.3e} m/s | "
        f"angular speed={status.cube_angular_speed:.3e} rad/s | "
        f"stable={status.cube_stable}"
    )
    print(
        f"candidate={status.grasp_candidate} | "
        f"success={status.grasp_success} | "
        f"reason={status.reason}"
    )


def main() -> None:
    # ------------------------------------------------------------------
    # 场景1～5：隔离手指和方块的接触，不让重力干扰实验。
    # ------------------------------------------------------------------
    scene = PandaScene()
    scene.model.opt.gravity[:] = 0.0

    place_cube_between_fingers(scene, offset=0.0)
    monitor = GraspMonitor(scene)

    # 1. 方块位于张开的夹爪中心，当前没有接触。
    monitor.reset(scene)
    no_contact = monitor.update(scene)
    print_status("1. NO CONTACT", no_contact)

    assert not no_contact.contact_snapshot.bilateral_contact
    assert not no_contact.grasp_candidate

    # 2. 方块只移动到左手指附近。
    place_cube_between_fingers(scene, offset=0.021)
    monitor.reset(scene)
    left_only = monitor.update(scene)
    print_status("2. LEFT ONLY", left_only)

    assert left_only.contact_snapshot.left_cube_contacts > 0
    assert left_only.contact_snapshot.right_cube_contacts == 0
    assert not left_only.grasp_candidate

    # 3. 方块只移动到右手指附近。
    place_cube_between_fingers(scene, offset=-0.021)
    monitor.reset(scene)
    right_only = monitor.update(scene)
    print_status("3. RIGHT ONLY", right_only)

    assert right_only.contact_snapshot.left_cube_contacts == 0
    assert right_only.contact_snapshot.right_cube_contacts > 0
    assert not right_only.grasp_candidate

    # 4～5. 方块回到中心，随后夹爪通过动力学逐渐闭合。
    place_cube_between_fingers(scene, offset=0.0)
    monitor.reset(scene)
    close_gripper(scene)

    short_bilateral = None
    persistent_bilateral = None

    max_steps = round(1.0 / scene.model.opt.timestep)

    for _ in range(max_steps):
        mujoco.mj_step(scene.model, scene.data)
        status = monitor.update(scene)

        # 距离门槛只差一步：双侧接触存在，但还不算持续接触。
        if (
            status.consecutive_bilateral_steps
            == status.required_bilateral_steps - 1
        ):
            short_bilateral = status

        # 第一次达到门槛后即可停止；不需要继续重复相同状态。
        if status.persistent_contact:
            persistent_bilateral = status
            break

    if short_bilateral is None:
        raise RuntimeError(
            "short bilateral contact state was not observed"
        )
    if persistent_bilateral is None:
        raise RuntimeError(
            "persistent bilateral contact was not reached"
        )

    print_status(
        "4. SHORT BILATERAL",
        short_bilateral,
    )
    assert short_bilateral.contact_snapshot.bilateral_contact
    assert not short_bilateral.persistent_contact
    assert not short_bilateral.grasp_candidate
    assert short_bilateral.reason == "contact_not_persistent"

    print_status(
        "5. PERSISTENT BILATERAL",
        persistent_bilateral,
    )
    assert persistent_bilateral.contact_snapshot.bilateral_contact
    assert persistent_bilateral.persistent_contact
    assert persistent_bilateral.width_valid
    assert persistent_bilateral.grasp_candidate
    assert not persistent_bilateral.grasp_success
    assert persistent_bilateral.reason == "cube_not_lifted"

    # ------------------------------------------------------------------
    # 场景6：正常重力下让方块落地，专门验证地面接触过滤。
    # ------------------------------------------------------------------
    floor_scene = PandaScene()
    settle_result = settle_cube(floor_scene)

    if not settle_result.settled:
        raise RuntimeError("cube failed to settle on floor")

    floor_monitor = GraspMonitor(floor_scene)
    floor_contact = floor_monitor.update(floor_scene)
    print_status("6. CUBE ON FLOOR", floor_contact)

    assert floor_contact.contact_snapshot.cube_floor_contacts > 0
    assert floor_contact.contact_snapshot.cube_on_floor
    assert not floor_contact.contact_snapshot.bilateral_contact
    assert not floor_contact.grasp_candidate
    assert not floor_contact.grasp_success

    print("\nDay 18 grasp status probe: PASSED")


if __name__ == "__main__":
    main()
