"""接触分类、持续计数和抓取状态判定测试。"""

import numpy as np
import pytest

from panda_mujoco.contact import (
    BilateralContactTracker,
    ContactSnapshot,
    GraspMonitor,
    GraspStatus,
)
from panda_mujoco.cube_reset import settle_cube
from panda_mujoco.simulation import PandaScene


def make_status(
    *,
    left_contacts: int = 1,
    right_contacts: int = 1,
    floor_contacts: int = 0,
    consecutive_steps: int = 50,
    required_steps: int = 50,
    width_valid: bool = True,
    cube_lifted: bool = True,
    cube_stable: bool = True,
) -> GraspStatus:
    """构造只用于规则单元测试的抓取状态。"""

    snapshot = ContactSnapshot(
        left_cube_contacts=left_contacts,
        right_cube_contacts=right_contacts,
        cube_floor_contacts=floor_contacts,
        total_contacts=(
            left_contacts
            + right_contacts
            + floor_contacts
        ),
    )

    return GraspStatus(
        contact_snapshot=snapshot,
        consecutive_bilateral_steps=consecutive_steps,
        required_bilateral_steps=required_steps,
        gripper_width=0.04,
        cube_height=0.07,
        lift_height=0.05,
        cube_linear_speed=0.0,
        cube_angular_speed=0.0,
        width_valid=width_valid,
        cube_lifted=cube_lifted,
        cube_stable=cube_stable,
    )


@pytest.mark.parametrize(
    (
        "left_contacts",
        "right_contacts",
        "expected_bilateral",
    ),
    [
        (0, 0, False),
        (1, 0, False),
        (0, 1, False),
        (1, 1, True),
    ],
)
def test_bilateral_contact_requires_both_fingers(
    left_contacts: int,
    right_contacts: int,
    expected_bilateral: bool,
) -> None:
    snapshot = ContactSnapshot(
        left_cube_contacts=left_contacts,
        right_cube_contacts=right_contacts,
        cube_floor_contacts=0,
        total_contacts=(
            left_contacts + right_contacts
        ),
    )

    assert (
        snapshot.bilateral_contact
        is expected_bilateral
    )


def test_floor_contact_is_not_bilateral_contact() -> None:
    snapshot = ContactSnapshot(
        left_cube_contacts=0,
        right_cube_contacts=0,
        cube_floor_contacts=4,
        total_contacts=4,
    )

    assert snapshot.cube_on_floor
    assert not snapshot.bilateral_contact


def test_tracker_reaches_threshold_exactly() -> None:
    snapshot = ContactSnapshot(1, 1, 0, 2)
    tracker = BilateralContactTracker(
        required_steps=3
    )

    assert not tracker.update(snapshot)
    assert not tracker.update(snapshot)
    assert tracker.update(snapshot)
    assert tracker.consecutive_steps == 3


def test_tracker_resets_after_contact_loss() -> None:
    bilateral = ContactSnapshot(1, 1, 0, 2)
    left_only = ContactSnapshot(1, 0, 0, 1)
    tracker = BilateralContactTracker(
        required_steps=3
    )

    tracker.update(bilateral)
    tracker.update(bilateral)
    tracker.update(left_only)

    assert tracker.consecutive_steps == 0
    assert not tracker.persistent_contact


@pytest.mark.parametrize("required_steps", [0, -1])
def test_tracker_rejects_nonpositive_threshold(
    required_steps: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="required_steps",
    ):
        BilateralContactTracker(required_steps)


def test_candidate_can_exist_before_lift() -> None:
    status = make_status(
        floor_contacts=4,
        cube_lifted=False,
    )

    assert status.persistent_contact
    assert status.grasp_candidate
    assert not status.grasp_success
    assert status.reason == "cube_on_floor"


def test_success_requires_lift_stability_and_no_floor() -> None:
    successful = make_status()
    not_lifted = make_status(cube_lifted=False)
    unstable = make_status(cube_stable=False)
    on_floor = make_status(floor_contacts=4)

    assert successful.grasp_success
    assert not not_lifted.grasp_success
    assert not unstable.grasp_success
    assert not on_floor.grasp_success


@pytest.mark.parametrize(
    ("status", "expected_reason"),
    [
        (
            make_status(right_contacts=0),
            "no_bilateral_contact",
        ),
        (
            make_status(consecutive_steps=49),
            "contact_not_persistent",
        ),
        (
            make_status(width_valid=False),
            "gripper_width_invalid",
        ),
        (
            make_status(floor_contacts=4),
            "cube_on_floor",
        ),
        (
            make_status(cube_lifted=False),
            "cube_not_lifted",
        ),
        (
            make_status(cube_stable=False),
            "cube_not_stable",
        ),
        (
            make_status(),
            "none",
        ),
    ],
)
def test_grasp_status_reports_first_failure_reason(
    status: GraspStatus,
    expected_reason: str,
) -> None:
    assert status.reason == expected_reason


def test_real_floor_contact_is_not_a_grasp() -> None:
    scene = PandaScene()
    settle_result = settle_cube(scene)

    assert settle_result.settled

    monitor = GraspMonitor(scene)
    status = monitor.update(scene)

    assert status.contact_snapshot.cube_floor_contacts > 0
    assert status.contact_snapshot.cube_on_floor
    assert not status.contact_snapshot.bilateral_contact
    assert not status.grasp_candidate
    assert not status.grasp_success


def test_monitor_update_does_not_modify_simulation() -> None:
    scene = PandaScene()
    monitor = GraspMonitor(scene)

    qpos_before = scene.data.qpos.copy()
    qvel_before = scene.data.qvel.copy()
    ctrl_before = scene.data.ctrl.copy()
    time_before = float(scene.data.time)

    monitor.update(scene)

    assert np.array_equal(
        scene.data.qpos,
        qpos_before,
    )
    assert np.array_equal(
        scene.data.qvel,
        qvel_before,
    )
    assert np.array_equal(
        scene.data.ctrl,
        ctrl_before,
    )
    assert float(scene.data.time) == time_before


def test_monitor_does_not_count_same_simulation_time_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = PandaScene()
    monitor = GraspMonitor(
        scene,
        contact_duration=(
            3.0 * scene.model.opt.timestep
        ),
    )
    bilateral = ContactSnapshot(1, 1, 0, 2)

    # 隔离时间记忆逻辑：让传感器在同一仿真时刻始终报告双侧接触。
    monkeypatch.setattr(
        monitor.sensor,
        "snapshot",
        lambda data: bilateral,
    )

    first = monitor.update(scene)
    repeated = monitor.update(scene)

    assert first.consecutive_bilateral_steps == 1
    assert repeated.consecutive_bilateral_steps == 1
    assert not repeated.persistent_contact


def test_monitor_reset_clears_contact_history() -> None:
    scene = PandaScene()
    monitor = GraspMonitor(scene)
    bilateral = ContactSnapshot(1, 1, 0, 2)

    monitor.tracker.update(bilateral)
    assert monitor.tracker.consecutive_steps == 1

    monitor.reset(
        scene,
        reference_cube_height=0.123,
    )

    assert monitor.tracker.consecutive_steps == 0
    assert monitor.reference_cube_height == 0.123


@pytest.mark.parametrize(
    "keyword_arguments",
    [
        {"contact_duration": 0.0},
        {"min_grasp_width": -0.01},
        {
            "min_grasp_width": 0.05,
            "max_grasp_width": 0.04,
        },
        {"min_lift_height": 0.0},
        {"max_linear_speed": 0.0},
        {"max_angular_speed": np.inf},
    ],
)
def test_monitor_rejects_invalid_parameters(
    keyword_arguments: dict[str, float],
) -> None:
    scene = PandaScene()

    with pytest.raises(ValueError):
        GraspMonitor(scene, **keyword_arguments)
