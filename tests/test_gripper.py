"""测试夹爪控制工具。"""
import mujoco
import pytest

# 现在 Python 会在上述目录中寻找 gripper.py

from panda_mujoco.gripper import (
    close_gripper,
    command_gripper,
    get_gripper_width,
    open_gripper,
    width_to_ctrl,
)
from panda_mujoco.simulation import PandaScene
def test_zero_width_maps_to_closed():
    result = width_to_ctrl(0.0)

    assert result == pytest.approx(0.0)


def test_half_width_maps_to_half_ctrl():
    result = width_to_ctrl(0.04)

    assert result == pytest.approx(127.5)


def test_max_width_maps_to_open():
    result = width_to_ctrl(0.08)

    assert result == pytest.approx(255.0)


def test_negative_width_is_rejected():
    with pytest.raises(ValueError):
        width_to_ctrl(-0.01)


def test_too_large_width_is_rejected():
    with pytest.raises(ValueError):
        width_to_ctrl(0.09)

def test_command_gripper_writes_correct_ctrl():
    scene = PandaScene()

    command_gripper(scene, 0.04)

    actuator_id = scene.model.actuator("actuator8").id
    actual_ctrl = scene.data.ctrl[actuator_id]

    assert actual_ctrl == pytest.approx(127.5)

def test_home_gripper_width_is_8cm():
    scene = PandaScene()

    actual_width = get_gripper_width(scene)

    assert actual_width == pytest.approx(0.08)
def test_command_does_not_instantly_change_width():
    scene = PandaScene()

    width_before = get_gripper_width(scene)

    command_gripper(scene, 0.04)

    width_after = get_gripper_width(scene)

    assert width_before == pytest.approx(0.08)
    assert width_after == pytest.approx(0.08)
def test_gripper_reaches_half_width_after_stepping():
    scene = PandaScene()

    command_gripper(scene, 0.04)

    physics_dt = scene.model.opt.timestep
    number_of_steps = int(1.0 / physics_dt)

    for _ in range(number_of_steps):
        mujoco.mj_step(scene.model, scene.data)

    actual_width = get_gripper_width(scene)

    assert actual_width == pytest.approx(0.04, abs=0.0005)
def test_open_gripper_writes_open_ctrl():
    scene = PandaScene()
    actuator_id = scene.model.actuator("actuator8").id

    # 先故意污染控制量，证明 open_gripper 确实进行了修改。
    scene.data.ctrl[actuator_id] = 123.0

    open_gripper(scene)

    assert scene.data.ctrl[actuator_id] == pytest.approx(255.0)


def test_close_gripper_writes_closed_ctrl():
    scene = PandaScene()
    actuator_id = scene.model.actuator("actuator8").id

    close_gripper(scene)

    assert scene.data.ctrl[actuator_id] == pytest.approx(0.0)