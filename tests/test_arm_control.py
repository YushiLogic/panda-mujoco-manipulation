"""Tests for Panda arm actuator control."""

import numpy as np
import pytest

from panda_mujoco.arm_control import (
    apply_arm_bias_compensation,
    command_arm_joint_positions,
)
from panda_mujoco.simulation import PandaScene


def test_joint_command_writes_ctrl_without_changing_state() -> None:
    """发送命令只应改变 ctrl，不应瞬间改变 qpos 或时间。"""
    scene = PandaScene()

    qpos_before = scene.data.qpos.copy()
    time_before = scene.data.time

    target = scene.data.qpos[:7].copy()
    target[0] = 0.1

    command_arm_joint_positions(
        scene,
        target,
    )

    np.testing.assert_allclose(
        scene.data.ctrl[:7],
        target,
    )

    np.testing.assert_array_equal(
        scene.data.qpos,
        qpos_before,
    )

    assert scene.data.time == pytest.approx(
        time_before
    )


def test_joint_command_rejects_invalid_shape() -> None:
    """关节目标必须包含7个数。"""
    scene = PandaScene()

    with pytest.raises(ValueError, match="shape"):
        command_arm_joint_positions(
            scene,
            np.zeros(6),
        )


def test_joint_command_rejects_nonfinite_values() -> None:
    """关节目标不能包含 NaN 或无穷大。"""
    scene = PandaScene()

    target = scene.data.qpos[:7].copy()
    target[2] = np.nan

    with pytest.raises(ValueError, match="finite"):
        command_arm_joint_positions(
            scene,
            target,
        )


def test_joint_command_rejects_out_of_range_target() -> None:
    """目标角不能超过执行器允许的控制范围。"""
    scene = PandaScene()

    target = scene.data.qpos[:7].copy()

    joint1_upper_limit = (
        scene.model.actuator_ctrlrange[0, 1]
    )

    target[0] = joint1_upper_limit + 0.1

    with pytest.raises(
        ValueError,
        match="control ranges",
    ):
        command_arm_joint_positions(
            scene,
            target,
        )


def test_bias_compensation_copies_current_bias_forces() -> None:
    """补偿函数应复制当前七个手臂自由度的偏置力。"""
    scene = PandaScene()

    time_before = scene.data.time
    qpos_before = scene.data.qpos.copy()

    expected_bias = (
        scene.data.qfrc_bias[:7].copy()
    )

    apply_arm_bias_compensation(scene)

    np.testing.assert_allclose(
        scene.data.qfrc_applied[:7],
        expected_bias,
    )

    # 补偿函数本身不应推进时间或修改位置。
    assert scene.data.time == pytest.approx(
        time_before
    )

    np.testing.assert_array_equal(
        scene.data.qpos,
        qpos_before,
    )