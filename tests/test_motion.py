"""Day17运动原语测试。"""

import numpy as np
import pytest

from panda_mujoco.grasp_task import (
    generate_grasp_targets,
)
from panda_mujoco.kinematics import get_ee_pose
from panda_mujoco.motion import (
    execute_joint_trajectory,
    linear_joint_trajectory,
    move_end_effector_to,
    plan_pose_target,
)
from panda_mujoco.simulation import PandaScene


def test_linear_joint_trajectory_values() -> None:
    """线性轨迹应包含正确的起点、中点和终点。"""

    start = np.zeros(7)

    target = np.array(
        [0.4, -0.4, 0.2, 0.0, 0.0, 0.0, 0.0]
    )

    trajectory = linear_joint_trajectory(
        start,
        target,
        command_count=5,
    )

    assert trajectory.shape == (5, 7)

    np.testing.assert_allclose(
        trajectory[0],
        start,
    )

    np.testing.assert_allclose(
        trajectory[2],
        0.5 * (start + target),
    )

    np.testing.assert_allclose(
        trajectory[-1],
        target,
    )


def test_linear_trajectory_does_not_modify_inputs() -> None:
    """轨迹生成不能修改调用者传入的数组。"""

    start = np.zeros(7)
    target = np.ones(7)

    start_before = start.copy()
    target_before = target.copy()

    linear_joint_trajectory(
        start,
        target,
        command_count=5,
    )

    np.testing.assert_array_equal(
        start,
        start_before,
    )

    np.testing.assert_array_equal(
        target,
        target_before,
    )


@pytest.mark.parametrize(
    "invalid_count",
    [1, 0, -1, 2.5, True],
)
def test_invalid_command_count_is_rejected(
    invalid_count,
) -> None:
    """控制点数量必须是至少为2的整数。"""

    with pytest.raises(
        ValueError,
        match="command_count",
    ):
        linear_joint_trajectory(
            np.zeros(7),
            np.ones(7),
            command_count=invalid_count,
        )


def test_pose_planning_does_not_modify_control_scene() -> None:
    """IK规划不能改变实际控制场景。"""

    scene = PandaScene()

    qpos_before = scene.data.qpos.copy()
    qvel_before = scene.data.qvel.copy()
    ctrl_before = scene.data.ctrl.copy()
    time_before = float(scene.data.time)

    targets = generate_grasp_targets(
        np.array([0.45, 0.0, 0.02]),
        cube_yaw=np.deg2rad(20.0),
    )

    plan = plan_pose_target(
        scene,
        targets.pregrasp_position,
        targets.rotation,
    )

    assert plan.success
    assert plan.reason == "none"
    assert plan.position_error_norm < 1e-4
    assert plan.orientation_error_norm < 1e-3

    np.testing.assert_array_equal(
        scene.data.qpos,
        qpos_before,
    )

    np.testing.assert_array_equal(
        scene.data.qvel,
        qvel_before,
    )

    np.testing.assert_array_equal(
        scene.data.ctrl,
        ctrl_before,
    )

    assert scene.data.time == time_before


def test_execute_joint_trajectory_uses_dynamics() -> None:
    """轨迹执行应推进时间并使实际关节接近目标。"""

    scene = PandaScene()

    start = scene.data.qpos[:7].copy()
    target = start.copy()
    target[0] += 0.1

    trajectory = linear_joint_trajectory(
        start,
        target,
        command_count=21,
    )

    qpos_before = scene.data.qpos.copy()

    duration = execute_joint_trajectory(
        scene,
        trajectory,
        physics_steps_per_command=5,
        settle_steps=250,
    )

    expected_duration = (
        (21 * 5 + 250)
        * scene.model.opt.timestep
    )

    assert np.isclose(
        duration,
        expected_duration,
        atol=1e-12,
    )

    assert not np.array_equal(
        scene.data.qpos,
        qpos_before,
    )

    assert abs(
        scene.data.qpos[0] - target[0]
    ) < 5e-4


def test_move_end_effector_to_small_offset() -> None:
    """完整运动原语应到达一个邻近末端目标。"""

    scene = PandaScene()

    start_position, start_rotation = get_ee_pose(
        scene
    )

    target_position = (
        start_position
        + np.array([0.01, -0.005, 0.01])
    )

    result = move_end_effector_to(
        scene,
        target_position,
        start_rotation,
        command_count=51,
        physics_steps_per_command=5,
        settle_steps=250,
    )

    assert result.success
    assert result.reason == "none"
    assert result.position_error_norm < 0.002
    assert result.orientation_error_norm < np.deg2rad(1.0)
    assert not result.joint_limit_violation
    assert np.all(
        np.isfinite(result.final_position)
    )
    assert np.all(
        np.isfinite(result.final_rotation)
    )