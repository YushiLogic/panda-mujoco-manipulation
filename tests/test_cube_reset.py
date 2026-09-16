"""Day 16：方块复位与随机化的自动测试。"""

import numpy as np
import pytest

from panda_mujoco.cube_reset import (
    DEFAULT_SPAWN_HEIGHT,
    DEFAULT_X_RANGE,
    DEFAULT_Y_RANGE,
    DEFAULT_YAW_RANGE,
    CubePose,
    get_cube_joint_addresses,
    reset_cube,
    sample_cube_pose,
    settle_cube,
    yaw_to_quaternion,
)
from panda_mujoco.simulation import PandaScene


def test_cube_joint_addresses_match_model() -> None:
    """方块free joint地址应与当前模型一致。"""

    scene = PandaScene()

    qpos_address, qvel_address = (
        get_cube_joint_addresses(
            scene.model
        )
    )

    assert qpos_address == 9
    assert qvel_address == 9


def test_zero_yaw_gives_identity_quaternion() -> None:
    """零yaw应产生单位四元数wxyz。"""

    quaternion = yaw_to_quaternion(0.0)

    np.testing.assert_allclose(
        quaternion,
        np.array([1.0, 0.0, 0.0, 0.0]),
        atol=1e-12,
        rtol=0.0,
    )


def test_ninety_degree_yaw_quaternion() -> None:
    """绕z轴90度时，w和qz应为根号二的一半。"""

    quaternion = yaw_to_quaternion(
        np.deg2rad(90.0)
    )

    expected = np.array(
        [
            np.sqrt(0.5),
            0.0,
            0.0,
            np.sqrt(0.5),
        ]
    )

    np.testing.assert_allclose(
        quaternion,
        expected,
        atol=1e-12,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    "invalid_yaw",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_yaw_rejects_nonfinite_values(
    invalid_yaw: float,
) -> None:
    """NaN和正负无穷不能作为yaw角。"""

    with pytest.raises(
        ValueError,
        match="yaw must be finite",
    ):
        yaw_to_quaternion(invalid_yaw)


def test_same_seed_produces_identical_pose() -> None:
    """相同seed必须产生逐位相同的结果。"""

    first = sample_cube_pose(42)
    repeated = sample_cube_pose(42)

    np.testing.assert_array_equal(
        first.position,
        repeated.position,
    )

    np.testing.assert_array_equal(
        first.quaternion,
        repeated.quaternion,
    )

    assert first.yaw == repeated.yaw


def test_different_seeds_change_pose() -> None:
    """不同seed不应总是返回同一位姿。"""

    first = sample_cube_pose(42)
    different = sample_cube_pose(43)

    first_qpos = np.concatenate(
        [
            first.position,
            first.quaternion,
        ]
    )

    different_qpos = np.concatenate(
        [
            different.position,
            different.quaternion,
        ]
    )

    assert not np.array_equal(
        first_qpos,
        different_qpos,
    )


def test_twenty_samples_stay_in_ranges() -> None:
    """前20个seed都必须满足采样边界。"""

    for seed in range(20):
        pose = sample_cube_pose(seed)

        assert (
            DEFAULT_X_RANGE[0]
            <= pose.position[0]
            <= DEFAULT_X_RANGE[1]
        )

        assert (
            DEFAULT_Y_RANGE[0]
            <= pose.position[1]
            <= DEFAULT_Y_RANGE[1]
        )

        assert (
            pose.position[2]
            == DEFAULT_SPAWN_HEIGHT
        )

        assert (
            DEFAULT_YAW_RANGE[0]
            <= pose.yaw
            <= DEFAULT_YAW_RANGE[1]
        )

        assert np.isclose(
            np.linalg.norm(
                pose.quaternion
            ),
            1.0,
            atol=1e-12,
            rtol=0.0,
        )


def test_reset_writes_pose_and_clears_velocity() -> None:
    """reset应写入方块，同时不影响机械臂和时间。"""

    scene = PandaScene()

    qpos_address, qvel_address = (
        get_cube_joint_addresses(
            scene.model
        )
    )

    arm_qpos_before = (
        scene.data.qpos[:9].copy()
    )
    ctrl_before = (
        scene.data.ctrl.copy()
    )
    time_before = float(
        scene.data.time
    )

    # 故意污染方块的六维速度。
    scene.data.qvel[
        qvel_address:qvel_address + 6
    ] = 9.0

    pose = sample_cube_pose(42)
    reset_cube(scene, pose)

    expected_qpos = np.concatenate(
        [
            pose.position,
            pose.quaternion,
        ]
    )

    np.testing.assert_array_equal(
        scene.data.qpos[
            qpos_address:qpos_address + 7
        ],
        expected_qpos,
    )

    np.testing.assert_array_equal(
        scene.data.qvel[
            qvel_address:qvel_address + 6
        ],
        np.zeros(6),
    )

    np.testing.assert_array_equal(
        scene.data.qpos[:9],
        arm_qpos_before,
    )

    np.testing.assert_array_equal(
        scene.data.ctrl,
        ctrl_before,
    )

    np.testing.assert_allclose(
        scene.data.body("cube").xpos,
        pose.position,
        atol=1e-12,
        rtol=0.0,
    )

    assert scene.data.time == time_before


@pytest.mark.parametrize(
    "invalid_pose",
    [
        CubePose(
            position=np.zeros(2),
            quaternion=np.array(
                [1.0, 0.0, 0.0, 0.0]
            ),
            yaw=0.0,
        ),
        CubePose(
            position=np.array(
                [0.45, 0.0, np.nan]
            ),
            quaternion=np.array(
                [1.0, 0.0, 0.0, 0.0]
            ),
            yaw=0.0,
        ),
        CubePose(
            position=np.array(
                [0.45, 0.0, 0.05]
            ),
            quaternion=np.zeros(3),
            yaw=0.0,
        ),
        CubePose(
            position=np.array(
                [0.45, 0.0, 0.05]
            ),
            quaternion=np.array(
                [np.inf, 0.0, 0.0, 0.0]
            ),
            yaw=0.0,
        ),
        CubePose(
            position=np.array(
                [0.45, 0.0, 0.05]
            ),
            quaternion=np.array(
                [2.0, 0.0, 0.0, 0.0]
            ),
            yaw=0.0,
        ),
        CubePose(
            position=np.array(
                [0.45, 0.0, 0.05]
            ),
            quaternion=np.array(
                [1.0, 0.0, 0.0, 0.0]
            ),
            yaw=np.nan,
        ),
    ],
    ids=[
        "wrong-position-shape",
        "nonfinite-position",
        "wrong-quaternion-shape",
        "nonfinite-quaternion",
        "nonunit-quaternion",
        "nonfinite-yaw",
    ],
)
def test_reset_rejects_invalid_pose(
    invalid_pose: CubePose,
) -> None:
    """非法位姿不能写入MuJoCo状态。"""

    scene = PandaScene()

    with pytest.raises(ValueError):
        reset_cube(
            scene,
            invalid_pose,
        )


def test_cube_falls_and_settles_on_floor() -> None:
    """方块应从生成高度落到地面并稳定。"""

    scene = PandaScene()
    pose = sample_cube_pose(42)

    reset_cube(scene, pose)

    result = settle_cube(
        scene,
        duration=1.0,
    )

    assert result.settled
    assert result.steps > 0

    assert np.isclose(
        result.final_position[2],
        0.02,
        atol=5e-4,
        rtol=0.0,
    )

    assert result.linear_speed <= 1e-4
    assert result.angular_speed <= 1e-3

    assert np.all(
        np.isfinite(
            result.final_position
        )
    )


@pytest.mark.parametrize(
    "keyword_arguments",
    [
        {"duration": 0.0},
        {"duration": np.nan},
        {"linear_speed_tolerance": 0.0},
        {"angular_speed_tolerance": -1.0},
    ],
)
def test_settle_rejects_invalid_parameters(
    keyword_arguments: dict,
) -> None:
    """落稳时间和速度阈值必须是有限正数。"""

    scene = PandaScene()

    with pytest.raises(ValueError):
        settle_cube(
            scene,
            **keyword_arguments,
        )