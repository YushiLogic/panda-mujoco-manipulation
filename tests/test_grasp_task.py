"""抓取目标几何的单元测试。"""

import numpy as np
import pytest

from panda_mujoco.grasp_task import (
    TOP_DOWN_GRASP_ROTATION,
    generate_grasp_targets,
)


def test_default_targets_match_expected_geometry() -> None:
    """默认参数应生成预期的三个位置。"""

    cube_position = np.array([0.45, 0.0, 0.02])

    targets = generate_grasp_targets(cube_position)

    np.testing.assert_allclose(
        targets.grasp_position,
        np.array([0.45, 0.0, 0.025]),
    )
    np.testing.assert_allclose(
        targets.pregrasp_position,
        np.array([0.45, 0.0, 0.125]),
    )
    np.testing.assert_allclose(
        targets.lift_position,
        np.array([0.45, 0.0, 0.105]),
    )


def test_target_height_order_is_correct() -> None:
    """预抓取点和抬升点都必须高于抓取点。"""

    targets = generate_grasp_targets(
        np.array([0.45, 0.0, 0.02])
    )

    assert (
        targets.pregrasp_position[2]
        > targets.grasp_position[2]
    )
    assert (
        targets.lift_position[2]
        > targets.grasp_position[2]
    )


def test_top_down_rotation_is_valid() -> None:
    """向下抓取姿态必须是合法旋转矩阵。"""

    targets = generate_grasp_targets(
        np.array([0.45, 0.0, 0.02])
    )
    rotation = targets.rotation

    np.testing.assert_allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=1e-12,
    )
    assert np.isclose(
        np.linalg.det(rotation),
        1.0,
        atol=1e-12,
    )

    # 第三列是接近方向，应指向世界-z。
    np.testing.assert_allclose(
        rotation[:, 2],
        np.array([0.0, 0.0, -1.0]),
    )


def test_input_position_is_not_modified() -> None:
    """目标计算不能修改调用者提供的方块位置。"""

    cube_position = np.array([0.45, 0.0, 0.02])
    position_before = cube_position.copy()

    generate_grasp_targets(cube_position)

    np.testing.assert_array_equal(
        cube_position,
        position_before,
    )


def test_returned_arrays_are_independent() -> None:
    """各目标数组之间不能共享可修改的内存。"""

    targets = generate_grasp_targets(
        np.array([0.45, 0.0, 0.02])
    )

    assert not np.shares_memory(
        targets.pregrasp_position,
        targets.grasp_position,
    )
    assert not np.shares_memory(
        targets.grasp_position,
        targets.lift_position,
    )
    assert not np.shares_memory(
        targets.rotation,
        TOP_DOWN_GRASP_ROTATION,
    )


def test_wrong_cube_position_shape_is_rejected() -> None:
    """方块位置必须恰好包含x、y、z三个数。"""

    with pytest.raises(
        ValueError,
        match="cube_position",
    ):
        generate_grasp_targets(
            np.array([0.45, 0.0])
        )


def test_non_finite_cube_position_is_rejected() -> None:
    """包含NaN或Inf的位置必须被拒绝。"""

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        generate_grasp_targets(
            np.array([0.45, np.nan, 0.02])
        )


@pytest.mark.parametrize(
    "keyword_arguments",
    [
        {"pregrasp_distance": 0.0},
        {"pregrasp_distance": -0.1},
        {"lift_height": 0.0},
        {"lift_height": np.inf},
        {"grasp_z_offset": np.nan},
    ],
)
def test_invalid_geometry_parameters_are_rejected(
    keyword_arguments: dict,
) -> None:
    """非法距离和非有限偏置必须被拒绝。"""

    with pytest.raises(ValueError):
        generate_grasp_targets(
            np.array([0.45, 0.0, 0.02]),
            **keyword_arguments,
        )

def test_grasp_rotation_follows_cube_yaw() -> None:
    """夹爪水平轴必须跟随方块yaw旋转。"""

    cube_yaw = np.deg2rad(30.0)

    targets = generate_grasp_targets(
        np.array([0.45, 0.0, 0.02]),
        cube_yaw=cube_yaw,
    )

    cosine = np.cos(cube_yaw)
    sine = np.sin(cube_yaw)

    cube_rotation = np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # 末端局部+x轴与方块局部+x轴同向。
    np.testing.assert_allclose(
        targets.rotation[:, 0],
        cube_rotation[:, 0],
        atol=1e-12,
    )

    # 末端局部+y轴与方块局部+y轴反向。
    np.testing.assert_allclose(
        targets.rotation[:, 1],
        -cube_rotation[:, 1],
        atol=1e-12,
    )

    # 接近方向仍然竖直向下。
    np.testing.assert_allclose(
        targets.rotation[:, 2],
        np.array([0.0, 0.0, -1.0]),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "invalid_yaw",
    [np.nan, np.inf, -np.inf],
)
def test_non_finite_cube_yaw_is_rejected(
    invalid_yaw: float,
) -> None:
    """NaN和Inf不能作为方块yaw。"""

    with pytest.raises(
        ValueError,
        match="cube_yaw",
    ):
        generate_grasp_targets(
            np.array([0.45, 0.0, 0.02]),
            cube_yaw=invalid_yaw,
        )