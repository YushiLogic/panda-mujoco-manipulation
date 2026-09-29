import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from panda_mujoco.action_mapping import map_action_to_targets


def test_zero_action_keeps_all_targets():
    position = np.array([0.45, 0.0, 0.10])
    rotation = Rotation.from_euler("x", 30, degrees=True).as_matrix()

    next_position, next_rotation, next_width = map_action_to_targets(
        np.zeros(7), position, rotation, 0.04
    )

    assert np.allclose(next_position, position)
    assert np.allclose(next_rotation, rotation)
    assert next_width == pytest.approx(0.04)


def test_action_is_scaled_and_clipped():
    action = np.array([2.0, -0.5, 0, 0, 0, 0.5, -2.0])

    position, rotation, width = map_action_to_targets(
        action, np.array([0.45, 0, 0.10]), np.eye(3), 0.04
    )

    assert np.allclose(position, [0.452, -0.001, 0.10])
    assert np.allclose(
        rotation,
        Rotation.from_euler("z", 1, degrees=True).as_matrix(),
    )
    assert width == pytest.approx(0.038)


@pytest.mark.parametrize(
    "action",
    [np.zeros(6), np.array([0, 0, 0, np.nan, 0, 0, 0])],
)
def test_invalid_action_is_rejected(action):
    with pytest.raises(ValueError):
        map_action_to_targets(action, np.zeros(3), np.eye(3), 0.04)

@pytest.mark.parametrize("axis_index", range(6))
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_six_axis_directions(axis_index, sign):
    start_position = np.array([0.45, 0.0, 0.10])
    start_rotation = Rotation.from_euler(
        "xyz", [15, -10, 30], degrees=True
    ).as_matrix()

    action = np.zeros(7)
    action[axis_index] = sign

    next_position, next_rotation, next_width = map_action_to_targets(
        action, start_position, start_rotation, 0.04
    )

    actual_position_mm = (next_position - start_position) * 1000
    delta_rotation = next_rotation @ start_rotation.T
    actual_rotation_deg = np.rad2deg(
        Rotation.from_matrix(delta_rotation).as_rotvec()
    )

    expected_position_mm = np.zeros(3)
    expected_rotation_deg = np.zeros(3)
    if axis_index < 3:
        expected_position_mm[axis_index] = sign * 2
    else:
        expected_rotation_deg[axis_index - 3] = sign * 2

    assert np.allclose(actual_position_mm, expected_position_mm)
    assert np.allclose(actual_rotation_deg, expected_rotation_deg)
    assert next_width == pytest.approx(0.04)