"""生成固定且可复现的50个末端位置目标。"""

import numpy as np

from panda_mujoco.kinematics import get_ee_position
from panda_mujoco.simulation import PandaScene


RANDOM_SEED = 20260913
TARGET_COUNT = 50

# 相对于Panda home末端位置的采样范围，单位为米。
LOWER_OFFSET = np.array(
    [-0.10, -0.15, -0.20]
)

UPPER_OFFSET = np.array(
    [0.10, 0.15, 0.05]
)


def generate_target_positions(
    home_position: np.ndarray,
    *,
    target_count: int,
    seed: int,
) -> np.ndarray:
    """在固定工作空间范围内生成末端目标位置。"""

    rng = np.random.default_rng(seed)

    offsets = rng.uniform(
        low=LOWER_OFFSET,
        high=UPPER_OFFSET,
        size=(target_count, 3),
    )

    target_positions = (
        home_position[None, :] + offsets
    )

    return target_positions


def main() -> None:
    scene = PandaScene()
    home_position = get_ee_position(scene)

    targets_first = generate_target_positions(
        home_position,
        target_count=TARGET_COUNT,
        seed=RANDOM_SEED,
    )

    # 使用相同seed重新生成一次。
    targets_second = generate_target_positions(
        home_position,
        target_count=TARGET_COUNT,
        seed=RANDOM_SEED,
    )

    # 使用不同seed生成，用于确认结果确实会改变。
    targets_different_seed = generate_target_positions(
        home_position,
        target_count=TARGET_COUNT,
        seed=RANDOM_SEED + 1,
    )

    offsets = (
        targets_first - home_position[None, :]
    )

    np.set_printoptions(
        precision=6,
        suppress=True,
    )

    print("Random seed:")
    print(RANDOM_SEED)

    print()
    print("Home end-effector position:")
    print(home_position)

    print()
    print("First five target positions:")
    print(targets_first[:5])

    print()
    print("Minimum sampled offset:")
    print(np.min(offsets, axis=0))

    print()
    print("Maximum sampled offset:")
    print(np.max(offsets, axis=0))

    print()
    print(
        "Same-seed targets identical:",
        np.array_equal(
            targets_first,
            targets_second,
        ),
    )

    print(
        "Different-seed targets different:",
        not np.array_equal(
            targets_first,
            targets_different_seed,
        ),
    )

    assert targets_first.shape == (
        TARGET_COUNT,
        3,
    )

    assert np.all(np.isfinite(targets_first))

    assert np.all(
        offsets >= LOWER_OFFSET
    )

    assert np.all(
        offsets <= UPPER_OFFSET
    )

    assert np.array_equal(
        targets_first,
        targets_second,
    )

    assert not np.array_equal(
        targets_first,
        targets_different_seed,
    )

    print()
    print("Deterministic target sampling: PASSED")


if __name__ == "__main__":
    main()