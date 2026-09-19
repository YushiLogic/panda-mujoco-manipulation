"""Day 20：检查任务reset、seed复现性和初始observation。"""

import numpy as np

from panda_mujoco.panda_pick_task import (
    CUBE_POSITION_SLICE,
    CUBE_YAW_SIN_COS_SLICE,
    OBSERVATION_SIZE,
    PandaPickTask,
)


def main() -> None:
    """比较相同seed与不同seed产生的episode初始状态。"""

    task = PandaPickTask()

    # 同一个任务对象连续reset，验证上一回合状态已被彻底清除。
    observation_42_a, info_42_a = task.reset(
        seed=42
    )
    observation_42_b, info_42_b = task.reset(
        seed=42
    )
    observation_43, info_43 = task.reset(
        seed=43
    )

    same_seed_identical = np.array_equal(
        observation_42_a,
        observation_42_b,
    )
    different_seed_different = not np.array_equal(
        observation_42_a,
        observation_43,
    )

    print("Task reset reproducibility")
    print("same seed identical:", same_seed_identical)
    print(
        "different seed different:",
        different_seed_different,
    )

    print("\nSeed 42")
    print(
        "cube position:",
        observation_42_a[CUBE_POSITION_SLICE],
    )
    print(
        "yaw sin/cos:",
        observation_42_a[
            CUBE_YAW_SIN_COS_SLICE
        ],
    )
    print("info:", info_42_a)

    print("\nSeed 43")
    print(
        "cube position:",
        observation_43[CUBE_POSITION_SLICE],
    )
    print(
        "yaw sin/cos:",
        observation_43[
            CUBE_YAW_SIN_COS_SLICE
        ],
    )
    print("info:", info_43)

    print("\nObservation contract")
    print("shape:", observation_42_a.shape)
    print(
        "all finite:",
        np.all(np.isfinite(observation_42_a)),
    )
    print("task state:", info_42_b["task_state"])
    print(
        "episode steps:",
        info_42_b["episode_steps"],
    )

    assert same_seed_identical
    assert different_seed_different
    assert observation_42_a.shape == (
        OBSERVATION_SIZE,
    )
    assert np.all(
        np.isfinite(observation_42_a)
    )
    assert info_42_b["episode_steps"] == 0
    assert info_42_b["task_state"] == "READY"

    print("\nTask reset probe: PASSED")


if __name__ == "__main__":
    main()
