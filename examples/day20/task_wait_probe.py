"""Day 20：验证WAIT动作与truncated超时语义。"""

from panda_mujoco.panda_pick_task import (
    PandaPickTask,
    PickAction,
)


def main() -> None:
    """连续WAIT三次，并在任务步数上限处验证截断。"""

    task = PandaPickTask(
        max_episode_steps=3,
        wait_physics_steps=10,
    )

    observation, reset_info = task.reset(
        seed=42
    )

    print("Reset")
    print("observation shape:", observation.shape)
    print("task state:", reset_info["task_state"])
    print(
        "simulation time:",
        task.scene.data.time,
        "s",
    )

    for step_index in range(1, 4):
        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = task.step(PickAction.WAIT)

        print(f"\nTask step {step_index}")
        print("action:", info["action"])
        print("reward:", reward)
        print("terminated:", terminated)
        print("truncated:", truncated)
        print("reason:", info["reason"])
        print("task state:", info["task_state"])
        print(
            "simulation time:",
            task.scene.data.time,
            "s",
        )

    assert observation.shape == (29,)
    assert reward == 0.0
    assert not terminated
    assert truncated
    assert info["reason"] == "time_limit"
    assert info["episode_steps"] == 3

    print("\nTask WAIT/truncation probe: PASSED")


if __name__ == "__main__":
    main()
