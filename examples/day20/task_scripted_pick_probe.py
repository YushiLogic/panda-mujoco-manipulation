"""Day 20：通过轻量任务接口运行一次Day19脚本抓取。"""

from panda_mujoco.panda_pick_task import (
    PandaPickTask,
    PickAction,
)


def main() -> None:
    task = PandaPickTask()

    initial_observation, reset_info = task.reset(
        seed=42
    )
    time_after_reset = float(
        task.scene.data.time
    )

    (
        final_observation,
        reward,
        terminated,
        truncated,
        info,
    ) = task.step(PickAction.RUN_SCRIPTED_PICK)

    print("Reset result")
    print("initial shape:", initial_observation.shape)
    print("seed:", reset_info["seed"])
    print("state:", reset_info["task_state"])
    print("simulation time:", time_after_reset, "s")

    print("\nScripted-pick task result")
    print("final shape:", final_observation.shape)
    print("reward:", reward)
    print("terminated:", terminated)
    print("truncated:", truncated)

    for key in (
        "success",
        "final_state",
        "failed_stage",
        "reason",
        "failure_category",
        "lift_height",
        "hold_duration",
        "gripper_width",
        "episode_steps",
        "stage_trace",
    ):
        print(f"{key}:", info[key])

    print(
        "simulation time after pick:",
        task.scene.data.time,
        "s",
    )

    assert reward == 1.0
    assert terminated
    assert not truncated
    assert info["success"]
    assert info["final_state"] == "DONE"
    assert info["failed_stage"] is None
    assert info["reason"] == "none"
    assert info["lift_height"] >= 0.05
    assert info["hold_duration"] >= 0.5

    # 控制器使用已经由task.reset准备好的场景，所以其RESET记录耗时为0，
    # 不会再次增加1.5秒方块落稳时间。
    assert info["stage_trace"][0] == "RESET"

    print("\nTask scripted-pick probe: PASSED")


if __name__ == "__main__":
    main()
