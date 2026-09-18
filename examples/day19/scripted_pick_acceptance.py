"""Day 19：固定场景连续五次完整抓取验收。"""

import numpy as np

from panda_mujoco.pick_controller import PickState, run_scripted_pick
from panda_mujoco.simulation import PandaScene


FIXED_SEED = 42
CASE_COUNT = 5
MINIMUM_LIFT_HEIGHT = 0.05
MINIMUM_HOLD_DURATION = 0.5

EXPECTED_STATES = (
    PickState.RESET,
    PickState.MOVE_ABOVE,
    PickState.APPROACH,
    PickState.CLOSE_GRIPPER,
    PickState.VERIFY_CONTACT,
    PickState.LIFT,
    PickState.CHECK_SUCCESS,
    PickState.DONE,
)


def main() -> None:
    """用相同seed和全新场景重复执行五次抓取。"""

    minimum_lift_height = float("inf")
    minimum_hold_duration = float("inf")
    maximum_linear_speed = 0.0
    maximum_angular_speed = 0.0

    print("Day 19 fixed-scene scripted-pick acceptance")
    print()

    for case_index in range(CASE_COUNT):
        # 每次都创建新场景，防止上一回合的qpos、qvel、接触历史
        # 或仿真时间污染下一回合。
        scene = PandaScene()
        result = run_scripted_pick(
            scene,
            seed=FIXED_SEED,
        )
        status = result.final_status
        visited_states = tuple(
            record.state for record in result.records
        )

        # 任务级验收：不仅检查函数返回True，还检查成功的物理证据。
        assert result.success
        assert result.final_state is PickState.DONE
        assert result.failed_stage is None
        assert result.reason == "none"
        assert visited_states == EXPECTED_STATES
        assert all(record.success for record in result.records)

        assert status.contact_snapshot.bilateral_contact
        assert status.persistent_contact
        assert status.grasp_success
        assert not status.contact_snapshot.cube_on_floor
        assert status.lift_height >= MINIMUM_LIFT_HEIGHT
        assert result.hold_duration >= MINIMUM_HOLD_DURATION

        assert np.isfinite(status.gripper_width)
        assert np.isfinite(status.cube_linear_speed)
        assert np.isfinite(status.cube_angular_speed)

        minimum_lift_height = min(
            minimum_lift_height,
            status.lift_height,
        )
        minimum_hold_duration = min(
            minimum_hold_duration,
            result.hold_duration,
        )
        maximum_linear_speed = max(
            maximum_linear_speed,
            status.cube_linear_speed,
        )
        maximum_angular_speed = max(
            maximum_angular_speed,
            status.cube_angular_speed,
        )

        print(
            f"case {case_index + 1}/{CASE_COUNT} | "
            f"state={result.final_state.value:<4} | "
            f"lift={status.lift_height * 1000.0:8.3f} mm | "
            f"hold={result.hold_duration:.3f} s | "
            f"width={status.gripper_width * 1000.0:7.3f} mm | "
            f"linear={status.cube_linear_speed:.3e} m/s | "
            f"angular={status.cube_angular_speed:.3e} rad/s"
        )

    print("\nSummary")
    print(f"success count: {CASE_COUNT}/{CASE_COUNT}")
    print("minimum lift height:", minimum_lift_height, "m")
    print("minimum hold duration:", minimum_hold_duration, "s")
    print("maximum linear speed:", maximum_linear_speed, "m/s")
    print("maximum angular speed:", maximum_angular_speed, "rad/s")
    print("\nDay 19 fixed-scene acceptance: PASSED")


if __name__ == "__main__":
    main()
