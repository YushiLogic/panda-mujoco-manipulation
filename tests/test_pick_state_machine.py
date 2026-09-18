"""Day 19 脚本抓取状态机测试。"""

import numpy as np
import pytest

import panda_mujoco.pick_controller as pick_controller
from panda_mujoco.grasp_task import GraspTargets
from panda_mujoco.gripper import open_gripper
from panda_mujoco.pick_controller import PickState, run_scripted_pick
from panda_mujoco.simulation import PandaScene


SUCCESS_STATES = (
    PickState.RESET,
    PickState.MOVE_ABOVE,
    PickState.APPROACH,
    PickState.CLOSE_GRIPPER,
    PickState.VERIFY_CONTACT,
    PickState.LIFT,
    PickState.CHECK_SUCCESS,
    PickState.DONE,
)


def test_fixed_scene_pick_reaches_done() -> None:
    """真实动力学中的固定场景应完成完整抓取。"""

    result = run_scripted_pick(
        PandaScene(),
        seed=42,
    )

    assert result.success
    assert result.final_state is PickState.DONE
    assert result.failed_stage is None
    assert result.reason == "none"
    assert tuple(
        record.state for record in result.records
    ) == SUCCESS_STATES

    assert result.final_status.grasp_success
    assert result.final_status.lift_height >= 0.05
    assert result.hold_duration >= 0.5


def test_short_close_timeout_reports_contact_failure() -> None:
    """夹爪尚未接触方块时，不允许状态机进入抬升阶段。"""

    result = run_scripted_pick(
        PandaScene(),
        seed=42,
        close_timeout=0.002,
    )

    assert not result.success
    assert result.final_state is PickState.FAILED
    assert result.failed_stage is PickState.VERIFY_CONTACT
    assert result.reason == "no_bilateral_contact"

    visited_states = tuple(
        record.state for record in result.records
    )
    assert PickState.LIFT not in visited_states
    assert PickState.DONE not in visited_states


def test_releasing_cube_during_lift_reports_failure() -> None:
    """抬升途中张开夹爪时，落在地面的方块不能被误判为成功。"""

    def release_after_lift_starts(
        callback_scene: PandaScene,
    ) -> None:
        # 方块落稳高度约为0.02 m。超过0.06 m说明已经开始抬升，
        # 此时故意张开夹爪，模拟物体滑落或控制命令错误。
        if (
            callback_scene.data.body("cube").xpos[2]
            > 0.06
        ):
            open_gripper(callback_scene)

    result = run_scripted_pick(
        PandaScene(),
        seed=42,
        step_callback=release_after_lift_starts,
    )

    assert not result.success
    assert result.final_state is PickState.FAILED
    assert result.failed_stage is PickState.LIFT
    assert result.reason == "no_bilateral_contact"
    assert not result.final_status.grasp_success


def test_unreachable_pregrasp_reports_ik_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """不可达目标应在MOVE_ABOVE失败，而不是继续执行后续动作。"""

    def unreachable_targets(
        cube_position: np.ndarray,
        *,
        cube_yaw: float = 0.0,
    ) -> GraspTargets:
        del cube_position, cube_yaw
        return GraspTargets(
            pregrasp_position=np.array(
                [5.0, 5.0, 5.0]
            ),
            grasp_position=np.array(
                [5.0, 5.0, 4.9]
            ),
            lift_position=np.array(
                [5.0, 5.0, 5.1]
            ),
            rotation=np.eye(3),
        )

    monkeypatch.setattr(
        pick_controller,
        "generate_grasp_targets",
        unreachable_targets,
    )

    result = run_scripted_pick(
        PandaScene(),
        seed=42,
    )

    assert not result.success
    assert result.final_state is PickState.FAILED
    assert result.failed_stage is PickState.MOVE_ABOVE
    assert result.reason == "ik_failed"

    assert tuple(
        record.state for record in result.records
    ) == (
        PickState.RESET,
        PickState.MOVE_ABOVE,
    )


@pytest.mark.parametrize(
    ("keyword_arguments"),
    [
        {"close_timeout": 0.0},
        {"close_timeout": -1.0},
        {"close_timeout": np.inf},
        {"hold_duration": 0.0},
        {"hold_duration": -1.0},
        {"hold_duration": np.nan},
    ],
)
def test_invalid_timing_parameters_are_rejected(
    keyword_arguments: dict[str, float],
) -> None:
    """状态机不接受非正数以及NaN/Inf计时参数。"""

    with pytest.raises(ValueError):
        run_scripted_pick(
            PandaScene(),
            **keyword_arguments,
        )
