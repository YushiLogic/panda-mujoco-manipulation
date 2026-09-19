"""Day 20：轻量Panda抓取任务接口测试。"""

from types import SimpleNamespace

import numpy as np
import pytest

from panda_mujoco.panda_pick_task import (
    OBSERVATION_SIZE,
    PandaPickTask,
    PickAction,
    classify_pick_failure,
    get_pick_observation,
)
from panda_mujoco.pick_controller import PickState
from panda_mujoco.simulation import PandaScene


def test_observation_has_fixed_finite_shape() -> None:
    """任何合法场景都应产生固定29维有限数观察。"""

    observation = get_pick_observation(PandaScene())

    assert observation.shape == (OBSERVATION_SIZE,)
    assert np.all(np.isfinite(observation))


def test_observation_is_an_independent_array() -> None:
    """修改返回的观察不能反向污染MuJoCo状态。"""

    scene = PandaScene()
    qpos_before = scene.data.qpos.copy()

    observation = get_pick_observation(scene)
    observation[:] = 123.0

    np.testing.assert_array_equal(
        scene.data.qpos,
        qpos_before,
    )


def test_reset_is_reproducible_by_seed() -> None:
    """相同seed的初始观察相同，不同seed应改变随机方块状态。"""

    task = PandaPickTask()

    first, first_info = task.reset(seed=42)
    repeated, repeated_info = task.reset(seed=42)
    different, _ = task.reset(seed=43)

    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, different)
    assert first_info["seed"] == 42
    assert repeated_info["task_state"] == "READY"
    assert repeated_info["episode_steps"] == 0


def test_wait_reaches_time_limit_as_truncation() -> None:
    """外部步数上限应设置truncated，而不是terminated。"""

    task = PandaPickTask(
        max_episode_steps=2,
        wait_physics_steps=1,
    )
    task.reset(seed=42)

    _, reward_1, terminated_1, truncated_1, info_1 = (
        task.step(PickAction.WAIT)
    )
    _, reward_2, terminated_2, truncated_2, info_2 = (
        task.step(PickAction.WAIT)
    )

    assert reward_1 == 0.0
    assert not terminated_1
    assert not truncated_1
    assert info_1["task_state"] == "RUNNING"

    assert reward_2 == 0.0
    assert not terminated_2
    assert truncated_2
    assert info_2["task_state"] == "TRUNCATED"
    assert info_2["reason"] == "time_limit"


def test_wait_calls_callback_after_each_physics_step() -> None:
    """验收器应能在WAIT包含的每一个物理步后读取状态。"""

    task = PandaPickTask(
        max_episode_steps=2,
        wait_physics_steps=4,
    )
    task.reset(seed=42)
    callback_times: list[float] = []

    def record_time(scene: PandaScene) -> None:
        callback_times.append(float(scene.data.time))

    task.step(
        PickAction.WAIT,
        step_callback=record_time,
    )

    assert len(callback_times) == 4
    assert callback_times == sorted(callback_times)
    assert len(set(callback_times)) == 4


def test_scripted_pick_success_is_termination() -> None:
    """脚本抓取成功是明确任务结局，应terminated而非truncated。"""

    task = PandaPickTask()
    task.reset(seed=42)

    observation, reward, terminated, truncated, info = (
        task.step(PickAction.RUN_SCRIPTED_PICK)
    )

    assert observation.shape == (OBSERVATION_SIZE,)
    assert reward == 1.0
    assert terminated
    assert not truncated
    assert info["success"] is True
    assert info["final_state"] == "DONE"
    assert info["failed_stage"] is None
    assert info["failure_category"] == "none"
    assert info["stage_trace"][-1] == "DONE"


def test_scripted_pick_exposes_physics_step_callback() -> None:
    """完整抓取应把内部物理步开放给只读验收回调。"""

    task = PandaPickTask()
    task.reset(seed=42)
    callback_count = 0
    all_states_finite = True

    def inspect_state(scene: PandaScene) -> None:
        nonlocal callback_count, all_states_finite
        callback_count += 1
        all_states_finite = bool(
            all_states_finite
            and np.all(np.isfinite(scene.data.qpos))
            and np.all(np.isfinite(scene.data.qvel))
            and np.all(np.isfinite(scene.data.ctrl))
        )

    _, _, terminated, _, info = task.step(
        PickAction.RUN_SCRIPTED_PICK,
        step_callback=inspect_state,
    )

    assert terminated
    assert info["success"] is True
    assert callback_count > 0
    assert all_states_finite


def test_step_requires_reset_and_new_reset_after_end() -> None:
    """episode开始前和结束后都不能继续调用step。"""

    task = PandaPickTask(
        max_episode_steps=1,
        wait_physics_steps=1,
    )

    with pytest.raises(
        RuntimeError,
        match=r"call reset\(\) before step\(\)",
    ):
        task.step(PickAction.WAIT)

    task.reset(seed=42)
    task.step(PickAction.WAIT)

    with pytest.raises(
        RuntimeError,
        match="episode has ended",
    ):
        task.step(PickAction.WAIT)


@pytest.mark.parametrize(
    "invalid_action",
    [True, -1, 2, "WAIT", None],
)
def test_invalid_actions_are_rejected(
    invalid_action: object,
) -> None:
    """动作必须是PickAction或对应的有效整数。"""

    task = PandaPickTask()
    task.reset(seed=42)

    with pytest.raises(
        ValueError,
        match="valid PickAction",
    ):
        task.step(invalid_action)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("reason", "failed_stage", "expected"),
    [
        (
            "ik_failed",
            PickState.MOVE_ABOVE,
            "planning_or_motion_failure",
        ),
        (
            "no_bilateral_contact",
            PickState.VERIFY_CONTACT,
            "contact_failure",
        ),
        (
            "no_bilateral_contact",
            PickState.LIFT,
            "lift_failure",
        ),
        (
            "cube_not_stable",
            PickState.CHECK_SUCCESS,
            "stability_failure",
        ),
    ],
)
def test_failure_categories_preserve_failed_stage(
    reason: str,
    failed_stage: PickState,
    expected: str,
) -> None:
    """批量统计类别应保留失败发生在哪个流程阶段的信息。"""

    result = SimpleNamespace(
        success=False,
        reason=reason,
        failed_stage=failed_stage,
    )

    assert classify_pick_failure(result) == expected


@pytest.mark.parametrize(
    "keyword_arguments",
    [
        {"max_episode_steps": 0},
        {"max_episode_steps": True},
        {"wait_physics_steps": 0},
        {"wait_physics_steps": 1.5},
    ],
)
def test_invalid_task_configuration_is_rejected(
    keyword_arguments: dict[str, object],
) -> None:
    """episode长度和每动作物理步数都必须是正整数。"""

    with pytest.raises(ValueError):
        PandaPickTask(**keyword_arguments)  # type: ignore[arg-type]
