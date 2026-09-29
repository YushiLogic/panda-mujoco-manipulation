"""Run reproducible failure-injection checks for the Panda pick pipeline.

These scenarios reuse the project controllers and monitors. Faults are injected
only to verify that the existing state machine stops at the expected stage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import panda_mujoco.pick_controller as pick_controller
from panda_mujoco.contact import ContactSnapshot
from panda_mujoco.continuous_env import PandaContinuousEnv
from panda_mujoco.grasp_task import GraspTargets
from panda_mujoco.gripper import open_gripper
from panda_mujoco.pick_controller import PickResult, PickState, run_scripted_pick
from panda_mujoco.simulation import PandaScene


@dataclass(frozen=True)
class ScenarioResult:
    """Small, consistent summary of one injected scenario."""

    name: str
    passed: bool
    observed: str
    expected: str


def summarize_pick_failure(
    *,
    name: str,
    result: PickResult,
    expected_stage: PickState,
    expected_reason: str,
    forbidden_states: tuple[PickState, ...] = (),
) -> ScenarioResult:
    """Check both the failure diagnosis and that later stages did not run."""

    visited = tuple(record.state for record in result.records)
    stopped_at_expected_stage = result.failed_stage is expected_stage
    reason_is_expected = result.reason == expected_reason
    later_stage_was_skipped = all(
        state not in visited for state in forbidden_states
    )
    passed = (
        not result.success
        and result.final_state is PickState.FAILED
        and stopped_at_expected_stage
        and reason_is_expected
        and later_stage_was_skipped
    )

    trace = " -> ".join(state.value for state in visited)
    observed = (
        f"stage={result.failed_stage.value if result.failed_stage else 'none'}, "
        f"reason={result.reason}, trace={trace}"
    )
    expected = (
        f"stage={expected_stage.value}, reason={expected_reason}, "
        f"skip={[state.value for state in forbidden_states]}"
    )
    return ScenarioResult(name, passed, observed, expected)


def run_unreachable_target() -> ScenarioResult:
    """Inject an unreachable pregrasp pose; later pick stages must not run."""

    original_generator = pick_controller.generate_grasp_targets

    def unreachable_targets(
        cube_position: np.ndarray,
        *,
        cube_yaw: float = 0.0,
    ) -> GraspTargets:
        del cube_position, cube_yaw
        return GraspTargets(
            pregrasp_position=np.array([5.0, 5.0, 5.0]),
            grasp_position=np.array([5.0, 5.0, 4.9]),
            lift_position=np.array([5.0, 5.0, 5.1]),
            rotation=np.eye(3),
        )

    try:
        pick_controller.generate_grasp_targets = unreachable_targets
        result = run_scripted_pick(PandaScene(), seed=42)
    finally:
        # Restore the real planner input even if the injected test raises.
        pick_controller.generate_grasp_targets = original_generator

    return summarize_pick_failure(
        name="unreachable target",
        result=result,
        expected_stage=PickState.MOVE_ABOVE,
        expected_reason="ik_failed",
        forbidden_states=(PickState.APPROACH, PickState.LIFT, PickState.DONE),
    )


def run_no_contact() -> ScenarioResult:
    """Give the gripper too little time to establish bilateral contact."""

    result = run_scripted_pick(
        PandaScene(),
        seed=42,
        close_timeout=0.002,
    )
    return summarize_pick_failure(
        name="no bilateral contact",
        result=result,
        expected_stage=PickState.VERIFY_CONTACT,
        expected_reason="no_bilateral_contact",
        forbidden_states=(PickState.LIFT, PickState.CHECK_SUCCESS, PickState.DONE),
    )


def run_single_sided_contact() -> ScenarioResult:
    """Verify that one finger touching alone is not bilateral contact."""

    snapshot = ContactSnapshot(
        left_cube_contacts=1,
        right_cube_contacts=0,
        cube_floor_contacts=0,
        total_contacts=1,
    )
    passed = not snapshot.bilateral_contact
    return ScenarioResult(
        name="single-sided contact",
        passed=passed,
        observed=f"bilateral_contact={snapshot.bilateral_contact}",
        expected="bilateral_contact=False",
    )


def run_contact_loss_during_lift() -> ScenarioResult:
    """Open the gripper after lift starts to simulate a dropped object."""

    def release_cube_after_lift_starts(scene: PandaScene) -> None:
        # The settled cube is near z=0.02 m. Crossing 0.06 m means lift began.
        if scene.data.body("cube").xpos[2] > 0.06:
            open_gripper(scene)

    result = run_scripted_pick(
        PandaScene(),
        seed=42,
        step_callback=release_cube_after_lift_starts,
    )
    return summarize_pick_failure(
        name="contact loss during lift",
        result=result,
        expected_stage=PickState.LIFT,
        expected_reason="no_bilateral_contact",
        forbidden_states=(PickState.CHECK_SUCCESS, PickState.DONE),
    )


def run_timeout() -> ScenarioResult:
    """Use a one-step episode limit to force a deterministic time truncation."""

    env = PandaContinuousEnv(max_episode_steps=1)
    try:
        env.reset(seed=42)
        _, _, terminated, truncated, info = env.step(
            np.zeros(7, dtype=np.float32)
        )
    finally:
        env.close()

    passed = (
        terminated is False
        and truncated is True
        and info["termination_reason"] == "time_limit"
    )
    observed = (
        f"terminated={terminated}, truncated={truncated}, "
        f"reason={info['termination_reason']}"
    )
    expected = "terminated=False, truncated=True, reason=time_limit"
    return ScenarioResult("time limit", passed, observed, expected)


def main() -> None:
    """Run all scenarios and fail the process if a diagnosis is unexpected."""

    scenarios = (
        run_unreachable_target(),
        run_no_contact(),
        run_single_sided_contact(),
        run_contact_loss_during_lift(),
        run_timeout(),
    )

    print("Day 25 deterministic failure scenarios")
    for scenario in scenarios:
        status = "PASS" if scenario.passed else "FAIL"
        print(f"\n{scenario.name}: {status}")
        print(f"  observed: {scenario.observed}")
        print(f"  expected: {scenario.expected}")

    passed_count = sum(scenario.passed for scenario in scenarios)
    print(f"\nScenario checks: {passed_count}/{len(scenarios)}")
    if passed_count != len(scenarios):
        raise SystemExit(1)
    print("Day 25 failure-scenario check: PASSED")


if __name__ == "__main__":
    main()
