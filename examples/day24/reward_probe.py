"""Print reward components for representative task transitions."""

import numpy as np

from panda_mujoco.rewards import compute_reward


def run_case(name: str, **overrides: object) -> None:
    """Evaluate one synthetic transition and print its reward breakdown."""

    transition: dict[str, object] = {
        "previous_reach_error_m": 0.030,
        "current_reach_error_m": 0.030,
        "previous_orientation_error_rad": 0.0,
        "current_orientation_error_rad": 0.0,
        "previous_persistent_contact": False,
        "persistent_contact": False,
        "previous_lift_height_m": 0.0,
        "current_lift_height_m": 0.0,
        "success": False,
        "action": np.zeros(7),
    }
    transition.update(overrides)

    result = compute_reward(**transition)

    print(
        f"{name:20s}"
        f" | reach={result.reach_progress:+.3f}"
        f" | contact={result.contact_bonus:+.3f}"
        f" | lift={result.lift_progress:+.3f}"
        f" | success={result.success_bonus:+.3f}"
        f" | action={result.action_penalty:+.3f}"
        f" | total={result.total:+.3f}"
    )


def main() -> None:
    run_case(
        "move closer",
        current_reach_error_m=0.029,
    )

    run_case(
        "move farther",
        current_reach_error_m=0.033,
    )

    run_case(
        "contact begins",
        persistent_contact=True,
    )

    run_case(
        "contact continues",
        previous_persistent_contact=True,
        persistent_contact=True,
    )

    run_case(
        "cube lifted 2 mm",
        current_lift_height_m=0.002,
    )

    run_case(
        "cube dropped",
        previous_lift_height_m=0.020,
        current_lift_height_m=0.0,
    )

    run_case(
        "task success",
        success=True,
    )


if __name__ == "__main__":
    main()