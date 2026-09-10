"""Generate the Day 12 Cartesian trajectory tracking figure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "day12"
DATA_PATH = OUTPUT_DIR / "cartesian_trajectory_data.npz"
PNG_PATH = OUTPUT_DIR / "cartesian_trajectory_tracking.png"
PDF_PATH = OUTPUT_DIR / "cartesian_trajectory_tracking.pdf"

# Okabe-Ito colorblind-safe palette.
DESIRED_COLOR = "#009E73"
REFERENCE_COLOR = "#0072B2"
ACTUAL_COLOR = "#D55E00"
WAYPOINT_COLOR = "#2F2F2F"


def configure_style() -> None:
    """Apply compact publication-style Matplotlib defaults."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "DejaVu Serif",
            ],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
            "lines.linewidth": 1.8,
            "lines.markersize": 4,
        }
    )


def load_trajectory_data() -> dict[str, np.ndarray]:
    """Load the trajectory arrays produced by the Day 12 probe."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Trajectory data is missing. Run "
            "examples/day12/cartesian_trajectory_tracking_probe.py "
            "before generating the figure."
        )

    with np.load(DATA_PATH) as data:
        return {
            name: data[name].copy()
            for name in data.files
        }


def prepend_start(
    trajectory: np.ndarray,
    start_position: np.ndarray,
) -> np.ndarray:
    """Add the shared initial position to a sampled trajectory."""

    return np.vstack(
        [start_position, trajectory]
    )


def main() -> None:
    configure_style()
    data = load_trajectory_data()

    time = data["time"]
    start_position = data["start_position"]
    waypoints = data["cartesian_waypoints"]

    desired = prepend_start(
        data["desired_position"],
        start_position,
    )
    reference = prepend_start(
        data["reference_position"],
        start_position,
    )
    actual = prepend_start(
        data["actual_position"],
        start_position,
    )

    # Use millimetres relative to the start so small differences are readable.
    desired_mm = (desired - start_position) * 1000.0
    reference_mm = (reference - start_position) * 1000.0
    actual_mm = (actual - start_position) * 1000.0
    waypoints_mm = (waypoints - start_position) * 1000.0

    plot_time = np.concatenate([[0.0], time])
    geometric_error_mm = np.concatenate(
        [[0.0], data["geometric_error_norm"] * 1000.0]
    )
    dynamic_error_mm = np.concatenate(
        [[0.0], data["dynamic_error_norm"] * 1000.0]
    )
    total_error_mm = np.concatenate(
        [[0.0], data["total_error_norm"] * 1000.0]
    )

    fig = plt.figure(
        figsize=(10.5, 4.2),
        constrained_layout=True,
    )

    trajectory_axis = fig.add_subplot(
        1,
        2,
        1,
        projection="3d",
    )

    trajectory_axis.plot(
        desired_mm[:, 0],
        desired_mm[:, 1],
        desired_mm[:, 2],
        color=DESIRED_COLOR,
        linestyle="--",
        label="Desired Cartesian line",
        zorder=4,
    )
    trajectory_axis.plot(
        reference_mm[:, 0],
        reference_mm[:, 1],
        reference_mm[:, 2],
        color=REFERENCE_COLOR,
        label="Kinematic reference",
        zorder=3,
    )
    trajectory_axis.plot(
        actual_mm[:, 0],
        actual_mm[:, 1],
        actual_mm[:, 2],
        color=ACTUAL_COLOR,
        label="Dynamic actual",
        zorder=2,
    )
    trajectory_axis.scatter(
        waypoints_mm[:, 0],
        waypoints_mm[:, 1],
        waypoints_mm[:, 2],
        color=WAYPOINT_COLOR,
        s=18,
        marker="o",
        label="Waypoints",
        zorder=5,
    )

    trajectory_axis.set_xlabel(r"$\Delta x$ (mm)")
    trajectory_axis.set_ylabel(r"$\Delta y$ (mm)")
    trajectory_axis.set_zlabel(r"$\Delta z$ (mm)")
    trajectory_axis.set_title("(a) End-effector trajectory")
    trajectory_axis.view_init(elev=23, azim=-58)

    combined = np.vstack(
        [desired_mm, reference_mm, actual_mm]
    )
    axis_ranges = np.ptp(combined, axis=0)
    trajectory_axis.set_box_aspect(
        np.maximum(axis_ranges, 1e-6)
    )
    trajectory_axis.legend(loc="upper left")

    error_axis = fig.add_subplot(1, 2, 2)
    error_axis.plot(
        plot_time,
        geometric_error_mm,
        color=DESIRED_COLOR,
        label="Geometric error",
    )
    error_axis.plot(
        plot_time,
        dynamic_error_mm,
        color=REFERENCE_COLOR,
        label="Dynamic error",
    )
    error_axis.plot(
        plot_time,
        total_error_mm,
        color=ACTUAL_COLOR,
        label="Total error",
    )

    # The experiment reaches a Cartesian waypoint every 0.4 seconds.
    waypoint_times = time[19::20]
    for waypoint_time in waypoint_times:
        error_axis.axvline(
            waypoint_time,
            color="#8C8C8C",
            linewidth=0.7,
            linestyle=":",
            alpha=0.55,
            zorder=0,
        )

    error_axis.set_xlabel("Simulation time (s)")
    error_axis.set_ylabel("Position error (mm)")
    error_axis.set_title("(b) Tracking-error decomposition")
    error_axis.set_xlim(0.0, float(time[-1]))
    error_axis.set_ylim(bottom=0.0)
    error_axis.spines["top"].set_visible(False)
    error_axis.spines["right"].set_visible(False)
    error_axis.legend(loc="lower right")

    maximum_total_error = float(
        np.max(total_error_mm)
    )
    error_axis.annotate(
        f"max total = {maximum_total_error:.3f} mm",
        xy=(
            float(time[np.argmax(data["total_error_norm"])]),
            maximum_total_error,
        ),
        xytext=(-88, 18),
        textcoords="offset points",
        fontsize=8,
        color=ACTUAL_COLOR,
        arrowprops={
            "arrowstyle": "->",
            "color": ACTUAL_COLOR,
            "linewidth": 0.8,
        },
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    fig.savefig(PNG_PATH, dpi=300)
    fig.savefig(PDF_PATH)
    plt.close(fig)

    print(f"Figure saved: {PNG_PATH}")
    print(f"Vector figure saved: {PDF_PATH}")


if __name__ == "__main__":
    main()
