import os
import random
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
PREPROCESS_DIR = ROOT_DIR / "1_preprocess"
for import_dir in (ROOT_DIR, PREPROCESS_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

cache_dir = Path(tempfile.gettempdir()) / "flight-segmentation-cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from flight_segmentation import get_smooth_window
from flight_segmentation import get_vertical_speed
from flight_segmentation import load_flight
from flight_segmentation import phase_summary
from flight_segmentation import segment_flight_phases
from flight_segmentation import smooth_altitude
from visualization_helpers import mark_airport_points
from visualization_helpers import mark_ground_endpoints


PHASE_COLORS = {
    "Ground": "#6b7280",
    "Takeoff": "#f97316",
    "Initial Climb": "#facc15",
    "Climb": "#16a34a",
    "Cruise": "#2563eb",
    "Descent": "#dc2626",
    "Approach": "#7c3aed",
}

LEVEL_COLORS = {
    "Climb": "#86efac",
    "Descent": "#f9a8d4",
    3: "#86efac",
    5: "#f9a8d4",
}
MAX_RANDOM_FILES = 50
ALTITUDE_COLUMN = "altitude_meters"
SPEED_COLUMN = "speed_kmh"
VERTICAL_SPEED_COLUMN = "verticalSpeed_ms"
POSITION_COLUMNS = ("latitude", "longitude")
REQUIRED_VIS_COLUMNS = (ALTITUDE_COLUMN, SPEED_COLUMN, *POSITION_COLUMNS)


def plot_segmented_flight(df, csv_path, overlay=None, legend_title="Flight phase"):
    """Plot flight time-series panels and the position path together."""
    df = df.copy()
    df["plot_x"] = range(len(df))

    fig = plt.figure(figsize=(16, 12))
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.25, 1.0],
        wspace=0.24,
        hspace=0.18,
        left=0.06,
        right=0.96,
        bottom=0.10,
        top=0.86,
    )
    altitude_ax = fig.add_subplot(grid[0, 0])
    speed_ax = fig.add_subplot(grid[1, 0], sharex=altitude_ax)
    vertical_speed_ax = fig.add_subplot(grid[2, 0], sharex=altitude_ax)
    position_ax = fig.add_subplot(grid[:, 1])
    axes = [altitude_ax, speed_ax, vertical_speed_ax]
    seen_phases = set()

    plot_phase_series(
        altitude_ax,
        df,
        ALTITUDE_COLUMN,
        "Altitude (meters)",
        seen_phases,
    )
    plot_phase_series(
        speed_ax,
        df,
        SPEED_COLUMN,
        "Speed (km/h)",
        seen_phases,
    )
    plot_phase_series(
        vertical_speed_ax,
        df,
        VERTICAL_SPEED_COLUMN,
        "Vertical speed (m/s)",
        seen_phases,
    )
    vertical_speed_ax.axhline(0, color="#111827", linewidth=1, alpha=0.6)
    plot_position_path(position_ax, df, legend_title)

    if overlay is not None:
        overlay(axes, df)

    def format_time_tick(value, position):
        """Show the nearest row's timestamp label on an evenly spaced axis."""
        row_idx = int(round(value))
        if row_idx < 0 or row_idx >= len(df):
            return ""

        row = df.iloc[row_idx]
        return f"{row['Time_local']:%Y-%m-%d %H:%M}\n({int(row['timestamp'])})"

    vertical_speed_ax.set_xlim(-0.5, len(df) - 0.5)
    vertical_speed_ax.xaxis.set_major_locator(
        mticker.MaxNLocator(nbins=24, integer=True)
    )
    vertical_speed_ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(format_time_tick)
    )
    vertical_speed_ax.xaxis.set_minor_locator(
        mticker.MaxNLocator(nbins=48, integer=True)
    )
    vertical_speed_ax.tick_params(
        axis="x",
        which="major",
        labelsize=5,
        labelrotation=45,
    )
    vertical_speed_ax.tick_params(axis="x", which="minor", length=3)
    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)
    for label in vertical_speed_ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_verticalalignment("top")

    vertical_speed_ax.set_xlabel("Timestamp rows, evenly spaced (Asia/Manila)")
    fig.suptitle(
        f"Segmented Flight Altitude, Speed, Vertical Speed, and Path: {csv_path.name}",
        y=0.98,
    )
    add_horizontal_legend(fig, position_ax, legend_title)


def add_horizontal_legend(fig, ax, legend_title):
    handles, labels = ax.get_legend_handles_labels()
    legend_items = [
        (handle, label)
        for handle, label in zip(handles, labels)
        if label and not label.startswith("_")
    ]
    if not legend_items:
        return

    legend_handles, legend_labels = zip(*legend_items)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=len(legend_labels),
        frameon=True,
    )


def plot_position_path(ax, df, legend_title):
    """Plot longitude vs latitude using the same phase colors as the panels."""
    seen_phases = set()

    for group in contiguous_phase_groups(df):
        phase = group["phase"].iloc[0]
        label = phase if phase not in seen_phases else "_nolegend_"
        ax.plot(
            group["longitude"],
            group["latitude"],
            color=PHASE_COLORS.get(phase, "black"),
            linewidth=2,
            marker=".",
            markersize=3,
            label=label,
        )
        seen_phases.add(phase)

    mark_ground_endpoints(ax, df, "longitude", "latitude")
    mark_airport_points(ax)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Flight path")
    ax.grid(True)
    ax.set_aspect("equal", adjustable="datalim")


def plot_phase_series(ax, df, column, ylabel, seen_phases):
    """Plot one time-series panel with the shared phase coloring."""
    highlight_level_phases(ax, df)

    for group in contiguous_phase_groups(df):
        phase = group["phase"].iloc[0]
        label = phase if phase not in seen_phases else "_nolegend_"
        ax.plot(
            group["plot_x"],
            group[column],
            color=PHASE_COLORS.get(phase, "black"),
            linewidth=2,
            label=label,
        )
        seen_phases.add(phase)

    mark_ground_endpoints(ax, df, "plot_x", column)

    ax.set_ylabel(ylabel)
    ax.grid(True)


def contiguous_phase_groups(df):
    """Yield contiguous chunks so each colored line segment keeps the phase color."""
    group_id = df["phase"].ne(df["phase"].shift()).cumsum()
    for _, group in df.groupby(group_id):
        yield group


def highlight_level_phases(ax, df):
    """Add translucent spans behind level-flight climb/descent segments."""
    if "is_level" not in df.columns:
        return

    level_group_id = df["is_level"].ne(df["is_level"].shift()).cumsum()
    for _, group in df[df["is_level"].eq(1)].groupby(level_group_id):
        phase = group["phase"].iloc[0]
        ax.axvspan(
            group["plot_x"].iloc[0],
            group["plot_x"].iloc[-1],
            color=LEVEL_COLORS.get(phase, "black"),
            alpha=0.24,
            label="_nolegend_",
        )


def add_vertical_speed(df):
    """Add vertical speed from the CSV column, or estimate it from altitude/time."""
    df = df.copy()
    smooth_window = get_smooth_window(len(df))
    smoothed_altitude = smooth_altitude(df[ALTITUDE_COLUMN], smooth_window)
    df[VERTICAL_SPEED_COLUMN] = get_vertical_speed(
        df,
        smoothed_altitude,
        smooth_window,
    )
    return df


def resolve_csv_paths(args):
    csv_paths = []
    for arg in args:
        path = Path(arg)
        if path.is_dir():
            csv_paths.extend(sorted(path.glob("*.csv")))
        else:
            csv_paths.append(path)
    return csv_paths


def main():
    """Process requested CSV files or directories."""
    if len(sys.argv) == 1:
        print(
            "Usage: python3 visualization/vis_altspdvspdpos_flightsegmentation.py <csv-file-or-directory> [...]",
            file=sys.stderr,
        )
        return

    csv_files = resolve_csv_paths(sys.argv[1:])
    if not csv_files:
        print("No CSV files found.", file=sys.stderr)
        return

    if len(csv_files) > MAX_RANDOM_FILES:
        total_files = len(csv_files)
        csv_files = sorted(random.sample(csv_files, MAX_RANDOM_FILES))
        print(f"Randomly selected {len(csv_files)} of {total_files} CSV files:")
        for csv_path in csv_files:
            print(f"  {csv_path.name}")

    for csv_path in csv_files:
        print(f"Processing {csv_path.name}...")
        try:
            df = load_flight(csv_path)
            if df.empty:
                print(f"  Skipping {csv_path.name}: no valid data")
                continue

            missing = [
                column for column in REQUIRED_VIS_COLUMNS if column not in df.columns
            ]
            if missing:
                print(f"  Skipping {csv_path.name}: missing columns {missing}")
                continue

            for column in REQUIRED_VIS_COLUMNS:
                df[column] = pd.to_numeric(df[column], errors="coerce")
            df = df.dropna(subset=REQUIRED_VIS_COLUMNS).reset_index(drop=True)
            if df.empty:
                print(
                    f"  Skipping {csv_path.name}: "
                    "no valid altitude/speed/position data"
                )
                continue

            df = segment_flight_phases(df, label_phases=True)
            df = add_vertical_speed(df)
            df[VERTICAL_SPEED_COLUMN] = pd.to_numeric(
                df[VERTICAL_SPEED_COLUMN],
                errors="coerce",
            )
            df = df.dropna(subset=[VERTICAL_SPEED_COLUMN]).reset_index(drop=True)
            if df.empty:
                print(f"  Skipping {csv_path.name}: no valid vertical speed data")
                continue

            plot_segmented_flight(df, csv_path)
            print(f"  Plotted {csv_path.name} with {len(df)} points")
            print(f"  Phase counts: {phase_summary(df)}")

        except Exception as exc:
            print(f"  Error processing {csv_path.name}: {exc}", file=sys.stderr)

    plt.show()


if __name__ == "__main__":
    main()
