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
REQUIRED_VIS_COLUMNS = (ALTITUDE_COLUMN, SPEED_COLUMN)


def plot_segmented_flight(df, csv_path, overlay=None, legend_title="Flight phase"):
    """Plot altitude, speed, and vertical speed by timestamp row."""
    df = df.copy()
    df["plot_x"] = range(len(df))

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    seen_phases = set()

    plot_phase_series(
        axes[0],
        df,
        ALTITUDE_COLUMN,
        "Altitude (meters)",
        seen_phases,
    )
    plot_phase_series(
        axes[1],
        df,
        SPEED_COLUMN,
        "Speed (km/h)",
        seen_phases,
    )
    plot_phase_series(
        axes[2],
        df,
        VERTICAL_SPEED_COLUMN,
        "Vertical speed (m/s)",
        seen_phases,
    )
    axes[2].axhline(0, color="#111827", linewidth=1, alpha=0.6)

    if overlay is not None:
        overlay(axes, df)

    def format_time_tick(value, position):
        """Show the nearest row's timestamp label on an evenly spaced axis."""
        row_idx = int(round(value))
        if row_idx < 0 or row_idx >= len(df):
            return ""

        row = df.iloc[row_idx]
        return f"{row['Time_local']:%Y-%m-%d %H:%M}\n({int(row['timestamp'])})"

    axes[2].set_xlim(-0.5, len(df) - 0.5)
    axes[2].xaxis.set_major_locator(mticker.MaxNLocator(nbins=24, integer=True))
    axes[2].xaxis.set_major_formatter(mticker.FuncFormatter(format_time_tick))
    axes[2].xaxis.set_minor_locator(mticker.MaxNLocator(nbins=48, integer=True))
    axes[2].tick_params(axis="x", which="major", labelsize=5, labelrotation=45)
    axes[2].tick_params(axis="x", which="minor", length=3)
    for label in axes[2].get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_verticalalignment("top")

    axes[2].set_xlabel("Timestamp rows, evenly spaced (Asia/Manila)")
    axes[0].legend(title=legend_title)
    fig.suptitle(f"Segmented Flight Altitude, Speed, and Vertical Speed: {csv_path.name}")
    plt.tight_layout()


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
            "Usage: python3 visualization/vis_altspdvspd_flightsegmentation.py <csv-file-or-directory> [...]",
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
                print(f"  Skipping {csv_path.name}: no valid altitude/speed data")
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
