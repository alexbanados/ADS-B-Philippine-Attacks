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
import numpy as np
import pandas as pd

from flight_segmentation import load_flight
from flight_segmentation import phase_summary
from flight_segmentation import segment_flight_phases
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
HEADING_COLUMN = "heading"
ALTITUDE_COLUMN = "altitude_meters"
REQUIRED_VIS_COLUMNS = (HEADING_COLUMN, ALTITUDE_COLUMN)


def plot_segmented_flight(df, csv_path, overlay=None, legend_title="Flight phase"):
    """Plot altitude by timestamp with arrows showing aircraft heading."""
    df = df.copy()
    df["plot_x"] = range(len(df))

    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    seen_phases = set()

    highlight_level_phases(ax, df)

    for group in contiguous_phase_groups(df):
        phase = group["phase"].iloc[0]
        label = phase if phase not in seen_phases else "_nolegend_"
        ax.plot(
            group["plot_x"],
            group[ALTITUDE_COLUMN],
            color=PHASE_COLORS.get(phase, "black"),
            linewidth=2,
            label=label,
        )
        seen_phases.add(phase)

    mark_ground_endpoints(ax, df, "plot_x", ALTITUDE_COLUMN)
    add_heading_arrows(ax, df)

    if overlay is not None:
        overlay(ax, df)

    def format_time_tick(value, position):
        """Show the nearest row's timestamp label on an evenly spaced axis."""
        row_idx = int(round(value))
        if row_idx < 0 or row_idx >= len(df):
            return ""

        row = df.iloc[row_idx]
        return f"{row['Time_local']:%Y-%m-%d %H:%M}\n({int(row['timestamp'])})"

    ax.set_xlim(-0.5, len(df) - 0.5)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=24, integer=True))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_time_tick))
    ax.xaxis.set_minor_locator(mticker.MaxNLocator(nbins=48, integer=True))
    ax.tick_params(axis="x", which="major", labelsize=5, labelrotation=45)
    ax.tick_params(axis="x", which="minor", length=3)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_verticalalignment("top")
    ax.set_xlabel("Timestamp rows, evenly spaced (Asia/Manila)")
    ax.set_ylabel("Altitude (meters)")
    ax.set_title(f"Segmented Flight Altitude with Heading Arrows: {csv_path.name}")
    ax.grid(True)
    ax.legend(title=legend_title)
    plt.tight_layout()


def contiguous_phase_groups(df):
    """Yield contiguous chunks so each colored line segment keeps the phase color."""
    group_id = df["phase"].ne(df["phase"].shift()).cumsum()
    for _, group in df.groupby(group_id):
        yield group


def add_heading_arrows(ax, df):
    """Draw arrows at nonzero-altitude rows where heading changes."""
    arrow_rows = df[df[ALTITUDE_COLUMN] > 0].copy()
    arrow_rows = arrow_rows[
        arrow_rows[HEADING_COLUMN].ne(arrow_rows[HEADING_COLUMN].shift())
    ]
    if arrow_rows.empty:
        return

    heading_rad = np.deg2rad(arrow_rows[HEADING_COLUMN])
    dx = np.sin(heading_rad)
    dy = np.cos(heading_rad)

    ax.quiver(
        arrow_rows["plot_x"],
        arrow_rows[ALTITUDE_COLUMN],
        dx,
        dy,
        angles="uv",
        scale=28,
        color="black",
        width=0.003,
        headwidth=4,
        headlength=5,
        alpha=0.75,
        label="Heading",
        zorder=3,
    )


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
            "Usage: python3 visualization/vis_hdg_flightsegmentation.py <csv-file-or-directory> [...]",
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
                print(f"  Skipping {csv_path.name}: no valid position/heading data")
                continue

            df = segment_flight_phases(df, label_phases=True)
            plot_segmented_flight(df, csv_path)
            print(f"  Plotted {csv_path.name} with {len(df)} points")
            print(f"  Phase counts: {phase_summary(df)}")

        except Exception as exc:
            print(f"  Error processing {csv_path.name}: {exc}", file=sys.stderr)

    plt.show()


if __name__ == "__main__":
    main()
