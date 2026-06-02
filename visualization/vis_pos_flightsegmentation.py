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
import pandas as pd

from flight_segmentation import load_flight
from flight_segmentation import phase_summary
from flight_segmentation import segment_flight_phases
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

MAX_RANDOM_FILES = 100
POSITION_COLUMNS = ("latitude", "longitude")


def plot_segmented_flight(df, csv_path, overlay=None, legend_title="Flight phase"):
    """Plot the flight path as longitude vs latitude, colored by phase."""
    df = df.copy()

    plt.figure(figsize=(10, 9))
    ax = plt.gca()
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

    if overlay is not None:
        overlay(ax, df)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Segmented Flight Path: {csv_path.name}")
    ax.grid(True)
    ax.legend(title=legend_title)
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()


def contiguous_phase_groups(df):
    """Yield contiguous chunks so each colored line segment keeps the phase color."""
    group_id = df["phase"].ne(df["phase"].shift()).cumsum()
    for _, group in df.groupby(group_id):
        yield group


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
            "Usage: python3 visualization/vis_pos_flightsegmentation.py <csv-file-or-directory> [...]",
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

            missing = [column for column in POSITION_COLUMNS if column not in df.columns]
            if missing:
                print(f"  Skipping {csv_path.name}: missing columns {missing}")
                continue

            for column in POSITION_COLUMNS:
                df[column] = pd.to_numeric(df[column], errors="coerce")
            df = df.dropna(subset=POSITION_COLUMNS).reset_index(drop=True)
            if df.empty:
                print(f"  Skipping {csv_path.name}: no valid position data")
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
