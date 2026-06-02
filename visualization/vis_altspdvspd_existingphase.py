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
from flight_segmentation import smooth_altitude
from visualization_helpers import mark_ground_endpoints


PHASE_COLORS = {
    "GND": "#6b7280",
    "Ground": "#6b7280",
    "TKF": "#f97316",
    "Takeoff": "#f97316",
    "ICL": "#facc15",
    "Initial Climb": "#facc15",
    "CLB": "#16a34a",
    "Climb": "#16a34a",
    "CRZ": "#2563eb",
    "Cruise": "#2563eb",
    "DSC": "#dc2626",
    "Descent": "#dc2626",
    "APP": "#7c3aed",
    "Approach": "#7c3aed",
}

PHASE_CODES = {
    0: "GND",
    1: "TKF",
    2: "ICL",
    3: "CLB",
    4: "CRZ",
    5: "DSC",
    6: "APP",
}

MAX_RANDOM_FILES = 50
ALTITUDE_COLUMN = "altitude_meters"
SPEED_COLUMN = "speed_kmh"
VERTICAL_SPEED_COLUMN = "verticalSpeed_ms"
REQUIRED_VIS_COLUMNS = (ALTITUDE_COLUMN, SPEED_COLUMN)


def resolve_csv_paths(args):
    csv_paths = []
    for arg in args:
        path = Path(arg)
        if path.is_dir():
            csv_paths.extend(sorted(path.glob("*.csv")))
        else:
            csv_paths.append(path)
    return csv_paths


def normalize_phase_value(value):
    if pd.isna(value) or str(value).strip() == "":
        return None

    phase_text = str(value).strip()
    try:
        phase_code = int(float(phase_text))
    except ValueError:
        phase_code = None

    if phase_code in PHASE_CODES:
        return PHASE_CODES[phase_code]

    return phase_text


def existing_phase_labels(df):
    if "phase_label" in df.columns:
        labels = df["phase_label"].map(normalize_phase_value)
        if labels.notna().any():
            return labels

    if "phase" not in df.columns:
        raise ValueError("missing 'phase' or 'phase_label' column")

    return df["phase"].map(normalize_phase_value)


def phase_summary(df):
    return df["phase"].value_counts(sort=False).to_dict()


def add_vertical_speed(df):
    df = df.copy()
    if VERTICAL_SPEED_COLUMN in df.columns:
        df[VERTICAL_SPEED_COLUMN] = pd.to_numeric(
            df[VERTICAL_SPEED_COLUMN],
            errors="coerce",
        )
        if df[VERTICAL_SPEED_COLUMN].notna().any():
            return df

    smooth_window = get_smooth_window(len(df))
    smoothed_altitude = smooth_altitude(df[ALTITUDE_COLUMN], smooth_window)
    df[VERTICAL_SPEED_COLUMN] = get_vertical_speed(
        df,
        smoothed_altitude,
        smooth_window,
    )
    return df


def load_existing_phase_csv(csv_path):
    df = load_flight(csv_path)
    if df.empty:
        raise ValueError("no valid data")

    missing = [column for column in REQUIRED_VIS_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}")

    for column in REQUIRED_VIS_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["phase"] = existing_phase_labels(df)
    df = df.dropna(subset=[*REQUIRED_VIS_COLUMNS, "phase"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("no valid altitude/speed and phase data")

    df = add_vertical_speed(df)
    df = df.dropna(subset=[VERTICAL_SPEED_COLUMN]).reset_index(drop=True)
    if df.empty:
        raise ValueError("no valid vertical speed data")
    return df


def contiguous_phase_groups(df):
    group_id = df["phase"].ne(df["phase"].shift()).cumsum()
    for _, group in df.groupby(group_id):
        yield group


def plot_existing_phase_flight(df, csv_path):
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

    def format_time_tick(value, position):
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
    axes[0].legend(title="CSV phase")
    fig.suptitle(
        f"Existing-Phase Altitude, Speed, and Vertical Speed: {csv_path.name}"
    )
    plt.tight_layout()


def plot_phase_series(ax, df, column, ylabel, seen_phases):
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


def main():
    if len(sys.argv) == 1:
        print(
            "Usage: python3 visualization/vis_altspdvspd_existingphase.py "
            "<csv-file-or-directory> [...]",
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
            df = load_existing_phase_csv(csv_path)
            plot_existing_phase_flight(df, csv_path)
            print(f"  Plotted {csv_path.name} with {len(df)} points")
            print(f"  Phase counts: {phase_summary(df)}")
        except Exception as exc:
            print(f"  Error processing {csv_path.name}: {exc}", file=sys.stderr)

    plt.show()


if __name__ == "__main__":
    main()
