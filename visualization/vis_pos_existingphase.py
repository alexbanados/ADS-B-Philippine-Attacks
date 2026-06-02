import os
import random
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

cache_dir = Path(tempfile.gettempdir()) / "flight-segmentation-cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))

import matplotlib.pyplot as plt
import pandas as pd

from visualization_helpers import mark_airport_points
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
POSITION_COLUMNS = ("latitude", "longitude")


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


def load_position_csv(csv_path):
    df = pd.read_csv(csv_path)
    missing = [column for column in POSITION_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}")

    df = df.copy()
    for column in POSITION_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["phase"] = existing_phase_labels(df)
    df = df.dropna(subset=[*POSITION_COLUMNS, "phase"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("no valid position and phase data")
    return df


def contiguous_phase_groups(df):
    group_id = df["phase"].ne(df["phase"].shift()).cumsum()
    for _, group in df.groupby(group_id):
        yield group


def phase_summary(df):
    return df["phase"].value_counts(sort=False).to_dict()


def plot_existing_phase_path(df, csv_path):
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

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Existing-Phase Flight Path: {csv_path.name}")
    ax.grid(True)
    ax.legend(title="CSV phase")
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()


def main():
    if len(sys.argv) == 1:
        print(
            "Usage: python3 visualization/vis_pos_existingphase.py "
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
            df = load_position_csv(csv_path)
            plot_existing_phase_path(df, csv_path)
            print(f"  Plotted {csv_path.name} with {len(df)} points")
            print(f"  Phase counts: {phase_summary(df)}")
        except Exception as exc:
            print(f"  Error processing {csv_path.name}: {exc}", file=sys.stderr)

    plt.show()


if __name__ == "__main__":
    main()
