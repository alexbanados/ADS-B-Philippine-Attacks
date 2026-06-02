import os
import random
import sys
import tempfile
from pathlib import Path

cache_dir = Path(tempfile.gettempdir()) / "typicalflight-cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


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


def route_progress_values(df):
    if "route_progress_bin_center" in df.columns:
        values = pd.to_numeric(df["route_progress_bin_center"], errors="coerce")
    elif "route_progress" in df.columns:
        values = pd.to_numeric(df["route_progress"], errors="coerce")
    elif "route_progress_bin" in df.columns:
        values = pd.to_numeric(df["route_progress_bin"], errors="coerce")
        max_value = values.max()
        if pd.notna(max_value) and max_value > 0:
            values = values / max_value
    else:
        raise ValueError("missing route_progress, route_progress_bin_center, or route_progress_bin")

    return values.clip(0, 1)


def load_typical_csv(csv_path):
    df = pd.read_csv(csv_path)
    missing = [column for column in POSITION_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}")

    df = df.copy()
    for column in POSITION_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["route_progress_color"] = route_progress_values(df)
    df = df.dropna(subset=[*POSITION_COLUMNS, "route_progress_color"]).reset_index(
        drop=True
    )
    if df.empty:
        raise ValueError("no valid position/bin data")
    return df


def first_value(df, column, default=None):
    if column not in df.columns:
        return default
    values = df[column].dropna()
    if values.empty:
        return default
    return values.iloc[0]


def plot_subtitle(df):
    if "route_progress_bin" in df.columns:
        bin_count = df["route_progress_bin"].nunique()
    else:
        bin_count = len(df)
    bin_count = first_value(df, "bin_count", bin_count)

    method = first_value(df, "windowing_method", "unknown")
    window_size = first_value(df, "window_size_bins")
    window_count = first_value(df, "window_count", len(df))

    if method == "none":
        windowing_text = "none"
    elif window_size is not None:
        windowing_text = f"{method}, size={int(window_size)} bins"
    else:
        windowing_text = str(method)

    return (
        f"# bins: {int(bin_count)} | "
        f"windowing used: {windowing_text} | "
        f"# windows: {int(window_count)}"
    )


def colored_path(ax, df, *, cmap, norm):
    x = df["longitude"].to_numpy()
    y = df["latitude"].to_numpy()
    color_values = df["route_progress_color"].to_numpy()

    ax.scatter(x, y, c=color_values, cmap=cmap, norm=norm, s=1, zorder=3)


def mark_endpoints(ax, df):
    ax.scatter(
        df["longitude"].iloc[0],
        df["latitude"].iloc[0],
        marker="s",
        s=80,
        facecolors="none",
        edgecolors="#111827",
        linewidths=1.8,
        label="Start",
        zorder=5,
    )
    ax.scatter(
        df["longitude"].iloc[-1],
        df["latitude"].iloc[-1],
        marker="x",
        s=90,
        color="#111827",
        linewidths=2,
        label="End",
        zorder=5,
    )


def plot_typical_path(df, csv_path):
    cmap = "viridis"
    norm = Normalize(vmin=0, vmax=1)

    fig, ax = plt.subplots(figsize=(10, 9))
    colored_path(ax, df, cmap=cmap, norm=norm)
    mark_endpoints(ax, df)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.suptitle(
        f"Typical Flight Path by Route-Progress Bin: {csv_path.name}",
        y=0.985,
    )
    fig.text(0.5, 0.955, plot_subtitle(df), ha="center", va="top", fontsize=10)
    ax.grid(True)
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout(rect=[0, 0, 1, 0.91])


def main():
    if len(sys.argv) == 1:
        print(
            "Usage: python3 visualization/vis_pos_bins.py "
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
            df = load_typical_csv(csv_path)
            plot_typical_path(df, csv_path)
            print(f"  Plotted {csv_path.name} with {len(df)} route-progress bins")
        except Exception as exc:
            print(f"  Error processing {csv_path.name}: {exc}", file=sys.stderr)

    plt.show()


if __name__ == "__main__":
    main()
