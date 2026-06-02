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
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


MAX_RANDOM_FILES = 50
ALTITUDE_COLUMN = "altitude_meters"
SPEED_COLUMN = "speed_kmh"
VERTICAL_SPEED_COLUMN = "verticalSpeed_ms"
REQUIRED_COLUMNS = (ALTITUDE_COLUMN, SPEED_COLUMN)


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


def add_time_columns(df):
    if "timestamp" not in df.columns:
        return df

    timestamp = pd.to_numeric(df["timestamp"], errors="coerce")
    df["timestamp"] = timestamp
    df["Time_local"] = pd.to_datetime(timestamp, unit="s", utc=True).dt.tz_convert(
        "Asia/Manila"
    )
    return df


def add_vertical_speed(df):
    if VERTICAL_SPEED_COLUMN in df.columns:
        df[VERTICAL_SPEED_COLUMN] = pd.to_numeric(
            df[VERTICAL_SPEED_COLUMN],
            errors="coerce",
        )
        if df[VERTICAL_SPEED_COLUMN].notna().any():
            return df

    if "t_elapsed_sec" in df.columns:
        elapsed = pd.to_numeric(df["t_elapsed_sec"], errors="coerce")
    elif "timestamp" in df.columns:
        timestamp = pd.to_numeric(df["timestamp"], errors="coerce")
        elapsed = timestamp - timestamp.min()
    else:
        elapsed = pd.Series(range(len(df)), index=df.index)

    altitude = pd.to_numeric(df[ALTITUDE_COLUMN], errors="coerce")
    dt = elapsed.diff().replace(0, np.nan)
    df[VERTICAL_SPEED_COLUMN] = altitude.diff() / dt
    df[VERTICAL_SPEED_COLUMN] = df[VERTICAL_SPEED_COLUMN].fillna(0)
    return df


def load_typical_csv(csv_path):
    df = pd.read_csv(csv_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}")

    df = df.copy()
    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["route_progress_color"] = route_progress_values(df)
    df = add_time_columns(df)
    df = add_vertical_speed(df)
    df = df.dropna(
        subset=[*REQUIRED_COLUMNS, VERTICAL_SPEED_COLUMN, "route_progress_color"]
    ).reset_index(drop=True)
    if df.empty:
        raise ValueError("no valid altitude/speed/bin data")
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


def colored_line(ax, x, y, color_values, *, cmap, norm):
    ax.scatter(x, y, c=color_values, cmap=cmap, norm=norm, s=1, zorder=3)


def mark_endpoints(ax, df, y_column):
    ax.scatter(
        df["plot_x"].iloc[0],
        df[y_column].iloc[0],
        marker="s",
        s=70,
        facecolors="none",
        edgecolors="#111827",
        linewidths=1.8,
        label="Start",
        zorder=5,
    )
    ax.scatter(
        df["plot_x"].iloc[-1],
        df[y_column].iloc[-1],
        marker="x",
        s=80,
        color="#111827",
        linewidths=2,
        label="End",
        zorder=5,
    )


def plot_panel(ax, df, column, ylabel, *, cmap, norm):
    x = df["plot_x"].to_numpy()
    y = df[column].to_numpy()
    color_values = df["route_progress_color"].to_numpy()
    colored_line(ax, x, y, color_values, cmap=cmap, norm=norm)
    mark_endpoints(ax, df, column)
    ax.set_ylabel(ylabel)
    ax.grid(True)


def plot_typical_flight(df, csv_path):
    df = df.copy()
    df["plot_x"] = range(len(df))

    cmap = "viridis"
    norm = Normalize(vmin=0, vmax=1)
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    plot_panel(
        axes[0],
        df,
        ALTITUDE_COLUMN,
        "Altitude (meters)",
        cmap=cmap,
        norm=norm,
    )
    plot_panel(
        axes[1],
        df,
        SPEED_COLUMN,
        "Speed (km/h)",
        cmap=cmap,
        norm=norm,
    )
    plot_panel(
        axes[2],
        df,
        VERTICAL_SPEED_COLUMN,
        "Vertical speed (m/s)",
        cmap=cmap,
        norm=norm,
    )
    axes[2].axhline(0, color="#111827", linewidth=1, alpha=0.6)

    def format_time_tick(value, position):
        row_idx = int(round(value))
        if row_idx < 0 or row_idx >= len(df):
            return ""
        row = df.iloc[row_idx]
        if "Time_local" in row and pd.notna(row["Time_local"]):
            return f"{row['Time_local']:%Y-%m-%d %H:%M}\n({int(row['timestamp'])})"
        return str(row_idx)

    axes[2].set_xlim(-0.5, len(df) - 0.5)
    axes[2].xaxis.set_major_locator(mticker.MaxNLocator(nbins=24, integer=True))
    axes[2].xaxis.set_major_formatter(mticker.FuncFormatter(format_time_tick))
    axes[2].tick_params(axis="x", which="major", labelsize=5, labelrotation=45)
    for label in axes[2].get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_verticalalignment("top")

    axes[2].set_xlabel("Route-progress bins, ordered from 0.00 to 1.00")
    axes[0].legend()
    fig.suptitle(f"Typical Flight by Route-Progress Bin: {csv_path.name}", y=0.985)
    fig.text(0.5, 0.955, plot_subtitle(df), ha="center", va="top", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.91])


def main():
    if len(sys.argv) == 1:
        print(
            "Usage: python3 visualization/vis_altspdvspd_bins.py "
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
            plot_typical_flight(df, csv_path)
            print(f"  Plotted {csv_path.name} with {len(df)} route-progress bins")
        except Exception as exc:
            print(f"  Error processing {csv_path.name}: {exc}", file=sys.stderr)

    plt.show()


if __name__ == "__main__":
    main()
