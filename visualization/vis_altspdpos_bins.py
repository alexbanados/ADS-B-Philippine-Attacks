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
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


MAX_RANDOM_FILES = 50
ALTITUDE_COLUMN = "altitude_meters"
SPEED_COLUMN = "speed_kmh"
POSITION_COLUMNS = ("latitude", "longitude")
REQUIRED_COLUMNS = (ALTITUDE_COLUMN, SPEED_COLUMN, *POSITION_COLUMNS)
ROUTE_DESTINATIONS = {
    "ceb": "Cebu",
    "dvo": "Davao",
    "ilo": "Iloilo",
    "mph": "Malay, Aklan",
    "pps": "Puerto Prinsesa",
}


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
        raise ValueError(
            "missing route_progress, route_progress_bin_center, or route_progress_bin"
        )

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
    df = df.dropna(subset=[*REQUIRED_COLUMNS, "route_progress_color"]).reset_index(
        drop=True
    )
    if df.empty:
        raise ValueError("no valid altitude/speed/position/bin data")
    return df


def plot_title(csv_path):
    route_tokens = csv_path.stem.lower().replace("-", "_").split("_")
    for route_code in route_tokens:
        if route_code in ROUTE_DESTINATIONS:
            return f"Typical Flight of Manila - {ROUTE_DESTINATIONS[route_code]}"

    return f"Typical Flight of Manila - {csv_path.stem}"


def colored_points(ax, x, y, color_values, *, cmap, norm, size=1):
    ax.scatter(x, y, c=color_values, cmap=cmap, norm=norm, s=size, zorder=3)


def mark_time_series_endpoints(ax, df, y_column):
    ax.scatter(
        df["plot_x"].iloc[0],
        df[y_column].iloc[0],
        marker="s",
        s=70,
        facecolors="none",
        edgecolors="#111827",
        linewidths=1.8,
        zorder=5,
    )
    ax.scatter(
        df["plot_x"].iloc[-1],
        df[y_column].iloc[-1],
        marker="x",
        s=80,
        color="#111827",
        linewidths=2,
        zorder=5,
    )


def mark_position_endpoints(ax, df):
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


def plot_panel(ax, df, column, ylabel, *, cmap, norm):
    colored_points(
        ax,
        df["plot_x"].to_numpy(),
        df[column].to_numpy(),
        df["route_progress_color"].to_numpy(),
        cmap=cmap,
        norm=norm,
    )
    mark_time_series_endpoints(ax, df, column)
    ax.set_ylabel(ylabel)
    ax.grid(True)


def plot_position(ax, df, *, cmap, norm):
    colored_points(
        ax,
        df["longitude"].to_numpy(),
        df["latitude"].to_numpy(),
        df["route_progress_color"].to_numpy(),
        cmap=cmap,
        norm=norm,
    )
    mark_position_endpoints(ax, df)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Flight path")
    ax.grid(True)
    ax.set_aspect("equal", adjustable="datalim")


def add_horizontal_legends(fig, position_ax, *, cmap, norm):
    handles, labels = position_ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.945),
            ncol=len(labels),
            frameon=True,
        )

    colorbar_ax = fig.add_axes([0.35, 0.865, 0.30, 0.018])
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=colorbar_ax,
        orientation="horizontal",
    )
    colorbar.set_label("Route progress")


def plot_typical_flight(df, csv_path):
    df = df.copy()
    df["plot_x"] = range(len(df))

    cmap = "viridis"
    norm = Normalize(vmin=0, vmax=1)
    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.25, 1.0],
        wspace=0.24,
        hspace=0.34,
        left=0.06,
        right=0.96,
        bottom=0.12,
        top=0.78,
    )
    altitude_ax = fig.add_subplot(grid[0, 0])
    speed_ax = fig.add_subplot(grid[1, 0], sharex=altitude_ax)
    position_ax = fig.add_subplot(grid[:, 1])

    plot_panel(
        altitude_ax,
        df,
        ALTITUDE_COLUMN,
        "Altitude (meters)",
        cmap=cmap,
        norm=norm,
    )
    plot_panel(
        speed_ax,
        df,
        SPEED_COLUMN,
        "Speed (km/h)",
        cmap=cmap,
        norm=norm,
    )
    plot_position(position_ax, df, cmap=cmap, norm=norm)

    def format_time_tick(value, position):
        row_idx = int(round(value))
        if row_idx < 0 or row_idx >= len(df):
            return ""
        row = df.iloc[row_idx]
        if "Time_local" in row and pd.notna(row["Time_local"]):
            return f"{row['Time_local']:%Y-%m-%d %H:%M}\n({int(row['timestamp'])})"
        return str(row_idx)

    speed_ax.set_xlim(-0.5, len(df) - 0.5)
    speed_ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=24, integer=True))
    speed_ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_time_tick))
    speed_ax.tick_params(axis="x", which="major", labelsize=5, labelrotation=45)
    altitude_ax.tick_params(axis="x", labelbottom=False)
    for label in speed_ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_verticalalignment("top")

    speed_ax.set_xlabel("Route-progress bins, ordered from 0.00 to 1.00")
    fig.suptitle(plot_title(csv_path), y=0.98)
    add_horizontal_legends(fig, position_ax, cmap=cmap, norm=norm)


def main():
    if len(sys.argv) == 1:
        print(
            "Usage: python3 visualization/vis_altspdpos_bins.py "
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
