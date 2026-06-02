import argparse
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
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from dataset_paths import resolve_dataset_path
from flight_segmentation import load_flight
from visualization_helpers import mark_airport_points
from visualization_helpers import mark_ground_endpoints


MAX_RANDOM_FILES = 50
ALTITUDE_COLUMN = "altitude_meters"
SPEED_COLUMN = "speed_kmh"
ROUTE_PROGRESS_COLUMN = "route_progress"
POSITION_COLUMNS = ("latitude", "longitude")
REQUIRED_COLUMNS = (
    ALTITUDE_COLUMN,
    SPEED_COLUMN,
    ROUTE_PROGRESS_COLUMN,
    *POSITION_COLUMNS,
)

ATTACK_COLOR = "#dc2626"
STYLE_LEGEND_COLOR = "#374151"
ROUTE_PROGRESS_CMAP = LinearSegmentedColormap.from_list(
    "route_progress_blue_yellow",
    ["#2563eb", "#facc15"],
)
ROUTE_PROGRESS_NORM = Normalize(vmin=0.0, vmax=1.0)
ATTACK_FILENAME_PREFIXES = ("modalt_", "modspd_", "modpos_")
ATTACK_FOLDER_PREFIXES = ("data_modalt_", "data_modspd_", "data_modpos_")
AUTHENTIC_FOLDER_PREFIX = "data_nolvl_"


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Plot raw altitude, speed, and position, coloring attacked rows red "
            "and overlaying matching authentic CSVs when available."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="CSV files or directories containing attack CSVs.",
    )
    parser.add_argument(
        "--authentic-folder",
        type=Path,
        help=(
            "Folder containing authentic CSVs. If omitted, dataset/data_modalt_<route>, "
            "dataset/data_modspd_<route>, and dataset/data_modpos_<route> inputs are "
            "matched to dataset/data_nolvl_<route>."
        ),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=MAX_RANDOM_FILES,
        help=(
            "Maximum number of CSV files to visualize from a folder. If more "
            f"files are found, a random subset is selected. Default: {MAX_RANDOM_FILES}."
        ),
    )
    return parser.parse_args(argv)


def attack_mask(df):
    """Return rows labeled as attacked by attack-generation metadata."""
    if "is_attacked" in df.columns:
        attacked = pd.to_numeric(df["is_attacked"], errors="coerce").fillna(0)
        return attacked.eq(1)

    if "attack_type" in df.columns:
        return df["attack_type"].astype(str).str.lower().ne("authentic")

    return pd.Series(False, index=df.index)


def attack_summary_text(df, attacked):
    attacked_rows = df[attacked]
    if attacked_rows.empty:
        return "Attacked timestamps: 0"

    attacked_timestamps = attacked_rows["timestamp"].nunique()
    return f"Attacked timestamps: {attacked_timestamps}"


def envelope_summary_text(df, attacked):
    attacked_rows = df[attacked]
    if attacked_rows.empty or "attack_envelope_type" not in attacked_rows.columns:
        return "Envelope: n/a"

    envelope_type = str(attacked_rows["attack_envelope_type"].dropna().iloc[0])
    if "attack_envelope_params" not in attacked_rows.columns:
        return f"Envelope: {envelope_type}"

    params = attacked_rows["attack_envelope_params"].dropna()
    if params.empty or str(params.iloc[0]).strip() == "":
        return f"Envelope: {envelope_type}"

    return f"Envelope: {envelope_type} ({params.iloc[0]})"


def contiguous_true_groups(df, mask):
    """Yield contiguous attacked chunks so red line segments do not bridge gaps."""
    group_id = mask.ne(mask.shift()).cumsum()
    for _, group in df[mask].groupby(group_id[mask]):
        yield group


def contiguous_mask_groups(df, mask):
    group_id = mask.ne(mask.shift()).cumsum()
    for _, group in df[mask].groupby(group_id[mask]):
        yield group


def attack_source_name(csv_path):
    for prefix in ATTACK_FILENAME_PREFIXES:
        if csv_path.name.startswith(prefix):
            return csv_path.name[len(prefix):]
    return csv_path.name


def inferred_authentic_folder(csv_path):
    folder_name = csv_path.parent.name
    route_suffix = None
    for prefix in ATTACK_FOLDER_PREFIXES:
        if folder_name.startswith(prefix):
            route_suffix = folder_name[len(prefix):]
            break
    if route_suffix is None:
        return None

    folder = ROOT_DIR / "dataset" / f"{AUTHENTIC_FOLDER_PREFIX}{route_suffix}"
    return folder if folder.is_dir() else None


def authentic_csv_path(csv_path, authentic_folder=None):
    source_name = attack_source_name(csv_path)

    if authentic_folder is not None:
        candidate = authentic_folder / source_name
        return candidate if candidate.exists() else None

    folder = inferred_authentic_folder(csv_path)
    if folder is None:
        return None

    candidate = folder / source_name
    return candidate if candidate.exists() else None


def load_plot_flight(csv_path):
    df = load_flight(csv_path)
    if df.empty:
        raise ValueError("no valid data")

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}")

    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
    if df.empty:
        raise ValueError("no valid plot data")

    df[ROUTE_PROGRESS_COLUMN] = df[ROUTE_PROGRESS_COLUMN].clip(0, 1)
    df["plot_x"] = range(len(df))
    return df


def format_time_axis(ax, df):
    def format_time_tick(value, position):
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


def plot_route_progress_gradient(
    ax,
    df,
    x_column,
    y_column,
    linewidth,
    linestyle="-",
    alpha=1.0,
    zorder=2,
):
    if df.empty:
        return

    x = df[x_column].to_numpy(dtype=float)
    y = df[y_column].to_numpy(dtype=float)
    route_progress = df[ROUTE_PROGRESS_COLUMN].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(route_progress)
    x = x[valid]
    y = y[valid]
    route_progress = route_progress[valid]
    if len(x) == 0:
        return

    if len(x) == 1:
        ax.scatter(
            x,
            y,
            c=route_progress,
            cmap=ROUTE_PROGRESS_CMAP,
            norm=ROUTE_PROGRESS_NORM,
            s=4,
            alpha=alpha,
            zorder=zorder,
        )
        return

    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    segment_progress = (route_progress[:-1] + route_progress[1:]) / 2
    lines = LineCollection(
        segments,
        cmap=ROUTE_PROGRESS_CMAP,
        norm=ROUTE_PROGRESS_NORM,
        linewidths=linewidth,
        linestyles=linestyle,
        alpha=alpha,
        zorder=zorder,
    )
    lines.set_array(segment_progress)
    ax.add_collection(lines)
    ax.autoscale_view()


def plot_time_series(ax, df, column, ylabel, attacked, authentic_df=None):
    if authentic_df is not None:
        plot_route_progress_gradient(
            ax,
            authentic_df,
            "plot_x",
            column,
            linewidth=1.5,
            linestyle="--",
            alpha=0.85,
            zorder=1,
        )

    for group in contiguous_mask_groups(df, ~attacked):
        plot_route_progress_gradient(
            ax,
            group,
            "plot_x",
            column,
            linewidth=1.8,
            linestyle="-",
            alpha=1.0,
            zorder=2,
        )

    attack_label_used = False
    for group in contiguous_true_groups(df, attacked):
        ax.plot(
            group["plot_x"],
            group[column],
            color=ATTACK_COLOR,
            linewidth=2.4,
            marker=".",
            markersize=3,
            label="Attack" if not attack_label_used else "_nolegend_",
        )
        attack_label_used = True

    mark_ground_endpoints(ax, df, "plot_x", column)
    ax.set_ylabel(ylabel)
    ax.grid(True)


def plot_position(ax, df, csv_path, attacked, authentic_df=None):
    if authentic_df is not None:
        plot_route_progress_gradient(
            ax,
            authentic_df,
            "longitude",
            "latitude",
            linewidth=1.5,
            linestyle="--",
            alpha=0.85,
            zorder=1,
        )

    for group in contiguous_mask_groups(df, ~attacked):
        plot_route_progress_gradient(
            ax,
            group,
            "longitude",
            "latitude",
            linewidth=1.8,
            linestyle="-",
            alpha=1.0,
            zorder=2,
        )

    attack_label_used = False
    for group in contiguous_true_groups(df, attacked):
        ax.plot(
            group["longitude"],
            group["latitude"],
            color=ATTACK_COLOR,
            linewidth=2.4,
            marker=".",
            markersize=3.5,
            label="Attack" if not attack_label_used else "_nolegend_",
        )
        attack_label_used = True

    mark_ground_endpoints(ax, df, "longitude", "latitude")
    mark_airport_points(ax)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True)
    ax.set_aspect("equal", adjustable="datalim")


def figure_legend_handles(summary_text, envelope_text):
    return [
        Line2D(
            [0],
            [0],
            color=STYLE_LEGEND_COLOR,
            linewidth=1.8,
            linestyle="--",
            label="Authentic path",
        ),
        Line2D(
            [0],
            [0],
            color=STYLE_LEGEND_COLOR,
            linewidth=2,
            linestyle="-",
            label="Modified path",
        ),
        Line2D(
            [0],
            [0],
            color=ATTACK_COLOR,
            linewidth=2.4,
            marker=".",
            markersize=3.5,
            label="Attack",
        ),
        Line2D(
            [0],
            [0],
            color="#111827",
            marker="s",
            markerfacecolor="none",
            linewidth=0,
            markersize=8,
            label="First Row",
        ),
        Line2D(
            [0],
            [0],
            color="#111827",
            marker="x",
            linewidth=0,
            markersize=8,
            label="Last Row",
        ),
        Line2D(
            [0],
            [0],
            color="none",
            linewidth=0,
            label=summary_text,
        ),
        Line2D(
            [0],
            [0],
            color="none",
            linewidth=0,
            label=envelope_text,
        ),
    ]


def plot_raw_attack_flight(df, csv_path, authentic_df=None, authentic_path=None):
    df = df.copy()
    attacked = attack_mask(df)
    summary_text = attack_summary_text(df, attacked)
    envelope_text = envelope_summary_text(df, attacked)

    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.25, 1.0],
        height_ratios=[1, 1],
        wspace=0.24,
        hspace=0.42,
        left=0.06,
        right=0.90,
        bottom=0.10,
        top=0.82,
    )
    altitude_ax = fig.add_subplot(grid[0, 0])
    speed_ax = fig.add_subplot(grid[1, 0])
    position_ax = fig.add_subplot(grid[:, 1])

    plot_time_series(
        altitude_ax,
        df,
        ALTITUDE_COLUMN,
        "Altitude (meters)",
        attacked,
        authentic_df=authentic_df,
    )
    format_time_axis(altitude_ax, df)

    plot_time_series(
        speed_ax,
        df,
        SPEED_COLUMN,
        "Speed (km/h)",
        attacked,
        authentic_df=authentic_df,
    )
    format_time_axis(speed_ax, df)

    plot_position(position_ax, df, csv_path, attacked, authentic_df=authentic_df)

    if authentic_path is None:
        title = f"Raw Flight Attack View: {csv_path.name}"
    else:
        title = f"Raw Flight Attack View: {csv_path.name} vs {authentic_path.name}"
    fig.suptitle(title, y=0.98)
    fig.legend(
        handles=figure_legend_handles(summary_text, envelope_text),
        loc="upper center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, 0.93),
        title="Flight / row label",
    )
    colorbar = fig.colorbar(
        ScalarMappable(norm=ROUTE_PROGRESS_NORM, cmap=ROUTE_PROGRESS_CMAP),
        ax=[altitude_ax, speed_ax, position_ax],
        fraction=0.025,
        pad=0.02,
    )
    colorbar.set_label("Route progress")


def resolve_csv_paths(args):
    csv_paths = []
    for arg in args:
        path = resolve_dataset_path(Path(arg))
        if path.is_dir():
            csv_paths.extend(sorted(path.glob("*.csv")))
        else:
            csv_paths.append(path)
    return csv_paths


def main():
    if len(sys.argv) == 1:
        print(
            "Usage: python3 visualization/vis_attack_raw.py "
            "[--authentic-folder dataset/data_nolvl_ceb] <csv-file-or-directory> [...]",
            file=sys.stderr,
        )
        return

    args = parse_args(sys.argv[1:])
    if args.max_files <= 0:
        print("--max-files must be greater than 0", file=sys.stderr)
        return

    csv_files = resolve_csv_paths(args.paths)
    if not csv_files:
        print("No CSV files found.", file=sys.stderr)
        return

    if len(csv_files) > args.max_files:
        total_files = len(csv_files)
        csv_files = sorted(random.sample(csv_files, args.max_files))
        print(f"Randomly selected {len(csv_files)} of {total_files} CSV files:")
        for csv_path in csv_files:
            print(f"  {csv_path.name}")

    for csv_path in csv_files:
        print(f"Processing {csv_path.name}...")
        try:
            df = load_plot_flight(csv_path)
            auth_path = authentic_csv_path(csv_path, args.authentic_folder)
            authentic_df = None
            if auth_path is not None and auth_path.resolve() != csv_path.resolve():
                try:
                    authentic_df = load_plot_flight(auth_path)
                    print(f"  Authentic baseline: {auth_path}")
                except Exception as exc:
                    print(f"  Could not load authentic baseline {auth_path}: {exc}")
                    auth_path = None
            elif csv_path.name.startswith(ATTACK_FILENAME_PREFIXES):
                print(f"  Authentic baseline not found for {csv_path.name}")

            plot_raw_attack_flight(
                df,
                csv_path,
                authentic_df=authentic_df,
                authentic_path=auth_path,
            )
            print(f"  Plotted {csv_path.name} with {len(df)} points")
            print(f"  Attack rows: {int(attack_mask(df).sum())}")

        except Exception as exc:
            print(f"  Error processing {csv_path.name}: {exc}", file=sys.stderr)

    plt.show()


if __name__ == "__main__":
    main()
