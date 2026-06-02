from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

cache_dir = Path(tempfile.gettempdir()) / "flight-noenv-attack-cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from mod_pos import latitude_longitude_from_relative_xy
from vis_attack_raw import ALTITUDE_COLUMN
from vis_attack_raw import MAX_RANDOM_FILES
from vis_attack_raw import POSITION_COLUMNS
from vis_attack_raw import ROUTE_PROGRESS_COLUMN
from vis_attack_raw import SPEED_COLUMN
from vis_attack_raw import attack_mask
from vis_attack_raw import contiguous_true_groups
from vis_attack_raw import envelope_summary_text
from vis_attack_raw import format_time_axis
from vis_attack_raw import load_plot_flight
from vis_attack_raw import resolve_csv_paths
from visualization_helpers import mark_airport_points
from visualization_helpers import mark_ground_endpoints


BASELINE_COLOR = "#6b7280"
SHAPED_COLOR = "#2563eb"
RAW_NOENV_COLOR = "#f97316"
WINDOW_COLOR = "#fef3c7"

ATTACK_SPECS = {
    "modified_altitude": {
        "feature": ALTITUDE_COLUMN,
        "delta": "attack_delta_altitude",
        "offset": "altitude_offset",
        "label": "Modified altitude",
    },
    "modified_speed": {
        "feature": SPEED_COLUMN,
        "delta": "attack_delta_speed",
        "offset": "speed_offset",
        "label": "Modified speed",
    },
    "modified_position": {
        "delta_x": "attack_delta_x",
        "delta_y": "attack_delta_y",
        "offset_x": "x_wrt0_offset",
        "offset_y": "y_wrt0_offset",
        "label": "Modified position",
    },
}
FILENAME_ATTACK_TYPES = (
    ("modalt_", "modified_altitude"),
    ("modspd_", "modified_speed"),
    ("modpos_", "modified_position"),
)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Visualize an attack CSV as if the sampled attack perturbation were "
            "applied directly, before envelope shaping."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Attack CSV files or directories containing attack CSVs.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=MAX_RANDOM_FILES,
        help=(
            "Maximum number of CSV files to visualize from a directory. If more "
            f"files are found, a random subset is selected. Default: {MAX_RANDOM_FILES}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Optional folder for PNG outputs. If omitted, figures are shown "
            "interactively."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively even when --output-dir is provided.",
    )
    return parser.parse_args(argv)


def numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype(float)


def infer_attack_type(df: pd.DataFrame, csv_path: Path) -> str:
    attacked = attack_mask(df)
    if attacked.any() and "attack_type" in df.columns:
        attack_types = (
            df.loc[attacked, "attack_type"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .unique()
        )
        for attack_type in attack_types:
            if attack_type in ATTACK_SPECS:
                return attack_type

    for prefix, attack_type in FILENAME_ATTACK_TYPES:
        if csv_path.name.startswith(prefix):
            return attack_type

    available_delta_columns = set(df.columns)
    if "attack_delta_altitude" in available_delta_columns:
        return "modified_altitude"
    if "attack_delta_speed" in available_delta_columns:
        return "modified_speed"
    if {"attack_delta_x", "attack_delta_y"}.issubset(available_delta_columns):
        return "modified_position"

    raise ValueError(
        "Could not infer attack type. Expected attack_type metadata or a "
        "modalt_/modspd_/modpos_ filename."
    )


def reconstruct_scalar_noenv(
    df: pd.DataFrame,
    attacked: pd.Series,
    feature: str,
    delta_column: str,
    offset_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_df = df.copy()
    raw_df = df.copy()
    offsets = numeric_series(df, offset_column)
    raw_delta = numeric_series(df, delta_column)

    baseline_values = pd.to_numeric(df[feature], errors="coerce").astype(float) - offsets
    baseline_df[feature] = baseline_values
    raw_df[feature] = baseline_values
    raw_df.loc[attacked, feature] = (
        baseline_values.loc[attacked] + raw_delta.loc[attacked]
    )
    return baseline_df, raw_df


def reconstruct_position_noenv(
    df: pd.DataFrame,
    attacked: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = ["x_wrt0", "y_wrt0", *POSITION_COLUMNS]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Position attack CSV is missing columns: {missing}")

    baseline_df = df.copy()
    raw_df = df.copy()
    x_values = pd.to_numeric(df["x_wrt0"], errors="coerce").astype(float)
    y_values = pd.to_numeric(df["y_wrt0"], errors="coerce").astype(float)
    x_offsets = numeric_series(df, "x_wrt0_offset")
    y_offsets = numeric_series(df, "y_wrt0_offset")
    raw_delta_x = numeric_series(df, "attack_delta_x")
    raw_delta_y = numeric_series(df, "attack_delta_y")

    baseline_x = x_values - x_offsets
    baseline_y = y_values - y_offsets
    raw_x = baseline_x.copy()
    raw_y = baseline_y.copy()
    raw_x.loc[attacked] = baseline_x.loc[attacked] + raw_delta_x.loc[attacked]
    raw_y.loc[attacked] = baseline_y.loc[attacked] + raw_delta_y.loc[attacked]

    origin_latitude = float(pd.to_numeric(df["latitude"], errors="coerce").iloc[0])
    origin_longitude = float(pd.to_numeric(df["longitude"], errors="coerce").iloc[0])
    baseline_latitude, baseline_longitude = latitude_longitude_from_relative_xy(
        baseline_x.to_numpy(dtype=float),
        baseline_y.to_numpy(dtype=float),
        origin_latitude=origin_latitude,
        origin_longitude=origin_longitude,
    )
    raw_latitude, raw_longitude = latitude_longitude_from_relative_xy(
        raw_x.to_numpy(dtype=float),
        raw_y.to_numpy(dtype=float),
        origin_latitude=origin_latitude,
        origin_longitude=origin_longitude,
    )

    baseline_df["x_wrt0"] = baseline_x
    baseline_df["y_wrt0"] = baseline_y
    baseline_df["latitude"] = baseline_latitude
    baseline_df["longitude"] = baseline_longitude
    raw_df["x_wrt0"] = raw_x
    raw_df["y_wrt0"] = raw_y
    raw_df["latitude"] = raw_latitude
    raw_df["longitude"] = raw_longitude
    return baseline_df, raw_df


def reconstruct_noenv_data(
    df: pd.DataFrame,
    csv_path: Path,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    attack_type = infer_attack_type(df, csv_path)
    attacked = attack_mask(df)
    if not attacked.any():
        raise ValueError("CSV has no attacked rows to reconstruct")

    if attack_type == "modified_position":
        baseline_df, raw_df = reconstruct_position_noenv(df, attacked)
    else:
        spec = ATTACK_SPECS[attack_type]
        baseline_df, raw_df = reconstruct_scalar_noenv(
            df,
            attacked,
            feature=spec["feature"],
            delta_column=spec["delta"],
            offset_column=spec["offset"],
        )

    return attack_type, baseline_df, raw_df


def shade_attack_window(ax, df: pd.DataFrame, attacked: pd.Series) -> None:
    if not attacked.any():
        return
    attack_x = df.loc[attacked, "plot_x"]
    ax.axvspan(
        float(attack_x.min()) - 0.5,
        float(attack_x.max()) + 0.5,
        color=WINDOW_COLOR,
        alpha=0.35,
        zorder=0,
    )


def plot_time_series_noenv(
    ax,
    shaped_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    column: str,
    ylabel: str,
    attacked: pd.Series,
    active: bool,
) -> None:
    shade_attack_window(ax, shaped_df, attacked)
    ax.plot(
        baseline_df["plot_x"],
        baseline_df[column],
        color=BASELINE_COLOR,
        linestyle="--",
        linewidth=1.2,
        label="Baseline",
        zorder=1,
    )
    ax.plot(
        shaped_df["plot_x"],
        shaped_df[column],
        color=SHAPED_COLOR,
        linewidth=1.4,
        label="Envelope-shaped attack",
        zorder=2,
    )

    if active:
        raw_label_used = False
        for group in contiguous_true_groups(raw_df, attacked):
            ax.plot(
                group["plot_x"],
                group[column],
                color=RAW_NOENV_COLOR,
                linewidth=2.0,
                marker=".",
                markersize=3.5,
                label=(
                    "Raw no-envelope attack"
                    if not raw_label_used
                    else "_nolegend_"
                ),
                zorder=3,
            )
            raw_label_used = True

    mark_ground_endpoints(ax, shaped_df, "plot_x", column)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def plot_position_noenv(
    ax,
    shaped_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    attacked: pd.Series,
    active: bool,
) -> None:
    ax.plot(
        baseline_df["longitude"],
        baseline_df["latitude"],
        color=BASELINE_COLOR,
        linestyle="--",
        linewidth=1.2,
        zorder=1,
    )
    ax.plot(
        shaped_df["longitude"],
        shaped_df["latitude"],
        color=SHAPED_COLOR,
        linewidth=1.4,
        zorder=2,
    )
    if active:
        raw_label_used = False
        for group in contiguous_true_groups(raw_df, attacked):
            ax.plot(
                group["longitude"],
                group["latitude"],
                color=RAW_NOENV_COLOR,
                linewidth=2.0,
                marker=".",
                markersize=3.5,
                label=(
                    "Raw no-envelope attack"
                    if not raw_label_used
                    else "_nolegend_"
                ),
                zorder=3,
            )
            raw_label_used = True

    mark_ground_endpoints(ax, shaped_df, "longitude", "latitude")
    mark_airport_points(ax)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")


def raw_delta_summary(df: pd.DataFrame, attack_type: str, attacked: pd.Series) -> str:
    if attack_type == "modified_altitude":
        raw_delta = numeric_series(df, "attack_delta_altitude").loc[attacked]
        return f"Raw delta: {raw_delta.iloc[0]:.2f} m"
    if attack_type == "modified_speed":
        raw_delta = numeric_series(df, "attack_delta_speed").loc[attacked]
        return f"Raw delta: {raw_delta.iloc[0]:.2f} km/h"

    dx = numeric_series(df, "attack_delta_x").loc[attacked].iloc[0]
    dy = numeric_series(df, "attack_delta_y").loc[attacked].iloc[0]
    magnitude = float(np.hypot(dx, dy))
    return f"Raw delta: dx={dx:.2f} m, dy={dy:.2f} m, |d|={magnitude:.2f} m"


def legend_handles(summary_text: str, envelope_text: str, delta_text: str):
    return [
        Line2D(
            [0],
            [0],
            color=BASELINE_COLOR,
            linewidth=1.2,
            linestyle="--",
            label="Baseline before attack",
        ),
        Line2D(
            [0],
            [0],
            color=SHAPED_COLOR,
            linewidth=1.4,
            label="Envelope-shaped attack",
        ),
        Line2D(
            [0],
            [0],
            color=RAW_NOENV_COLOR,
            linewidth=2.0,
            marker=".",
            markersize=3.5,
            label="Raw no-envelope attack",
        ),
        Line2D(
            [0],
            [0],
            color=WINDOW_COLOR,
            linewidth=8,
            alpha=0.55,
            label="Attack window",
        ),
        Line2D([0], [0], color="none", linewidth=0, label=summary_text),
        Line2D([0], [0], color="none", linewidth=0, label=envelope_text),
        Line2D([0], [0], color="none", linewidth=0, label=delta_text),
    ]


def plot_noenv_attack_flight(df: pd.DataFrame, csv_path: Path):
    attack_type, baseline_df, raw_df = reconstruct_noenv_data(df, csv_path)
    attacked = attack_mask(df)
    spec = ATTACK_SPECS[attack_type]
    summary_text = f"Attack type: {spec['label']}; rows: {int(attacked.sum())}"
    envelope_text = envelope_summary_text(df, attacked)
    delta_text = raw_delta_summary(df, attack_type, attacked)

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

    plot_time_series_noenv(
        altitude_ax,
        df,
        baseline_df,
        raw_df,
        ALTITUDE_COLUMN,
        "Altitude (meters)",
        attacked,
        active=attack_type == "modified_altitude",
    )
    format_time_axis(altitude_ax, df)

    plot_time_series_noenv(
        speed_ax,
        df,
        baseline_df,
        raw_df,
        SPEED_COLUMN,
        "Speed (km/h)",
        attacked,
        active=attack_type == "modified_speed",
    )
    format_time_axis(speed_ax, df)

    plot_position_noenv(
        position_ax,
        df,
        baseline_df,
        raw_df,
        attacked,
        active=attack_type == "modified_position",
    )

    fig.suptitle(f"Raw No-Envelope Attack View: {csv_path.name}", y=0.98)
    fig.legend(
        handles=legend_handles(summary_text, envelope_text, delta_text),
        loc="upper center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, 0.93),
        title="Comparison",
    )
    return fig


def output_path_for(csv_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{csv_path.stem}_raw_noenv.png"


def main():
    if len(sys.argv) == 1:
        print(
            "Usage: python3 visualization/vis_attack_raw_noenv.py "
            "[--output-dir visualized_noenv] <attack-csv-or-directory> [...]",
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

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    figures = []
    for csv_path in csv_files:
        print(f"Processing {csv_path.name}...")
        try:
            df = load_plot_flight(csv_path)
            fig = plot_noenv_attack_flight(df, csv_path)
            if args.output_dir is not None:
                output_path = output_path_for(csv_path, args.output_dir)
                fig.savefig(output_path, dpi=300)
                print(f"  Wrote {output_path}")
            if args.show or args.output_dir is None:
                figures.append(fig)
            else:
                plt.close(fig)
        except Exception as exc:
            print(f"  Error processing {csv_path.name}: {exc}", file=sys.stderr)

    if figures:
        plt.show()


if __name__ == "__main__":
    main()
