import sys

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

import vis_attack_raw as base
from visualization_helpers import AIRPORT_POINTS


AXIS_TEXT_COLOR = "#f8fafc"
LEGEND_EDGE_COLOR = "#cbd5e1"
GRID_COLOR = "#f8fafc"
AIRPORT_MARKER_COLOR = "#0f172a"

base.ROUTE_PROGRESS_CMAP = LinearSegmentedColormap.from_list(
    "route_progress_gray_white",
    ["#94a3b8", "#ffffff"],
)


def mark_ground_endpoints_dark(ax, df, x_column, y_column):
    if df.empty:
        return

    first_row = df.iloc[0]
    last_row = df.iloc[-1]

    ax.scatter(
        first_row[x_column],
        first_row[y_column],
        marker="s",
        s=70,
        facecolors="none",
        edgecolors=AXIS_TEXT_COLOR,
        linewidths=1.8,
        label="First Row",
        zorder=5,
    )
    ax.scatter(
        last_row[x_column],
        last_row[y_column],
        marker="x",
        s=80,
        color=AXIS_TEXT_COLOR,
        linewidths=2,
        label="Last Row",
        zorder=5,
    )


def mark_airport_points_dark(ax):
    for code, (latitude, longitude) in AIRPORT_POINTS.items():
        ax.scatter(
            longitude,
            latitude,
            marker="o",
            s=46,
            color=AIRPORT_MARKER_COLOR,
            edgecolors=AXIS_TEXT_COLOR,
            linewidths=0.8,
            zorder=6,
        )
        ax.annotate(
            code,
            (longitude, latitude),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color=AXIS_TEXT_COLOR,
            zorder=7,
        )


base.mark_ground_endpoints = mark_ground_endpoints_dark
base.mark_airport_points = mark_airport_points_dark


def style_dark_axis(ax):
    ax.set_facecolor("none")
    ax.patch.set_alpha(0)
    ax.tick_params(axis="both", colors=AXIS_TEXT_COLOR)
    ax.xaxis.label.set_color(AXIS_TEXT_COLOR)
    ax.yaxis.label.set_color(AXIS_TEXT_COLOR)
    ax.title.set_color(AXIS_TEXT_COLOR)
    ax.grid(True, color=GRID_COLOR, alpha=0.16, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_color(AXIS_TEXT_COLOR)
        spine.set_linewidth(0.8)


def style_dark_legend(legend):
    frame = legend.get_frame()
    frame.set_facecolor("none")
    frame.set_edgecolor(LEGEND_EDGE_COLOR)
    frame.set_alpha(0)
    for text in legend.get_texts():
        text.set_color(AXIS_TEXT_COLOR)
    legend.get_title().set_color(AXIS_TEXT_COLOR)


def style_dark_colorbar(colorbar):
    colorbar.ax.set_facecolor("none")
    colorbar.ax.patch.set_alpha(0)
    colorbar.ax.tick_params(colors=AXIS_TEXT_COLOR)
    colorbar.ax.yaxis.label.set_color(AXIS_TEXT_COLOR)
    colorbar.outline.set_edgecolor(AXIS_TEXT_COLOR)


def figure_legend_handles(summary_text, envelope_text):
    return [
        Line2D(
            [0],
            [0],
            color=AXIS_TEXT_COLOR,
            linewidth=1.8,
            linestyle="--",
            label="Authentic path",
        ),
        Line2D(
            [0],
            [0],
            color=AXIS_TEXT_COLOR,
            linewidth=2,
            linestyle="-",
            label="Modified path",
        ),
        Line2D(
            [0],
            [0],
            color=base.ATTACK_COLOR,
            linewidth=2.4,
            marker=".",
            markersize=3.5,
            label="Attack",
        ),
        Line2D(
            [0],
            [0],
            color=AXIS_TEXT_COLOR,
            marker="s",
            markerfacecolor="none",
            linewidth=0,
            markersize=8,
            label="First Row",
        ),
        Line2D(
            [0],
            [0],
            color=AXIS_TEXT_COLOR,
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
    attacked = base.attack_mask(df)
    summary_text = base.attack_summary_text(df, attacked)
    envelope_text = base.envelope_summary_text(df, attacked)

    fig = plt.figure(figsize=(16, 9), facecolor="none")
    fig.patch.set_alpha(0)
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

    base.plot_time_series(
        altitude_ax,
        df,
        base.ALTITUDE_COLUMN,
        "Altitude (meters)",
        attacked,
        authentic_df=authentic_df,
    )
    base.format_time_axis(altitude_ax, df)

    base.plot_time_series(
        speed_ax,
        df,
        base.SPEED_COLUMN,
        "Speed (km/h)",
        attacked,
        authentic_df=authentic_df,
    )
    base.format_time_axis(speed_ax, df)

    base.plot_position(position_ax, df, csv_path, attacked, authentic_df=authentic_df)

    if authentic_path is None:
        title = f"Raw Flight Attack View: {csv_path.name}"
    else:
        title = f"Raw Flight Attack View: {csv_path.name} vs {authentic_path.name}"
    fig.suptitle(title, y=0.98, color=AXIS_TEXT_COLOR)

    for ax in (altitude_ax, speed_ax, position_ax):
        style_dark_axis(ax)

    legend = fig.legend(
        handles=figure_legend_handles(summary_text, envelope_text),
        loc="upper center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, 0.93),
        title="Flight / row label",
    )
    style_dark_legend(legend)

    colorbar = fig.colorbar(
        ScalarMappable(norm=base.ROUTE_PROGRESS_NORM, cmap=base.ROUTE_PROGRESS_CMAP),
        ax=[altitude_ax, speed_ax, position_ax],
        fraction=0.025,
        pad=0.02,
    )
    colorbar.set_label("Route progress")
    style_dark_colorbar(colorbar)


def main():
    if len(sys.argv) == 1:
        print(
            "Usage: python3 visualization/vis_attack_raw_trans.py "
            "[--authentic-folder dataset/data_nolvl_ceb] <csv-file-or-directory> [...]",
            file=sys.stderr,
        )
        return

    args = base.parse_args(sys.argv[1:])
    if args.max_files <= 0:
        print("--max-files must be greater than 0", file=sys.stderr)
        return

    csv_files = base.resolve_csv_paths(args.paths)
    if not csv_files:
        print("No CSV files found.", file=sys.stderr)
        return

    if len(csv_files) > args.max_files:
        total_files = len(csv_files)
        csv_files = sorted(base.random.sample(csv_files, args.max_files))
        print(f"Randomly selected {len(csv_files)} of {total_files} CSV files:")
        for csv_path in csv_files:
            print(f"  {csv_path.name}")

    for csv_path in csv_files:
        print(f"Processing {csv_path.name}...")
        try:
            df = base.load_plot_flight(csv_path)
            auth_path = base.authentic_csv_path(csv_path, args.authentic_folder)
            authentic_df = None
            if auth_path is not None and auth_path.resolve() != csv_path.resolve():
                try:
                    authentic_df = base.load_plot_flight(auth_path)
                    print(f"  Authentic baseline: {auth_path}")
                except Exception as exc:
                    print(f"  Could not load authentic baseline {auth_path}: {exc}")
                    auth_path = None
            elif csv_path.name.startswith(base.ATTACK_FILENAME_PREFIXES):
                print(f"  Authentic baseline not found for {csv_path.name}")

            plot_raw_attack_flight(
                df,
                csv_path,
                authentic_df=authentic_df,
                authentic_path=auth_path,
            )
            print(f"  Plotted {csv_path.name} with {len(df)} points")
            print(f"  Attack rows: {int(base.attack_mask(df).sum())}")

        except Exception as exc:
            print(f"  Error processing {csv_path.name}: {exc}", file=sys.stderr)

    plt.show()


if __name__ == "__main__":
    main()
