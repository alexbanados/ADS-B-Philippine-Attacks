import os
import tempfile
from pathlib import Path


cache_dir = Path(tempfile.gettempdir()) / "attack-fnr-as-authentic-cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


ROUTES = [
    ("ceb", "MNL-CEB", "poisceb"),
    ("dvo", "MNL-DVO", "poisdvo"),
    ("ilo", "MNL-ILO", "poisilo"),
    ("mph", "MNL-MPH", "poismph"),
    ("pps", "MNL-PPS", "poispps"),
]
ARCHITECTURES = {
    "seq": "pargen_on_seqmodel",
    "par": "seqgen_on_parmodel",
}
MODEL_SET_COLUMNS = [
    ("seq", "synthetic", "Sequential Model on\nTest Set"),
    ("par", "synthetic", "Parallel Model on\nTest Set"),
    ("seq", "poisoned", "Sequential Model on\nPoisoned Set"),
    ("par", "poisoned", "Parallel Model on\nPoisoned Set"),
]
COMBINED_MODEL_SET_LABELS = [
    "Sequential\nModel",
    "Parallel\nModel",
    "Sequential\nModel",
    "Parallel\nModel",
]
COMBINED_V2_MODEL_SET_LABELS = ["SEQ", "PAR", "SEQ", "PAR"]
COMBINED_DATASET_GROUP_LABELS = [("Test Set", 0.5), ("Poisoned Set", 2.5)]
COMBINED_ATTACK_COLORMAPS = ["Reds", "Blues", "Greens"]
COMBINED_ATTACK_LABEL_COLORS = ["#fb6a4a", "#3182bd", "#31a354"]
COMBINED_OUTPUT_PATH = Path("fnr_heatmap_all_attacks_wrt_authentic.png")
COMBINED_V2_OUTPUT_PATH = Path("fnr_heatmap_all_attacks_wrt_authentic (2).png")
COMBINED_TRANSPARENT_OUTPUT_PATH = Path(
    "fnr_heatmap_all_attacks_wrt_authentic (3).png"
)
ATTACK_FIGURES = [
    {
        "number": 1,
        "attack_class": "modified_altitude",
        "panel_title": "Modified Altitude",
        "title": "FNR of Modified Altitude with respect to Authentic",
        "output_path": Path("fnr_heatmap_modified_altitude_wrt_authentic.png"),
    },
    {
        "number": 2,
        "attack_class": "modified_speed",
        "panel_title": "Modified Speed",
        "title": "FNR of Modified Speed with respect to Authentic",
        "output_path": Path("fnr_heatmap_modified_speed_wrt_authentic.png"),
    },
    {
        "number": 3,
        "attack_class": "modified_position",
        "panel_title": "Modified Position",
        "title": "FNR of Modified Position with respect to Authentic",
        "output_path": Path("fnr_heatmap_modified_position_wrt_authentic.png"),
    },
]


def fnr_as_authentic(confmat_path, attack_class):
    confmat = pd.read_csv(confmat_path, index_col=0)
    confmat = confmat.apply(pd.to_numeric)

    if attack_class not in confmat.index:
        raise ValueError(f"Missing true class {attack_class} in {confmat_path}")
    if "authentic" not in confmat.columns:
        raise ValueError(f"Missing predicted authentic column in {confmat_path}")

    row_total = confmat.loc[attack_class].sum()
    if row_total == 0:
        return 0.0

    return float(confmat.loc[attack_class, "authentic"] / row_total)


def confmat_path(route_code, poison_dir, architecture, dataset):
    if dataset == "synthetic":
        return (
            Path("models")
            / f"{route_code}_{architecture}"
            / "test_split_eval_balanced_argmax4"
            / "confmat_balanced_argmax4.csv"
        )

    poison_case = ARCHITECTURES[architecture]
    return (
        Path("poison")
        / poison_dir
        / "results"
        / poison_case
        / f"{poison_case}_confmat_balanced_argmax4.csv"
    )


def heatmap_frame(attack_class):
    rows = []
    for route_code, route_label, poison_dir in ROUTES:
        rows.append(
            [
                fnr_as_authentic(
                    confmat_path(route_code, poison_dir, architecture, dataset),
                    attack_class,
                )
                for architecture, dataset, _ in MODEL_SET_COLUMNS
            ]
        )

    return pd.DataFrame(
        rows,
        index=[route_label for _, route_label, _ in ROUTES],
        columns=[label for _, _, label in MODEL_SET_COLUMNS],
    )


def all_heatmaps():
    matrices = []
    for figure in ATTACK_FIGURES:
        frame = heatmap_frame(figure["attack_class"])
        matrices.append((figure, frame))
    return matrices


def annotate_cells(
    ax,
    frame,
    fontsize=10,
    column_offset=0,
    text_vmax=None,
    dark_ratio_threshold=None,
    force_text_color=None,
    force_low_value_black_below=None,
):
    for row_index, row_label in enumerate(frame.index):
        for col_index, col_label in enumerate(frame.columns):
            value = frame.loc[row_label, col_label]
            dark_relative_cell = (
                text_vmax is not None
                and dark_ratio_threshold is not None
                and value / text_vmax >= dark_ratio_threshold
            )
            if (
                force_low_value_black_below is not None
                and value < force_low_value_black_below
            ):
                text_color = "#000000"
            elif force_text_color is None:
                text_color = (
                    "#ffffff" if value > 0.10 or dark_relative_cell else "#111827"
                )
            else:
                text_color = force_text_color
            ax.text(
                column_offset + col_index,
                row_index,
                f"{value:.1%}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=fontsize,
            )


def transparent_attack_rgba(values, attack_index, vmax):
    base_color = np.array(plt.get_cmap(COMBINED_ATTACK_COLORMAPS[attack_index])(0.9))
    rgba = np.broadcast_to(base_color, values.shape + (4,)).copy()
    rgba[..., 3] = np.clip(values / vmax, 0.0, 1.0)
    return rgba


def plot_heatmap(figure, frame, vmax):
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    image = ax.imshow(frame.values, cmap="Reds", vmin=0.0, vmax=vmax)
    ax.set_aspect("auto")

    ax.set_title(figure["title"], pad=12)
    ax.set_xticks(range(len(frame.columns)))
    ax.set_xticklabels(frame.columns, fontsize=8)
    ax.set_yticks(range(len(frame.index)))
    ax.set_yticklabels(frame.index, fontsize=10)
    annotate_cells(ax, frame)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("FNR")
    colorbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    colorbar.outline.set_visible(False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(figure["output_path"], dpi=300)
    plt.close(fig)


def plot_combined_heatmap(matrices, vmax):
    route_count = len(ROUTES)
    column_count = len(MODEL_SET_COLUMNS)
    total_columns = column_count * len(matrices)
    fig, ax = plt.subplots(figsize=(7.3, 3.55))

    for attack_index, (figure, frame) in enumerate(matrices):
        start_col = attack_index * column_count
        end_col = start_col + column_count
        attack_vmax = max(0.01, frame.values.max() * 1.05)
        ax.imshow(
            frame.values,
            cmap=COMBINED_ATTACK_COLORMAPS[attack_index],
            vmin=0.0,
            vmax=attack_vmax,
            aspect="equal",
            extent=(start_col - 0.5, end_col - 0.5, route_count - 0.5, -0.5),
        )
        annotate_cells(
            ax,
            frame,
            fontsize=5.5,
            column_offset=start_col,
            text_vmax=attack_vmax,
            dark_ratio_threshold=0.72,
        )
        ax.text(
            start_col + (column_count - 1) / 2,
            -0.92,
            figure["panel_title"],
            ha="center",
            va="bottom",
            fontsize=8.2,
        )
        for label, offset in COMBINED_DATASET_GROUP_LABELS:
            ax.text(
                start_col + offset,
                -0.23,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=6.4,
            )

    ax.set_xlim(-0.5, total_columns - 0.5)
    ax.set_ylim(route_count - 0.5, -1.15)
    ax.set_title("FNR with respect to Authentic", pad=15, fontsize=11)
    ax.set_xticks(range(total_columns))
    ax.set_xticklabels(COMBINED_MODEL_SET_LABELS * len(matrices), fontsize=4.3)
    ax.set_yticks(range(route_count))
    ax.set_yticklabels([route_label for _, route_label, _ in ROUTES], fontsize=4.3)
    ax.tick_params(axis="x", length=2, pad=2)
    ax.tick_params(axis="y", length=2)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(left=0.23, right=0.72, bottom=0.25, top=0.82)
    fig.savefig(COMBINED_OUTPUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_combined_heatmap_v2(matrices, vmax):
    route_count = len(ROUTES)
    column_count = len(MODEL_SET_COLUMNS)
    total_columns = column_count * len(matrices)
    fig, ax = plt.subplots(figsize=(7.3, 3.55))
    group_boundaries = set()

    for attack_index, (figure, frame) in enumerate(matrices):
        start_col = attack_index * column_count
        end_col = start_col + column_count
        attack_vmax = max(0.01, frame.values.max() * 1.05)
        ax.imshow(
            frame.values,
            cmap=COMBINED_ATTACK_COLORMAPS[attack_index],
            vmin=0.0,
            vmax=attack_vmax,
            aspect="auto",
            extent=(start_col - 0.5, end_col - 0.5, route_count - 0.5, -0.5),
        )
        annotate_cells(
            ax,
            frame,
            fontsize=5.5,
            column_offset=start_col,
            text_vmax=attack_vmax,
            dark_ratio_threshold=0.72,
        )
        ax.text(
            start_col + (column_count - 1) / 2,
            -0.66,
            figure["panel_title"],
            ha="center",
            va="bottom",
            fontsize=6.4,
            color=COMBINED_ATTACK_LABEL_COLORS[attack_index],
        )
        for boundary in (start_col - 0.5, end_col - 0.5):
            ax.plot(
                [boundary, boundary],
                [-0.82, -0.66],
                color="#6b7280",
                alpha=0.25,
                linewidth=0.8,
                clip_on=False,
            )
        for label, offset in COMBINED_DATASET_GROUP_LABELS:
            group_center = start_col + offset
            group_left = group_center - 1.0
            group_right = group_center + 1.0
            group_boundaries.update((group_left, group_right))
            ax.text(
                group_center,
                -0.13,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=5.2,
            )

    for boundary in sorted(group_boundaries):
        ax.plot(
            [boundary, boundary],
            [-0.12, -0.165],
            transform=ax.get_xaxis_transform(),
            color="#6b7280",
            alpha=0.25,
            linewidth=0.8,
            clip_on=False,
        )

    ax.set_xlim(-0.5, total_columns - 0.5)
    ax.set_ylim(route_count - 0.5, -0.84)
    ax.set_aspect("equal", adjustable="box", anchor="C")
    ax.set_title("FNR with respect to Authentic", pad=9, fontsize=8.2)
    ax.set_xticks(range(total_columns))
    ax.set_xticklabels(COMBINED_V2_MODEL_SET_LABELS * len(matrices), fontsize=4.3)
    ax.set_yticks(range(route_count))
    ax.set_yticklabels([route_label for _, route_label, _ in ROUTES], fontsize=4.3)
    ax.tick_params(axis="x", length=2, pad=2)
    ax.tick_params(axis="y", length=2)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(left=0.23, right=0.72, bottom=0.18, top=0.85)
    fig.savefig(COMBINED_V2_OUTPUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def plot_combined_heatmap_transparent(matrices, vmax):
    route_count = len(ROUTES)
    column_count = len(MODEL_SET_COLUMNS)
    total_columns = column_count * len(matrices)
    fig, ax = plt.subplots(figsize=(7.3, 3.55), facecolor="none")
    ax.set_facecolor("none")
    group_boundaries = set()

    for attack_index, (figure, frame) in enumerate(matrices):
        start_col = attack_index * column_count
        end_col = start_col + column_count
        attack_vmax = max(0.01, frame.values.max() * 1.05)
        ax.imshow(
            transparent_attack_rgba(frame.values, attack_index, attack_vmax),
            aspect="auto",
            extent=(start_col - 0.5, end_col - 0.5, route_count - 0.5, -0.5),
        )
        annotate_cells(
            ax,
            frame,
            fontsize=5.5,
            column_offset=start_col,
            text_vmax=attack_vmax,
            dark_ratio_threshold=0.72,
            force_text_color="#ffffff",
        )
        ax.text(
            start_col + (column_count - 1) / 2,
            -0.66,
            figure["panel_title"],
            ha="center",
            va="bottom",
            fontsize=6.4,
            color="#ffffff",
        )
        for boundary in (start_col - 0.5, end_col - 0.5):
            ax.plot(
                [boundary, boundary],
                [-0.82, -0.66],
                color="#ffffff",
                alpha=0.16,
                linewidth=0.8,
                clip_on=False,
            )
        for label, offset in COMBINED_DATASET_GROUP_LABELS:
            group_center = start_col + offset
            group_left = group_center - 1.0
            group_right = group_center + 1.0
            group_boundaries.update((group_left, group_right))
            ax.text(
                group_center,
                -0.13,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=5.2,
                color="#ffffff",
            )

    for boundary in sorted(group_boundaries):
        ax.plot(
            [boundary, boundary],
            [-0.12, -0.165],
            transform=ax.get_xaxis_transform(),
            color="#ffffff",
            alpha=0.16,
            linewidth=0.8,
            clip_on=False,
        )

    ax.set_xlim(-0.5, total_columns - 0.5)
    ax.set_ylim(route_count - 0.5, -0.84)
    ax.set_aspect("equal", adjustable="box", anchor="C")
    ax.set_title(
        "FNR with respect to Authentic",
        pad=9,
        fontsize=8.2,
        color="#ffffff",
    )
    ax.set_xticks(range(total_columns))
    ax.set_xticklabels(COMBINED_V2_MODEL_SET_LABELS * len(matrices), fontsize=4.3)
    ax.set_yticks(range(route_count))
    ax.set_yticklabels([route_label for _, route_label, _ in ROUTES], fontsize=4.3)
    ax.tick_params(axis="x", length=2, pad=2, colors="#ffffff")
    ax.tick_params(axis="y", length=2, colors="#ffffff")

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(left=0.23, right=0.72, bottom=0.18, top=0.85)
    fig.savefig(
        COMBINED_TRANSPARENT_OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.01,
        transparent=True,
    )
    plt.close(fig)


def main():
    matrices = all_heatmaps()
    vmax = max(frame.values.max() for _, frame in matrices)
    vmax = max(0.01, vmax * 1.05)

    for figure, frame in matrices:
        plot_heatmap(figure, frame, vmax)
        print(f"Wrote {figure['output_path']}")
        print(frame)

    plot_combined_heatmap(matrices, vmax)
    print(f"Wrote {COMBINED_OUTPUT_PATH}")
    plot_combined_heatmap_v2(matrices, vmax)
    print(f"Wrote {COMBINED_V2_OUTPUT_PATH}")
    plot_combined_heatmap_transparent(matrices, vmax)
    print(f"Wrote {COMBINED_TRANSPARENT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
