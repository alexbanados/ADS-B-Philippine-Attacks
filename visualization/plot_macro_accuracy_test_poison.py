import os
import tempfile
from pathlib import Path


OUTPUT_PATH = Path("macro_accuracy_test_vs_poison.png")
ARCHITECTURE_MARKER_OUTPUT_PATH = Path("macro_accuracy_test_vs_poison (2).png")
TRANSPARENT_OUTPUT_PATH = Path("macro_accuracy_test_vs_poison (3).png")
PLOT_TITLE = "Macro Accuracy Shift Across Test and Poisoned Sets"

cache_dir = Path(tempfile.gettempdir()) / "macro-accuracy-test-poison-cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import pandas as pd


ROUTES = [
    ("ceb", "Cebu", "poisceb"),
    ("dvo", "Davao", "poisdvo"),
    ("ilo", "Iloilo", "poisilo"),
    ("mph", "Malay", "poismph"),
    ("pps", "Puerto\nPrincesa", "poispps"),
]
ARCHITECTURES = [
    ("seq", "Sequential", "pargen_on_seqmodel"),
    ("par", "Parallel", "seqgen_on_parmodel"),
]
COLORS = {
    "Test Set": "#2563eb",
    "Poison Set": "#dc2626",
}
TRANSPARENT_COLORS = {
    "Test Set": "#dc2626",
    "Poison Set": "#16a34a",
}
MARKERS = {
    "Sequential": "o",
    "Parallel": "s",
}
DATASET_MARKERS = {
    "Test Set": "o",
    "Poison Set": "s",
}


def macro_accuracy(confmat_path):
    confmat = pd.read_csv(confmat_path, index_col=0)
    confmat = confmat.apply(pd.to_numeric)

    row_totals = confmat.sum(axis=1)
    valid_rows = row_totals.gt(0)
    diagonal = pd.Series(
        {
            label: confmat.loc[label, label] if label in confmat.columns else 0
            for label in confmat.index
        }
    )

    return float((diagonal[valid_rows] / row_totals[valid_rows]).mean())


def load_data():
    records = []

    for route_code, route_label, poison_dir in ROUTES:
        for architecture, architecture_label, poison_case in ARCHITECTURES:
            model_name = f"{route_code}_{architecture}"
            test_confmat = (
                Path("docs")
                / "testsplit_results"
                / model_name
                / "confmat_balanced_argmax4.csv"
            )
            poison_confmat = (
                Path("docs")
                / "poison_results"
                / route_code
                / poison_case
                / f"{poison_case}_confmat_balanced_argmax4.csv"
            )

            records.extend(
                [
                    {
                        "model": model_name,
                        "route": route_label,
                        "architecture": architecture_label,
                        "dataset": "Test Set",
                        "accuracy": macro_accuracy(test_confmat),
                    },
                    {
                        "model": model_name,
                        "route": route_label,
                        "architecture": architecture_label,
                        "dataset": "Poison Set",
                        "accuracy": macro_accuracy(poison_confmat),
                    },
                ]
            )

    return pd.DataFrame(records)


def style_legend_text(legend, color):
    for text in legend.get_texts():
        text.set_color(color)


def plot(
    data,
    output_path,
    marker_mode="dataset",
    colors=None,
    transparent=False,
    font_color="#111827",
):
    if colors is None:
        colors = COLORS

    model_order = []
    x_labels = []
    x_positions = {}
    group_centers = {}
    group_gap = 0
    current_x = 0

    for architecture, architecture_label, _ in ARCHITECTURES:
        group_start = current_x
        for route_code, route_label, _ in ROUTES:
            model = f"{route_code}_{architecture}"
            model_order.append(model)
            x_labels.append(route_label)
            x_positions[model] = current_x
            current_x += 1
        group_end = current_x - 1
        group_centers[architecture_label] = (group_start + group_end) / 2
        current_x += group_gap

    if transparent:
        fig, ax = plt.subplots(figsize=(10.5, 5.8), facecolor="none")
    else:
        fig, ax = plt.subplots(figsize=(10.5, 5.8))
    if transparent:
        ax.set_facecolor("none")

    for model in model_order:
        x = x_positions[model]
        row = data[data["model"].eq(model)]
        test_acc = row[row["dataset"].eq("Test Set")]["accuracy"].iloc[0]
        poison_acc = row[row["dataset"].eq("Poison Set")]["accuracy"].iloc[0]

        ax.plot(
            [x, x],
            [test_acc, poison_acc],
            color="#ffffff" if transparent else "#9ca3af",
            alpha=0.42 if transparent else 1.0,
            linewidth=1.2,
            zorder=1,
        )

    if marker_mode == "architecture":
        for dataset, color in COLORS.items():
            for architecture_label, marker in MARKERS.items():
                rows = data[
                    data["dataset"].eq(dataset)
                    & data["architecture"].eq(architecture_label)
                ]
                ax.scatter(
                    rows["model"].map(x_positions),
                    rows["accuracy"],
                    s=62,
                    marker=marker,
                    color=colors[dataset],
                    edgecolor="#ffffff" if transparent else "#111827",
                    linewidth=0.6,
                    zorder=3,
                )
    else:
        for dataset, marker in DATASET_MARKERS.items():
            rows = data[data["dataset"].eq(dataset)]
            ax.scatter(
                rows["model"].map(x_positions),
                rows["accuracy"],
                s=62,
                marker=marker,
                color=colors[dataset],
                edgecolor="#ffffff" if transparent else "#111827",
                linewidth=0.6,
                label=dataset,
                zorder=3,
            )

    ax.set_title(PLOT_TITLE, pad=14, color=font_color)
    ax.set_xlabel("Route", color=font_color)
    ax.set_ylabel("Macro accuracy", color=font_color)
    ax.set_xticks([x_positions[model] for model in model_order])
    ax.set_xticklabels(x_labels)

    y_min = max(0.0, data["accuracy"].min() - 0.005)
    y_max = min(1.0, data["accuracy"].max() + 0.005)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.6, max(x_positions.values()) + 0.6)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    if transparent:
        ax.grid(axis="y", alpha=0.22, color="#ffffff")
        ax.grid(axis="x", alpha=0.10, color="#ffffff")
    else:
        ax.grid(axis="y", alpha=0.28)
        ax.grid(axis="x", alpha=0.12)
    ax.tick_params(axis="both", colors=font_color)
    if marker_mode == "architecture":
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=colors["Test Set"],
                markeredgecolor="#ffffff" if transparent else "#111827",
                markersize=8,
                label="Test Set",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=colors["Poison Set"],
                markeredgecolor="#ffffff" if transparent else "#111827",
                markersize=8,
                label="Poison Set",
            ),
            Line2D(
                [0],
                [0],
                marker=MARKERS["Sequential"],
                color="none",
                markerfacecolor="none" if transparent else "#ffffff",
                markeredgecolor="#ffffff" if transparent else "#111827",
                markersize=8,
                label="Sequential",
            ),
            Line2D(
                [0],
                [0],
                marker=MARKERS["Parallel"],
                color="none",
                markerfacecolor="none" if transparent else "#ffffff",
                markeredgecolor="#ffffff" if transparent else "#111827",
                markersize=8,
                label="Parallel",
            ),
        ]
        legend = ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=4,
            frameon=False,
        )
    else:
        legend = ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            frameon=False,
        )
    style_legend_text(legend, font_color)
    ax.axvline(
        len(ROUTES) - 0.5,
        color="#ffffff" if transparent else "#d1d5db",
        alpha=0.35 if transparent else 1.0,
        linewidth=0.8,
        zorder=0,
    )

    for architecture_label, center in group_centers.items():
        ax.text(
            center,
            -0.14,
            architecture_label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10,
            color=font_color,
        )

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color(font_color)

    fig.subplots_adjust(bottom=0.22, top=0.86, left=0.08, right=0.98)
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.04,
        transparent=transparent,
    )
    plt.close(fig)


def main():
    data = load_data()
    plot(data, OUTPUT_PATH, marker_mode="dataset")
    plot(data, ARCHITECTURE_MARKER_OUTPUT_PATH, marker_mode="architecture")
    plot(
        data,
        TRANSPARENT_OUTPUT_PATH,
        marker_mode="architecture",
        colors=TRANSPARENT_COLORS,
        transparent=True,
        font_color="#ffffff",
    )
    print(data.pivot(index="model", columns="dataset", values="accuracy"))
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote {ARCHITECTURE_MARKER_OUTPUT_PATH}")
    print(f"Wrote {TRANSPARENT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
