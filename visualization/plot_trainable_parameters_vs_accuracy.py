import os
import tempfile
from pathlib import Path


CSV_PATH = Path("COMPLEXITYVSACCURACY.csv")
cache_dir = Path(tempfile.gettempdir()) / "trainable-parameters-vs-accuracy-cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import pandas as pd


OUTPUTS = {
    "TEST Set": Path("trainable_parameters_vs_accuracy_test_set.png"),
    "Pois Set": Path("trainable_parameters_vs_accuracy_poison_set.png"),
}
COMBINED_OUTPUT_PATH = Path("trainable_parameters_vs_accuracy_combined.png")
ARCHITECTURE_MARKER_COMBINED_OUTPUT_PATH = Path(
    "trainable_parameters_vs_accuracy_combined (2).png"
)
TRANSPARENT_ARCHITECTURE_MARKER_OUTPUT_PATH = Path(
    "trainable_parameters_vs_accuracy_combined (3).png"
)
DATASET_TITLES = {
    "TEST Set": "Test Set",
    "Pois Set": "Poison Set",
}
DATASET_COLORS = {
    "TEST Set": "#2563eb",
    "Pois Set": "#dc2626",
}
TRANSPARENT_DATASET_COLORS = {
    "TEST Set": "#dc2626",
    "Pois Set": "#16a34a",
}
ARCHITECTURES = {
    "SEQ": {
        "complexity": ("SEQ Complexity T", "SEQ Complexity NT"),
        "accuracy": "SEQ Accuracy",
        "marker": "o",
        "color": "#2563eb",
        "label": "Sequential",
    },
    "PAR": {
        "complexity": "Par Complexity T",
        "accuracy": "Par Accuracy",
        "marker": "s",
        "color": "#dc2626",
        "label": "Parallel",
    },
}
ROUTE_LABELS = {
    "CEBU": "CEB",
    "DAVAO": "DVO",
    "ILOILO": "ILO",
    "MALAY": "MPH",
    "PUERTO PRINCESA": "PPS",
}


def clean_number(value):
    return float(str(value).strip().replace(",", ""))


def trainable_block(raw, section_column):
    marker = raw.index[raw[section_column].eq("TRAINABLE")]
    if marker.empty:
        raise ValueError("Missing TRAINABLE block in COMPLEXITYVSACCURACY.csv")

    block = raw.iloc[int(marker[0]) + 1 :].copy()
    block[section_column] = block[section_column].ffill()
    return block.dropna(how="all")


def metric_row(section_rows, metric_column, metric_names):
    if isinstance(metric_names, str):
        metric_names = (metric_names,)

    for metric_name in metric_names:
        rows = section_rows[section_rows[metric_column].eq(metric_name)]
        if not rows.empty:
            return rows

    return pd.DataFrame()


def load_plot_data(csv_path):
    raw = pd.read_csv(csv_path, encoding="utf-8-sig")
    section_column = raw.columns[0]
    metric_column = raw.columns[1]
    route_columns = list(raw.columns[2:])
    raw = trainable_block(raw, section_column)

    records = []
    for section in OUTPUTS:
        section_rows = raw[raw[section_column].eq(section)]
        for architecture, fields in ARCHITECTURES.items():
            complexity_row = metric_row(
                section_rows, metric_column, fields["complexity"]
            )
            accuracy_row = metric_row(section_rows, metric_column, fields["accuracy"])
            if complexity_row.empty or accuracy_row.empty:
                raise ValueError(f"Missing {architecture} trainable rows for {section}")

            for route in route_columns:
                records.append(
                    {
                        "dataset": section,
                        "architecture": architecture,
                        "route": route,
                        "route_label": ROUTE_LABELS.get(route, route),
                        "trainable_parameters": clean_number(
                            complexity_row.iloc[0][route]
                        ),
                        "accuracy": clean_number(accuracy_row.iloc[0][route]),
                    }
                )

    return pd.DataFrame(records)


def label_offsets(route_label, architecture):
    return 0, 10


def add_dataset_points(ax, subset, marker_size=80, annotation_fontsize=8):
    for architecture, fields in ARCHITECTURES.items():
        arch_rows = subset[subset["architecture"].eq(architecture)]
        ax.scatter(
            arch_rows["trainable_parameters"],
            arch_rows["accuracy"],
            s=marker_size,
            marker=fields["marker"],
            color=fields["color"],
            edgecolor="#111827",
            linewidth=0.7,
            label=fields["label"],
            zorder=3,
        )

        for row in arch_rows.itertuples(index=False):
            offset = label_offsets(row.route_label, architecture)
            ax.annotate(
                row.route_label,
                (row.trainable_parameters, row.accuracy),
                xytext=offset,
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=annotation_fontsize,
            )


def add_architecture_marker_points(
    ax,
    subset,
    dataset,
    marker_size=80,
    annotation_fontsize=8,
    dataset_colors=None,
    edge_color="#111827",
    label_color="#111827",
):
    if dataset_colors is None:
        dataset_colors = DATASET_COLORS

    for architecture, fields in ARCHITECTURES.items():
        arch_rows = subset[subset["architecture"].eq(architecture)]
        ax.scatter(
            arch_rows["trainable_parameters"],
            arch_rows["accuracy"],
            s=marker_size,
            marker=fields["marker"],
            color=dataset_colors[dataset],
            edgecolor=edge_color,
            linewidth=0.7,
            zorder=3,
        )

        for row in arch_rows.itertuples(index=False):
            offset = label_offsets(row.route_label, architecture)
            ax.annotate(
                row.route_label,
                (row.trainable_parameters, row.accuracy),
                xytext=offset,
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=annotation_fontsize,
                color=label_color,
            )


def format_axes(ax, xlabel, show_ylabel=True, font_color="#111827", transparent=False):
    ax.set_xlabel(xlabel, color=font_color)
    if show_ylabel:
        ax.set_ylabel("Macro accuracy", color=font_color)
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(0.86, 1.005)
    if transparent:
        ax.grid(True, alpha=0.22, color="#ffffff")
    else:
        ax.grid(True, alpha=0.28)
    ax.tick_params(axis="both", colors=font_color)
    for spine in ax.spines.values():
        spine.set_color(font_color)
        spine.set_linewidth(0.6)


def plot_dataset(data, dataset, output_path):
    subset = data[data["dataset"].eq(dataset)]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    add_dataset_points(ax, subset)
    ax.set_title("Trainable Parameters vs. Accuracy")
    format_axes(ax, "Number of trainable parameters")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_combined(data):
    fig, axes = plt.subplots(1, 2, figsize=(9, 5.2), sharey=True)

    for axis_index, (ax, dataset) in enumerate(zip(axes, OUTPUTS)):
        subset = data[data["dataset"].eq(dataset)]
        add_dataset_points(ax, subset, marker_size=58, annotation_fontsize=7)
        format_axes(ax, DATASET_TITLES[dataset], show_ylabel=axis_index == 0)
        ax.legend(frameon=True)

    fig.suptitle("Trainable Parameters vs. Accuracy", y=0.94)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.12, top=0.87, wspace=0.08)
    fig.savefig(COMBINED_OUTPUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def style_legend_text(legend, color):
    for text in legend.get_texts():
        text.set_color(color)


def plot_architecture_marker_combined(
    data,
    output_path=ARCHITECTURE_MARKER_COMBINED_OUTPUT_PATH,
    dataset_colors=None,
    transparent=False,
    font_color="#111827",
):
    if dataset_colors is None:
        dataset_colors = DATASET_COLORS

    if transparent:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(9, 5.2),
            sharey=True,
            facecolor="none",
        )
    else:
        fig, axes = plt.subplots(1, 2, figsize=(9, 5.2), sharey=True)

    for axis_index, (ax, dataset) in enumerate(zip(axes, OUTPUTS)):
        if transparent:
            ax.set_facecolor("none")
        subset = data[data["dataset"].eq(dataset)]
        add_architecture_marker_points(
            ax,
            subset,
            dataset,
            marker_size=58,
            annotation_fontsize=7,
            dataset_colors=dataset_colors,
            edge_color="#ffffff" if transparent else "#111827",
            label_color=font_color,
        )
        format_axes(
            ax,
            DATASET_TITLES[dataset],
            show_ylabel=axis_index == 0,
            font_color=font_color,
            transparent=transparent,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=dataset_colors["TEST Set"],
            markeredgecolor="#ffffff" if transparent else "#111827",
            markersize=8,
            label="Test Set",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=dataset_colors["Pois Set"],
            markeredgecolor="#ffffff" if transparent else "#111827",
            markersize=8,
            label="Poison Set",
        ),
        Line2D(
            [0],
            [0],
            marker=ARCHITECTURES["SEQ"]["marker"],
            color="none",
            markerfacecolor="none" if transparent else "#ffffff",
            markeredgecolor="#ffffff" if transparent else "#111827",
            markersize=8,
            label=ARCHITECTURES["SEQ"]["label"],
        ),
        Line2D(
            [0],
            [0],
            marker=ARCHITECTURES["PAR"]["marker"],
            color="none",
            markerfacecolor="none" if transparent else "#ffffff",
            markeredgecolor="#ffffff" if transparent else "#111827",
            markersize=8,
            label=ARCHITECTURES["PAR"]["label"],
        ),
    ]

    fig.suptitle("Trainable Parameters vs. Accuracy", y=0.915, color=font_color)
    legend = fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.875),
        ncol=4,
        frameon=False,
        fontsize=9,
        handletextpad=0.4,
        columnspacing=1.1,
    )
    style_legend_text(legend, font_color)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.12, top=0.79, wspace=0.08)
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.01,
        transparent=transparent,
    )
    plt.close(fig)


def main():
    data = load_plot_data(CSV_PATH)
    for dataset, output_path in OUTPUTS.items():
        plot_dataset(data, dataset, output_path)
        print(f"Wrote {output_path}")
    plot_combined(data)
    print(f"Wrote {COMBINED_OUTPUT_PATH}")
    plot_architecture_marker_combined(data)
    print(f"Wrote {ARCHITECTURE_MARKER_COMBINED_OUTPUT_PATH}")
    plot_architecture_marker_combined(
        data,
        output_path=TRANSPARENT_ARCHITECTURE_MARKER_OUTPUT_PATH,
        dataset_colors=TRANSPARENT_DATASET_COLORS,
        transparent=True,
        font_color="#ffffff",
    )
    print(f"Wrote {TRANSPARENT_ARCHITECTURE_MARKER_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
