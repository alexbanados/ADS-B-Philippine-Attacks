import os
import tempfile
from pathlib import Path

CSV_PATH = Path("COMPLEXITYVSACCURACY.csv")
cache_dir = Path(tempfile.gettempdir()) / "complexity-vs-accuracy-cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
OUTPUTS = {
    "TEST Set": Path("complexity_vs_accuracy_test_set.png"),
    "Pois Set": Path("complexity_vs_accuracy_poison_set.png"),
}
DATASET_TITLES = {
    "TEST Set": "Test Set",
    "Pois Set": "Poison Set",
}
ARCHITECTURES = {
    "SEQ": {
        "complexity": "SEQ Complexity",
        "accuracy": "SEQ Accuracy",
        "marker": "o",
        "color": "#2563eb",
        "label": "SEQ",
    },
    "PAR": {
        "complexity": "Par Complexity",
        "accuracy": "Par Accuracy",
        "marker": "s",
        "color": "#dc2626",
        "label": "PAR",
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


def load_plot_data(csv_path):
    raw = pd.read_csv(csv_path, encoding="utf-8-sig")
    section_column = raw.columns[0]
    metric_column = raw.columns[1]
    route_columns = list(raw.columns[2:])

    raw[section_column] = raw[section_column].ffill()
    raw = raw.dropna(how="all")

    records = []
    for section in OUTPUTS:
        section_rows = raw[raw[section_column].eq(section)]
        for architecture, fields in ARCHITECTURES.items():
            complexity_row = section_rows[
                section_rows[metric_column].eq(fields["complexity"])
            ]
            accuracy_row = section_rows[
                section_rows[metric_column].eq(fields["accuracy"])
            ]
            if complexity_row.empty or accuracy_row.empty:
                raise ValueError(f"Missing {architecture} rows for {section}")

            for route in route_columns:
                records.append(
                    {
                        "dataset": section,
                        "architecture": architecture,
                        "route": route,
                        "route_label": ROUTE_LABELS.get(route, route),
                        "complexity": clean_number(complexity_row.iloc[0][route]),
                        "accuracy": clean_number(accuracy_row.iloc[0][route]),
                    }
                )

    return pd.DataFrame(records)


def label_offsets(route_label, architecture):
    offset = {
        "CEB": (-8, 8),
        "DVO": (8, 8),
        "ILO": (8, -12),
        "MPH": (-8, -12),
        "PPS": (8, 8),
    }.get(route_label, (6, 6))
    if architecture == "PAR":
        return -offset[0], offset[1]
    return offset


def plot_dataset(data, dataset, output_path):
    subset = data[data["dataset"].eq(dataset)]
    fig, ax = plt.subplots(figsize=(9, 5.2))

    for architecture, fields in ARCHITECTURES.items():
        arch_rows = subset[subset["architecture"].eq(architecture)]
        ax.scatter(
            arch_rows["complexity"],
            arch_rows["accuracy"],
            s=80,
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
                (row.complexity, row.accuracy),
                xytext=offset,
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=8,
            )

    ax.set_title(f"Complexity vs Accuracy - {DATASET_TITLES[dataset]}")
    ax.set_xlabel("Number of trainable parameters")
    ax.set_ylabel("Macro accuracy")
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(0.86, 1.005)
    ax.grid(True, alpha=0.28)
    ax.legend(title="Model", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    data = load_plot_data(CSV_PATH)
    for dataset, output_path in OUTPUTS.items():
        plot_dataset(data, dataset, output_path)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
