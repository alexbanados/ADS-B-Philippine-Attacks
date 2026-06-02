import argparse
import os
import tempfile
from pathlib import Path

cache_dir = Path(tempfile.gettempdir()) / "envelope-cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))

import matplotlib.pyplot as plt
import numpy as np

from mod_tuner import ADJUSTABLE_SPLINE_CONTROL_POINTS_MAX
from mod_tuner import ADJUSTABLE_SPLINE_CONTROL_POINTS_MIN
from mod_tuner import ADJUSTABLE_SPLINE_MIN_CONTROL_VALUE
from mod_tuner import asymmetric_hann_envelope
from mod_tuner import beta_curve_envelope
from mod_tuner import hann_envelope
from mod_tuner import normalize_envelope
from mod_tuner import raised_cosine_envelope


DEFAULT_ROWS = 201
DEFAULT_SEED = 42
SHAPE_TITLES = {
    "hann": "Hann Envelope",
    "asymmetric_hann": "Asymmetric Hann",
    "beta": "Beta Curve",
    "raised_cosine": "Raised-Cosine Pulse",
    "random_spline": "Random B-spline",
}
PARAMETERS_BY_SHAPE = {
    "hann": ("n",),
    "asymmetric_hann": ("n", "a"),
    "beta": ("n", "a", "b"),
    "raised_cosine": ("n", "a", "b"),
    "random_spline": ("n", "m", "p"),
}
PARAMETER_SYMBOLS = {
    "n": "n",
    "p": "π",
    "a": "α",
    "b": "β",
    "c": "κ",
    "m": "μ",
}
PARAMETER_MARKERS = {
    "n": "s",
    "p": "^",
    "a": "o",
    "b": "D",
    "c": "P",
    "m": "X",
}


def envelope_shapes(row_count, seed):
    return {
        "hann": build_shape(hann_envelope, row_count, None),
        "asymmetric_hann": build_shape(
            asymmetric_hann_envelope,
            row_count,
            np.random.default_rng(seed + 1),
        ),
        "beta": build_shape(
            beta_curve_envelope,
            row_count,
            np.random.default_rng(seed + 2),
        ),
        "raised_cosine": build_shape(
            raised_cosine_envelope,
            row_count,
            np.random.default_rng(seed + 3),
        ),
        "random_spline": build_random_spline_shape(
            row_count,
            np.random.default_rng(seed + 4),
        ),
    }


def build_shape(builder, row_count, rng):
    if rng is None:
        envelope, params = builder(row_count)
    else:
        envelope, params = builder(row_count, rng)
    return envelope, params, {}


def build_random_spline_shape(row_count, rng):
    if row_count <= 0:
        return np.array([], dtype=float), "control_points=0", {"control_points": []}
    if row_count == 1:
        return np.array([1.0], dtype=float), "control_points=0", {
            "control_points": [(0.0, 1.0)]
        }
    if row_count == 2:
        return np.array([0.0, 0.0], dtype=float), "control_points=0", {
            "control_points": [(0.0, 0.0), (1.0, 0.0)]
        }

    internal_count = int(
        rng.integers(
            ADJUSTABLE_SPLINE_CONTROL_POINTS_MIN,
            ADJUSTABLE_SPLINE_CONTROL_POINTS_MAX + 1,
        )
    )
    base_x = np.linspace(0, 1, internal_count + 2)[1:-1]
    spacing = 1 / (internal_count + 1)
    jitter = rng.uniform(-0.30 * spacing, 0.30 * spacing, size=internal_count)
    internal_x = np.sort(np.clip(base_x + jitter, spacing * 0.35, 1 - spacing * 0.35))

    internal_y = rng.uniform(
        ADJUSTABLE_SPLINE_MIN_CONTROL_VALUE,
        1.0,
        size=internal_count,
    )
    center_rank = internal_count // 2
    peak_rank = int(
        np.clip(center_rank + rng.integers(-1, 2), 0, internal_count - 1)
    )
    internal_y[peak_rank] = 1.0

    control_x = np.concatenate(([0.0], internal_x, [1.0]))
    control_y = np.concatenate(([0.0], internal_y, [0.0]))

    x = np.linspace(0, 1, row_count)
    envelope = np.zeros(row_count, dtype=float)
    for left in range(len(control_x) - 1):
        right = left + 1
        in_segment = (x >= control_x[left]) & (x <= control_x[right])
        if not in_segment.any():
            continue

        span = control_x[right] - control_x[left]
        position = (x[in_segment] - control_x[left]) / span
        eased_position = 0.5 - 0.5 * np.cos(np.pi * position)
        envelope[in_segment] = (
            control_y[left]
            + (control_y[right] - control_y[left]) * eased_position
        )

    envelope = normalize_envelope(envelope)
    control_y_on_curve = np.interp(control_x, x, envelope)
    params = (
        f"control_points={internal_count};"
        f"min_control_value={ADJUSTABLE_SPLINE_MIN_CONTROL_VALUE:.2f};"
        f"peak_control_index={peak_rank + 1}"
    )
    metadata = {
        "control_points": list(zip(control_x.tolist(), control_y_on_curve.tolist()))
    }
    return envelope, params, metadata


def plot_envelopes(row_count, seed):
    shapes = envelope_shapes(row_count, seed)
    x = np.linspace(0, 1, row_count)

    fig, axes = plt.subplots(1, 5, figsize=(16, 4.6), sharex=True, sharey=True)

    for ax, (name, (envelope, params, metadata)) in zip(axes, shapes.items()):
        ax.plot(x, envelope, linewidth=1.6)
        ax.fill_between(x, envelope, alpha=0.18)
        ax.set_title(SHAPE_TITLES[name], fontweight="normal")
        mark_parameter_effects(ax, x, envelope, name, params, metadata)
        ax.grid(True, alpha=0.35)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["attack_start", "attack_end"], fontsize=7)
        labels = ax.get_xticklabels()
        if len(labels) == 2:
            labels[0].set_horizontalalignment("left")
            labels[1].set_horizontalalignment("right")
        ax.tick_params(axis="x", length=3)
        for spine in ax.spines.values():
            spine.set_linewidth(0.25)
    axes[0].set_ylabel("Envelope strength")

    fig.subplots_adjust(
        left=0.05,
        right=0.99,
        bottom=0.14,
        top=0.86,
        wspace=0.22,
    )
    return fig


def mark_parameter_effects(ax, x, envelope, name, params, metadata):
    mark_window_length(ax)

    if name == "asymmetric_hann":
        mark_peak(ax, x, envelope, "a")
    elif name == "beta":
        mark_beta_sides(ax, x, envelope)
    elif name == "raised_cosine":
        values = parse_params(params)
        rise = values.get("rise_fraction")
        fall = values.get("fall_fraction")
        if rise is not None:
            mark_vertical(
                ax,
                rise,
                "a",
                y_value=0.96,
                xytext=(7, 0),
                ha="left",
                va="center",
            )
        if fall is not None:
            mark_vertical(
                ax,
                1 - fall,
                "b",
                y_value=0.96,
                xytext=(-7, 0),
                ha="right",
                va="center",
            )
    elif name == "random_spline":
        mark_spline_effects(ax, x, envelope, params, metadata)


def parse_params(params):
    values = {}
    for item in params.split(";"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        try:
            values[key] = float(value)
        except ValueError:
            continue
    return values


def mark_peak(ax, x, envelope, label="p"):
    peak_index = int(np.argmax(envelope))
    mark_vertical(
        ax,
        float(x[peak_index]),
        label,
        y_value=float(envelope[peak_index]),
        xytext=(7, 0),
        ha="left",
        va="center",
    )


def mark_beta_sides(ax, x, envelope):
    peak_index = int(np.argmax(envelope))
    if peak_index > 2:
        add_letter(ax, float(x[peak_index // 2]), envelope[peak_index // 2], "a")
    if peak_index < len(envelope) - 3:
        right_index = peak_index + (len(envelope) - peak_index) // 2
        add_letter(ax, float(x[right_index]), envelope[right_index], "b")


def mark_spline_effects(ax, x, envelope, params, metadata):
    values = parse_params(params)
    min_control_value = values.get("min_control_value")
    if min_control_value is not None:
        mark_horizontal(ax, min_control_value, "m")

    mark_peak(ax, x, envelope)
    mark_control_points(ax, metadata.get("control_points", []))


def mark_vertical(
    ax,
    x_value,
    label,
    y_value=0.96,
    xytext=(0, 7),
    ha="center",
    va="bottom",
):
    ax.axvline(x_value, color="#111827", linewidth=1.2, linestyle="--", alpha=0.65)
    add_letter(ax, x_value, y_value, label, xytext=xytext, ha=ha, va=va)


def mark_horizontal(ax, y_value, label):
    ax.axhline(y_value, color="#111827", linewidth=1.2, linestyle=":", alpha=0.65)
    add_letter(ax, 0.90, y_value, label)


def mark_control_points(ax, control_points):
    if not control_points:
        return
    x_values, y_values = zip(*control_points)
    ax.scatter(
        x_values,
        y_values,
        color="#111827",
        marker="o",
        s=12,
        zorder=5,
    )


def mark_window_length(ax):
    y_value = 0.06
    ax.annotate(
        "",
        xy=(1.0, y_value),
        xytext=(0.0, y_value),
        arrowprops={
            "arrowstyle": "<->",
            "color": "#111827",
            "linewidth": 1.2,
            "alpha": 0.75,
        },
    )
    add_letter(ax, 0.5, y_value, "n")


def add_letter(
    ax,
    x_value,
    y_value,
    label,
    xytext=(0, 7),
    ha="center",
    va="bottom",
):
    ax.scatter(
        [x_value],
        [y_value],
        color="#111827",
        marker=PARAMETER_MARKERS[label],
        s=24,
        zorder=5,
    )
    ax.annotate(
        PARAMETER_SYMBOLS[label],
        (x_value, y_value),
        xytext=xytext,
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=10,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot representative attack envelope shapes from 3_attackgen/mod_tuner.py."
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_ROWS,
        help=f"Number of points in each envelope. Default: {DEFAULT_ROWS}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed for deterministic random envelope parameters. Default: {DEFAULT_SEED}.",
    )
    parser.add_argument(
        "--save",
        help="Optional output image path, e.g. envelope.png.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.rows <= 0:
        raise ValueError("--rows must be greater than 0")

    fig = plot_envelopes(args.rows, args.seed)
    if args.save:
        fig.savefig(args.save, dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    main()
