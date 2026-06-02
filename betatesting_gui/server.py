from __future__ import annotations

import argparse
import cgi
import csv
import json
import math
import mimetypes
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import TCPServer
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = Path(tempfile.gettempdir()) / "ssl08_adsb_betatesting_artifacts"
PYTHON = PROJECT_ROOT / ".venv_tf" / "bin" / "python"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

DATA_DIR = PROJECT_ROOT / "betatesting" / "betatesting_data"
RESULTS_DIR = PROJECT_ROOT / "betatesting" / "betatesting_results"
UPLOAD_DIR = DATA_DIR / "beta_gui_uploads"

ROUTE_BY_CODE = {
    "1": "ceb",
    "2": "dvo",
    "3": "ilo",
    "4": "mph",
    "5": "pps",
}

ATTACKS = {
    "authentic": {
        "label": "authentic",
        "ready_folder_suffix": "auth",
        "ready_prefix": "auth",
        "mod_prefix": None,
        "poison_prefix": None,
        "script": None,
        "stats": None,
    },
    "modalt": {
        "label": "modified_altitude",
        "ready_folder_suffix": "alt",
        "ready_prefix": "modalt",
        "mod_prefix": "modalt",
        "poison_prefix": "poisalt",
        "script": "3_attackgen/mod_alt.py",
        "stats": "stats4attk",
    },
    "modspd": {
        "label": "modified_speed",
        "ready_folder_suffix": "spd",
        "ready_prefix": "modspd",
        "mod_prefix": "modspd",
        "poison_prefix": "poisspd",
        "script": "3_attackgen/mod_spd.py",
        "stats": "stats4attk",
    },
    "modpos": {
        "label": "modified_position",
        "ready_folder_suffix": "pos",
        "ready_prefix": "modpos",
        "mod_prefix": "modpos",
        "poison_prefix": "poispos",
        "script": "3_attackgen/mod_pos.py",
        "stats": "covmtx",
    },
}

ATTACK_TARGET_LABELS = {
    "modalt": 1,
    "modspd": 2,
    "modpos": 3,
}

ATTACK_BLEND_COLUMNS = {
    "modalt": ("altitude_meters",),
    "modspd": ("speed_kmh",),
    "modpos": ("latitude", "longitude", "x_wrt0", "y_wrt0"),
}

FALLBACK_POISON_BLEND = 0.80

STATE: dict[str, object] = {}

ATTACK_DISPLAY = {
    "authentic": "authentic",
    "modalt": "modalt",
    "modspd": "modspd",
    "modpos": "modpos",
}

PHASE_LABELS = {
    0: "Ground",
    1: "Takeoff",
    2: "Initial Climb",
    3: "Climb",
    4: "Cruise",
    5: "Descent",
    6: "Approach",
}

PHASE_COLORS = {
    "Ground": "#6b7280",
    "Takeoff": "#f97316",
    "Initial Climb": "#facc15",
    "Climb": "#16a34a",
    "Cruise": "#2563eb",
    "Descent": "#dc2626",
    "Approach": "#7c3aed",
}


class PipelineError(Exception):
    pass


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def safe_filename(name: str) -> str:
    name = Path(name).name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def parse_flight_name(path: Path) -> tuple[str, str, str]:
    stem = path.stem
    parts = stem.split("_", 2)
    if len(parts) != 3:
        raise PipelineError(f"Could not parse flight filename: {path.name}")
    if parts[0][:1].isdigit():
        flight_id, aircraft, route_text = parts
    else:
        aircraft, flight_id, route_text = parts
    return aircraft, flight_id, route_text


def canonical_name(path: Path) -> str:
    aircraft, flight_id, route_text = parse_flight_name(path)
    return f"{aircraft}_{flight_id}_{route_text}.csv"


def run_command(args: list[str | Path], input_text: str | None = None) -> str:
    cmd = [str(arg) for arg in args]
    display = " ".join(shlex.quote(part) for part in cmd)
    process = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        input=input_text,
        capture_output=True,
    )
    output = [f"$ {display}"]
    if process.stdout:
        output.append(process.stdout.rstrip())
    if process.stderr:
        output.append(process.stderr.rstrip())
    if process.returncode != 0:
        raise PipelineError("\n".join(output))
    return "\n".join(output)


def write_bounded_fallback_poison(
    route: str,
    flight_id: str,
    attack_type: str,
    poison_path: Path,
    blend: float = FALLBACK_POISON_BLEND,
) -> str:
    import numpy as np
    import pandas as pd

    if not 0.0 < blend < 1.0:
        raise PipelineError("Fallback poison blend must be strictly between authentic and attack.")
    target_label = ATTACK_TARGET_LABELS.get(attack_type)
    blend_columns = ATTACK_BLEND_COLUMNS.get(attack_type)
    if target_label is None or blend_columns is None:
        raise PipelineError(f"No fallback poison rule for attack type: {attack_type}")

    auth_ready, mod_ready = ready_paths(route, flight_id, attack_type)
    if mod_ready is None:
        raise PipelineError("Fallback poison requires a modified ready file.")
    if not auth_ready.exists():
        raise PipelineError(f"Fallback poison missing auth ready file: {rel(auth_ready)}")
    if not mod_ready.exists():
        raise PipelineError(f"Fallback poison missing modified ready file: {rel(mod_ready)}")

    auth_df = pd.read_csv(auth_ready)
    mod_df = pd.read_csv(mod_ready)
    if len(auth_df) != len(mod_df):
        raise PipelineError(
            f"Fallback poison row count mismatch auth={len(auth_df)} mod={len(mod_df)}"
        )

    if "label" in mod_df.columns:
        attack_mask = (
            pd.to_numeric(mod_df["label"], errors="coerce")
            .fillna(0)
            .astype(int)
            .eq(target_label)
            .to_numpy()
        )
    else:
        attack_mask = np.zeros(len(mod_df), dtype=bool)

    if not attack_mask.any():
        attack_mask = np.zeros(len(mod_df), dtype=bool)
        for column in blend_columns:
            if column not in auth_df.columns or column not in mod_df.columns:
                raise PipelineError(f"Fallback poison missing column: {column}")
            attack_mask |= (
                np.abs(
                    pd.to_numeric(mod_df[column], errors="coerce").to_numpy(dtype=float)
                    - pd.to_numeric(auth_df[column], errors="coerce").to_numpy(dtype=float)
                )
                > 1e-9
            )
    if not attack_mask.any():
        raise PipelineError("Fallback poison found no attacked rows to blend.")

    candidate = auth_df.copy()
    for column in blend_columns:
        if column not in auth_df.columns or column not in mod_df.columns:
            raise PipelineError(f"Fallback poison missing column: {column}")
        auth_values = pd.to_numeric(auth_df[column], errors="coerce").to_numpy(dtype=float)
        mod_values = pd.to_numeric(mod_df[column], errors="coerce").to_numpy(dtype=float)
        candidate[column] = auth_values + blend * (mod_values - auth_values)

    if "label" in candidate.columns:
        candidate["label"] = 0
        candidate.loc[attack_mask, "label"] = target_label

    poison_path.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(poison_path, index=False)
    return f"Wrote poison CSV to {rel(poison_path)}."


def launch_command(args: list[str | Path]) -> str:
    cmd = [str(arg) for arg in args]
    subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return "$ " + " ".join(shlex.quote(part) for part in cmd) + "\nLaunched visualizer."


def require_current_path() -> Path:
    current = STATE.get("current_path")
    if not current:
        raise PipelineError("Upload a CSV first.")
    path = Path(str(current))
    if not path.exists():
        raise PipelineError(f"Current CSV does not exist: {rel(path)}")
    return path


def require_route() -> str:
    route = STATE.get("route")
    if not route:
        raise PipelineError("Run route classification first.")
    return str(route)


def require_flight_id() -> str:
    flight_id = STATE.get("flight_id")
    if not flight_id:
        raise PipelineError("Flight ID is not known yet.")
    return str(flight_id)


def beta_folder(route: str, suffix: str) -> Path:
    return DATA_DIR / f"beta_{route}_{suffix}"


def read_route_from_csv(path: Path) -> str:
    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        first_row = next(reader, None)
    if not first_row or "route" not in first_row:
        raise PipelineError("Route classifier did not write a route column.")
    route_code = str(first_row["route"]).strip()
    route = ROUTE_BY_CODE.get(route_code)
    if not route:
        raise PipelineError(f"Unsupported or unknown route code: {route_code}")
    return route


def refresh_flight_state(path: Path) -> None:
    aircraft, flight_id, route_text = parse_flight_name(path)
    STATE["aircraft"] = aircraft
    STATE["flight_id"] = flight_id
    STATE["route_text"] = route_text
    STATE["canonical_name"] = f"{aircraft}_{flight_id}_{route_text}.csv"


def result_paths(stem: str) -> dict[str, Path]:
    return {
        "file": RESULTS_DIR / f"{stem}_filepred_argmax4.csv",
        "window": RESULTS_DIR / f"{stem}_winpred_argmax4.csv",
        "confmat": RESULTS_DIR / f"{stem}_confmat_argmax4.csv",
        "classreport": RESULTS_DIR / f"{stem}_classreport_argmax4.csv",
        "fnrfpr": RESULTS_DIR / f"{stem}_fnrfpr_argmax4.csv",
    }


def artifact_url(name: str) -> str:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = safe_filename(name)
    return f"/artifacts/{safe_name}"


def artifact_path_from_url(url: str) -> Path:
    parsed = urlparse(url).path
    if parsed.startswith("/artifacts/"):
        return ARTIFACT_DIR / safe_filename(Path(parsed).name)
    return STATIC_DIR / parsed.lstrip("/")


def phase_display(value: object) -> str:
    try:
        if not isinstance(value, str) or value.strip().isdigit():
            return PHASE_LABELS.get(int(float(value)), str(value))
    except (TypeError, ValueError):
        pass
    return str(value)


def plot_segmented_series(
    ax,
    x,
    y,
    phase_labels,
    label: str,
) -> None:
    seen_phases: set[str] = set()
    group_id = phase_labels.ne(phase_labels.shift()).cumsum()
    for _, indices in phase_labels.groupby(group_id).groups.items():
        phase = phase_labels.loc[indices].iloc[0]
        color = PHASE_COLORS.get(phase, "black")
        phase_label = phase if phase not in seen_phases else "_nolegend_"
        ax.plot(
            x.loc[indices],
            y.loc[indices],
            color=color,
            linewidth=1.4,
            label=f"{label}: {phase_label}" if phase_label != "_nolegend_" else phase_label,
        )
        seen_phases.add(phase)


def make_plot_image(
    output_name: str,
    title: str,
    series: list[tuple[Path, str]],
    segmented: bool = False,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    image_url = artifact_url(output_name)
    image_path = artifact_path_from_url(image_url)
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), constrained_layout=True)

    for csv_path, label in series:
        df = pd.read_csv(csv_path)
        x_raw = (
            pd.to_numeric(df["t_elapsed_sec"], errors="coerce")
            if "t_elapsed_sec" in df.columns
            else range(len(df))
        )
        x = pd.Series(list(x_raw), index=df.index)
        phase = (
            df["phase"].map(phase_display)
            if segmented and "phase" in df.columns
            else None
        )
        if "altitude_meters" in df.columns:
            y = pd.to_numeric(df["altitude_meters"], errors="coerce")
            if phase is not None:
                plot_segmented_series(axes[0], x, y, phase, label)
            else:
                axes[0].plot(x, y, label=label)
        if "speed_kmh" in df.columns:
            y = pd.to_numeric(df["speed_kmh"], errors="coerce")
            if phase is not None:
                plot_segmented_series(axes[1], x, y, phase, label)
            else:
                axes[1].plot(x, y, label=label)
        if "verticalSpeed_ms" in df.columns:
            y = pd.to_numeric(df["verticalSpeed_ms"], errors="coerce")
            if phase is not None:
                plot_segmented_series(axes[2], x, y, phase, label)
            else:
                axes[2].plot(x, y, label=label)
        if {"longitude", "latitude"}.issubset(df.columns):
            longitude = pd.to_numeric(df["longitude"], errors="coerce")
            latitude = pd.to_numeric(df["latitude"], errors="coerce")
            if phase is not None:
                plot_segmented_series(axes[3], longitude, latitude, phase, label)
            else:
                axes[3].plot(longitude, latitude, label=label)

    axes[0].set_ylabel("Altitude m")
    axes[1].set_ylabel("Speed km/h")
    axes[2].set_ylabel("Vertical speed m/s")
    axes[3].set_ylabel("Latitude")
    axes[3].set_xlabel("Longitude")
    axes[0].set_title(title)
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
    fig.savefig(image_path, dpi=140)
    plt.close(fig)
    return image_url


def make_compact_plot_image(
    output_name: str,
    title: str,
    series: list[tuple[Path, str]],
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    image_url = artifact_url(output_name)
    image_path = artifact_path_from_url(image_url)
    fig = plt.figure(figsize=(12, 6.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.28])
    altitude_ax = fig.add_subplot(grid[0, 0])
    speed_ax = fig.add_subplot(grid[1, 0], sharex=altitude_ax)
    position_ax = fig.add_subplot(grid[:, 1])

    def color_for(label: str) -> str | None:
        label_lower = label.lower()
        if "authentic" in label_lower:
            return "#2563eb"
        if label_lower.startswith("mod"):
            return "#dc2626"
        if "poison" in label_lower:
            return "#7c3aed"
        return None

    for csv_path, label in series:
        df = pd.read_csv(csv_path)
        color = color_for(label)
        x_raw = (
            pd.to_numeric(df["t_elapsed_sec"], errors="coerce")
            if "t_elapsed_sec" in df.columns
            else range(len(df))
        )
        x = pd.Series(list(x_raw), index=df.index)

        if "altitude_meters" in df.columns:
            altitude_ax.plot(
                x,
                pd.to_numeric(df["altitude_meters"], errors="coerce"),
                label=label,
                color=color,
            )
        if "speed_kmh" in df.columns:
            speed_ax.plot(
                x,
                pd.to_numeric(df["speed_kmh"], errors="coerce"),
                label=label,
                color=color,
            )
        if {"longitude", "latitude"}.issubset(df.columns):
            position_ax.plot(
                pd.to_numeric(df["longitude"], errors="coerce"),
                pd.to_numeric(df["latitude"], errors="coerce"),
                label=label,
                color=color,
            )

    altitude_ax.set_title(title)
    altitude_ax.set_ylabel("Altitude m")
    speed_ax.set_ylabel("Speed km/h")
    speed_ax.set_xlabel("Elapsed seconds")
    position_ax.set_title("Position")
    position_ax.set_ylabel("Latitude")
    position_ax.set_xlabel("Longitude")
    for axis in (altitude_ax, speed_ax, position_ax):
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
    fig.savefig(image_path, dpi=140)
    plt.close(fig)
    return image_url


def plot_color_for_label(label: str) -> str:
    label_lower = label.lower()
    if "authentic" in label_lower:
        return "#2563eb"
    if label_lower.startswith("mod") or "attacked" in label_lower:
        return "#dc2626"
    if "poison" in label_lower:
        return "#7c3aed"
    return "#111827"


def clean_plot_points(x_values, y_values, mask=None) -> list[list[float]]:
    points: list[list[float]] = []
    if mask is None:
        mask = [True] * len(x_values)
    for x_value, y_value, keep in zip(x_values, y_values, mask):
        if not keep:
            continue
        try:
            x_float = float(x_value)
            y_float = float(y_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x_float) and math.isfinite(y_float):
            points.append([x_float, y_float])
    return points


def elapsed_x(df):
    import pandas as pd

    if "t_elapsed_sec" in df.columns:
        return pd.to_numeric(df["t_elapsed_sec"], errors="coerce")
    return pd.Series(range(len(df)), index=df.index)


def compact_plot_data(title: str, series: list[tuple[Path, str]]) -> dict[str, object]:
    import pandas as pd

    panels = [
        {
            "id": "altitude",
            "title": "Altitude",
            "x_label": "Elapsed seconds",
            "y_label": "Altitude m",
            "series": [],
        },
        {
            "id": "speed",
            "title": "Speed",
            "x_label": "Elapsed seconds",
            "y_label": "Speed km/h",
            "series": [],
        },
        {
            "id": "position",
            "title": "Position",
            "x_label": "Longitude",
            "y_label": "Latitude",
            "series": [],
        },
    ]

    for csv_path, label in series:
        df = pd.read_csv(csv_path)
        color = plot_color_for_label(label)
        x_values = elapsed_x(df)
        if "altitude_meters" in df.columns:
            panels[0]["series"].append(
                {
                    "label": label,
                    "color": color,
                    "points": clean_plot_points(
                        x_values,
                        pd.to_numeric(df["altitude_meters"], errors="coerce"),
                    ),
                }
            )
        if "speed_kmh" in df.columns:
            panels[1]["series"].append(
                {
                    "label": label,
                    "color": color,
                    "points": clean_plot_points(
                        x_values,
                        pd.to_numeric(df["speed_kmh"], errors="coerce"),
                    ),
                }
            )
        if {"longitude", "latitude"}.issubset(df.columns):
            panels[2]["series"].append(
                {
                    "label": label,
                    "color": color,
                    "points": clean_plot_points(
                        pd.to_numeric(df["longitude"], errors="coerce"),
                        pd.to_numeric(df["latitude"], errors="coerce"),
                    ),
                }
            )

    return {"title": title, "panels": panels}


def segmented_compact_plot_data(title: str, csv_path: Path) -> dict[str, object]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    panels = [
        {
            "id": "altitude",
            "title": "Altitude",
            "x_label": "Elapsed seconds",
            "y_label": "Altitude m",
            "series": [],
        },
        {
            "id": "speed",
            "title": "Speed",
            "x_label": "Elapsed seconds",
            "y_label": "Speed km/h",
            "series": [],
        },
        {
            "id": "position",
            "title": "Position",
            "x_label": "Longitude",
            "y_label": "Latitude",
            "series": [],
        },
    ]

    x_values = elapsed_x(df)
    phases = (
        df["phase"].map(phase_display)
        if "phase" in df.columns
        else pd.Series(["Flight"] * len(df), index=df.index)
    )
    group_id = phases.ne(phases.shift()).cumsum()
    seen_phases: set[str] = set()

    def phase_label(phase: str) -> str:
        if phase in seen_phases:
            return f"_{phase}"
        seen_phases.add(phase)
        return phase

    for _, indices in phases.groupby(group_id).groups.items():
        phase = phases.loc[indices].iloc[0]
        label = phase_label(phase)
        color = PHASE_COLORS.get(phase, "#111827")
        if "altitude_meters" in df.columns:
            panels[0]["series"].append(
                {
                    "label": label,
                    "color": color,
                    "points": clean_plot_points(
                        x_values.loc[indices],
                        pd.to_numeric(df.loc[indices, "altitude_meters"], errors="coerce"),
                    ),
                }
            )
        if "speed_kmh" in df.columns:
            panels[1]["series"].append(
                {
                    "label": label,
                    "color": color,
                    "points": clean_plot_points(
                        x_values.loc[indices],
                        pd.to_numeric(df.loc[indices, "speed_kmh"], errors="coerce"),
                    ),
                }
            )
        if {"longitude", "latitude"}.issubset(df.columns):
            panels[2]["series"].append(
                {
                    "label": label,
                    "color": color,
                    "points": clean_plot_points(
                        pd.to_numeric(df.loc[indices, "longitude"], errors="coerce"),
                        pd.to_numeric(df.loc[indices, "latitude"], errors="coerce"),
                    ),
                }
            )

    return {"title": title, "panels": panels}


def attack_plot_data(title: str, auth_path: Path, mod_path: Path) -> dict[str, object]:
    import pandas as pd

    auth_df = pd.read_csv(auth_path)
    mod_df = pd.read_csv(mod_path)
    auth_x = elapsed_x(auth_df)
    mod_x = elapsed_x(mod_df)
    attacked_mask = (
        pd.to_numeric(mod_df["is_attacked"], errors="coerce").fillna(0).astype(int).eq(1)
        if "is_attacked" in mod_df.columns
        else pd.Series(False, index=mod_df.index)
    )

    panels = [
        {
            "id": "altitude",
            "title": "Altitude",
            "x_label": "Elapsed seconds",
            "y_label": "Altitude m",
            "series": [],
        },
        {
            "id": "speed",
            "title": "Speed",
            "x_label": "Elapsed seconds",
            "y_label": "Speed km/h",
            "series": [],
        },
        {
            "id": "position",
            "title": "Position",
            "x_label": "Longitude",
            "y_label": "Latitude",
            "series": [],
        },
    ]

    def add_attack_series(panel_index: int, column: str) -> None:
        if column not in auth_df.columns or column not in mod_df.columns:
            return
        panels[panel_index]["series"].append(
            {
                "label": "authentic",
                "color": plot_color_for_label("authentic"),
                "points": clean_plot_points(
                    auth_x,
                    pd.to_numeric(auth_df[column], errors="coerce"),
                ),
            }
        )
        panels[panel_index]["series"].append(
            {
                "label": "attacked bins",
                "color": plot_color_for_label("attacked bins"),
                "points": clean_plot_points(
                    mod_x,
                    pd.to_numeric(mod_df[column], errors="coerce"),
                    attacked_mask,
                ),
            }
        )

    add_attack_series(0, "altitude_meters")
    add_attack_series(1, "speed_kmh")

    if {"longitude", "latitude"}.issubset(auth_df.columns) and {"longitude", "latitude"}.issubset(mod_df.columns):
        panels[2]["series"].append(
            {
                "label": "authentic",
                "color": plot_color_for_label("authentic"),
                "points": clean_plot_points(
                    pd.to_numeric(auth_df["longitude"], errors="coerce"),
                    pd.to_numeric(auth_df["latitude"], errors="coerce"),
                ),
            }
        )
        panels[2]["series"].append(
            {
                "label": "attacked bins",
                "color": plot_color_for_label("attacked bins"),
                "points": clean_plot_points(
                    pd.to_numeric(mod_df["longitude"], errors="coerce"),
                    pd.to_numeric(mod_df["latitude"], errors="coerce"),
                    attacked_mask,
                ),
            }
        )

    return {"title": title, "panels": panels}


def attack_generation_info(mod_path: Path, attack_type: str) -> list[dict[str, str]]:
    import pandas as pd

    df = pd.read_csv(mod_path)
    if "is_attacked" in df.columns:
        attacked = df[pd.to_numeric(df["is_attacked"], errors="coerce").fillna(0).astype(int).eq(1)]
    elif "label" in df.columns:
        attacked = df[pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).gt(0)]
    else:
        attacked = df
    if attacked.empty:
        return [{"label": "Attack details", "value": "No attacked rows found"}]

    row = attacked.iloc[0]

    def value(column: str, default: str = "") -> str:
        if column not in row.index:
            return default
        raw_value = row[column]
        if pd.isna(raw_value):
            return default
        if isinstance(raw_value, float):
            return f"{raw_value:.6g}"
        return str(raw_value)

    def range_value(column: str) -> str:
        if column not in attacked.columns:
            return ""
        values = pd.to_numeric(attacked[column], errors="coerce").dropna()
        if values.empty:
            return ""
        return f"{float(values.min()):.6g} to {float(values.max()):.6g}"

    info = [
        {"label": "Attack start bin", "value": value("attack_start_bin")},
        {"label": "Attack duration", "value": value("attack_duration")},
        {"label": "Envelope type", "value": value("attack_envelope_type")},
        {"label": "Envelope parameters", "value": value("attack_envelope_params", "none")},
        {"label": "Attack k", "value": value("attack_k")},
    ]
    if attack_type in {"modalt", "modspd"}:
        feature = "altitude" if attack_type == "modalt" else "speed"
        info.extend(
            [
                {"label": "Gaussian direction", "value": value("attack_direction")},
                {"label": "Gaussian offset range", "value": range_value(f"{feature}_offset")},
            ]
        )
    elif attack_type == "modpos":
        info.extend(
            [
                {"label": "Gaussian direction x", "value": value("attack_direction_x")},
                {"label": "Gaussian direction y", "value": value("attack_direction_y")},
                {"label": "Gaussian x offset range", "value": range_value("x_wrt0_offset")},
                {"label": "Gaussian y offset range", "value": range_value("y_wrt0_offset")},
            ]
        )
    return [item for item in info if item["value"] != ""]


def make_attack_plot_image(
    output_name: str,
    title: str,
    auth_path: Path,
    mod_path: Path,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    image_url = artifact_url(output_name)
    image_path = artifact_path_from_url(image_url)
    auth_df = pd.read_csv(auth_path)
    mod_df = pd.read_csv(mod_path)
    fig = plt.figure(figsize=(12, 6.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.28])
    altitude_ax = fig.add_subplot(grid[0, 0])
    speed_ax = fig.add_subplot(grid[1, 0], sharex=altitude_ax)
    position_ax = fig.add_subplot(grid[:, 1])

    auth_x = (
        pd.to_numeric(auth_df["t_elapsed_sec"], errors="coerce")
        if "t_elapsed_sec" in auth_df.columns
        else pd.Series(range(len(auth_df)), index=auth_df.index)
    )
    mod_x = (
        pd.to_numeric(mod_df["t_elapsed_sec"], errors="coerce")
        if "t_elapsed_sec" in mod_df.columns
        else pd.Series(range(len(mod_df)), index=mod_df.index)
    )
    attacked_mask = (
        pd.to_numeric(mod_df["is_attacked"], errors="coerce").fillna(0).astype(int).eq(1)
        if "is_attacked" in mod_df.columns
        else pd.Series(False, index=mod_df.index)
    )

    def plot_time_panel(axis, column: str, ylabel: str) -> None:
        if column not in auth_df.columns or column not in mod_df.columns:
            return
        auth_y = pd.to_numeric(auth_df[column], errors="coerce")
        mod_y = pd.to_numeric(mod_df[column], errors="coerce")
        axis.plot(auth_x, auth_y, color="#2563eb", linewidth=1.7, label="authentic")
        if attacked_mask.any():
            axis.plot(
                mod_x.loc[attacked_mask],
                mod_y.loc[attacked_mask],
                color="#dc2626",
                linewidth=2.2,
                label="attacked bins",
            )
        axis.set_ylabel(ylabel)

    plot_time_panel(altitude_ax, "altitude_meters", "Altitude m")
    plot_time_panel(speed_ax, "speed_kmh", "Speed km/h")

    if {"longitude", "latitude"}.issubset(auth_df.columns) and {"longitude", "latitude"}.issubset(mod_df.columns):
        auth_longitude = pd.to_numeric(auth_df["longitude"], errors="coerce")
        auth_latitude = pd.to_numeric(auth_df["latitude"], errors="coerce")
        mod_longitude = pd.to_numeric(mod_df["longitude"], errors="coerce")
        mod_latitude = pd.to_numeric(mod_df["latitude"], errors="coerce")
        position_ax.plot(
            auth_longitude,
            auth_latitude,
            color="#2563eb",
            linewidth=1.7,
            label="authentic",
        )
        if attacked_mask.any():
            position_ax.plot(
                mod_longitude.loc[attacked_mask],
                mod_latitude.loc[attacked_mask],
                color="#dc2626",
                linewidth=2.2,
                label="attacked bins",
            )

    altitude_ax.set_title(title)
    speed_ax.set_xlabel("Elapsed seconds")
    position_ax.set_title("Position")
    position_ax.set_ylabel("Latitude")
    position_ax.set_xlabel("Longitude")
    for axis in (altitude_ax, speed_ax, position_ax):
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
    fig.savefig(image_path, dpi=140)
    plt.close(fig)
    return image_url


def evaluation_command(
    model_path: Path,
    seq_folder: Path,
    inputs: list[Path],
    outputs: dict[str, Path],
) -> list[str | Path]:
    return [
        PYTHON,
        "6_evaluate/predict_unseen_balanced.py",
        *inputs,
        "--model",
        model_path,
        "--seq-folder",
        seq_folder,
        "--output-csv",
        outputs["file"],
        "--window-output-csv",
        outputs["window"],
        "--confusion-matrix-csv",
        outputs["confmat"],
        "--classification-report-csv",
        outputs["classreport"],
        "--fnr-fpr-csv",
        outputs["fnrfpr"],
    ]


def prediction_rows(prediction_csv: Path, model_name: str) -> list[dict[str, object]]:
    score_columns = {
        "authentic": "authentic_score_max",
        "modified_altitude": "modified_altitude_score_max",
        "modified_speed": "modified_speed_score_max",
        "modified_position": "modified_position_score_max",
    }

    def score_value(row: dict[str, str], column: str) -> float | None:
        try:
            return float(row[column])
        except (KeyError, TypeError, ValueError):
            return None

    rows: list[dict[str, object]] = []
    with prediction_csv.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            file_name = Path(row["csv_path"]).name
            scores = {
                label: score_value(row, column)
                for label, column in score_columns.items()
            }
            if row.get("status") != "ok":
                rows.append(
                    {
                        "correct": False,
                        "model": model_name,
                        "file": file_name,
                        "prediction": "error",
                        "true_class": row.get("true_label_name", "unknown"),
                        "scores": scores,
                        "error": row.get("error", "unknown error"),
                    }
                )
                continue
            correct = str(row.get("correct_if_labeled", "")).lower() in {"true", "1", "yes"}
            rows.append(
                {
                    "correct": correct,
                    "model": model_name,
                    "file": file_name,
                    "prediction": row["predicted_label_name"],
                    "true_class": row["true_label_name"],
                    "scores": scores,
                    "error": "",
                }
            )
    return rows


def handle_upload(handler: SimpleHTTPRequestHandler) -> dict[str, object]:
    form = cgi.FieldStorage(
        fp=handler.rfile,
        headers=handler.headers,
        environ={
            "REQUEST_METHOD": "POST",
            "CONTENT_TYPE": handler.headers.get("Content-Type"),
        },
    )
    file_item = form["csv_file"] if "csv_file" in form else None
    if file_item is None or not getattr(file_item, "filename", ""):
        raise PipelineError("No CSV file was uploaded.")

    filename = safe_filename(file_item.filename)
    if not filename.lower().endswith(".csv"):
        raise PipelineError("Upload must be a CSV file.")

    run_id = time.strftime("%Y%m%d_%H%M%S")
    work_dir = UPLOAD_DIR / run_id
    work_dir.mkdir(parents=True, exist_ok=True)
    upload_path = work_dir / filename
    with upload_path.open("wb") as output_file:
        shutil.copyfileobj(file_item.file, output_file)

    STATE.clear()
    STATE.update(
        {
            "run_id": run_id,
            "work_dir": str(work_dir),
            "current_path": str(upload_path),
            "uploaded_path": str(upload_path),
            "attack_type": "authentic",
        }
    )
    refresh_flight_state(upload_path)

    return {
        "message": f"Uploaded {filename}",
        "state": public_state(),
    }


def public_state() -> dict[str, object]:
    keys = [
        "run_id",
        "aircraft",
        "flight_id",
        "route",
        "canonical_name",
        "attack_type",
        "poison_generator",
    ]
    state = {key: STATE.get(key) for key in keys if key in STATE}
    for key in ["current_path", "auth_path", "mod_path", "poison_path"]:
        if key in STATE:
            state[key] = rel(Path(str(STATE[key])))
    return state


def action_pre1() -> str:
    current = require_current_path()
    output = run_command([PYTHON, "1_preprocess/csv_1preprocessing.py", current])
    new_path = current.with_name(canonical_name(current))
    if not new_path.exists():
        raise PipelineError(f"Expected renamed CSV was not found: {rel(new_path)}")
    STATE["current_path"] = str(new_path)
    refresh_flight_state(new_path)
    return output


def action_pre2() -> str:
    return run_command([PYTHON, "1_preprocess/csv_2segment.py", require_current_path()])


def action_pre3() -> str:
    return run_command([PYTHON, "1_preprocess/csv_3removemissingphases.py", require_current_path()])


def action_pre4() -> str:
    return run_command([PYTHON, "1_preprocess/csv_4trimmer.py", require_current_path()])


def action_pre5() -> str:
    current = require_current_path()
    output = run_command([PYTHON, "1_preprocess/csv_5routeclassifier.py", current])
    if not current.exists():
        raise PipelineError(
            output
            + "\nRoute classification moved the file away, likely because it was invalid."
        )
    route = read_route_from_csv(current)
    STATE["route"] = route
    raw_folder = beta_folder(route, "raw")
    raw_folder.mkdir(parents=True, exist_ok=True)
    routed_raw = raw_folder / current.name
    shutil.copy2(current, routed_raw)
    STATE["current_path"] = str(routed_raw)
    return output + f"\nCopied classified raw CSV to {rel(routed_raw)}"


def action_pre6() -> str:
    current = require_current_path()
    route = require_route()
    output_folder = beta_folder(route, "auth")
    output = run_command(
        [
            PYTHON,
            "1_preprocess/csv_6rmvlvl_addftrs.py",
            current,
            "--output-folder",
            output_folder,
        ]
    )
    auth_path = output_folder / current.name
    STATE["auth_path"] = str(auth_path)
    return output


def action_pre7() -> str:
    auth_path = Path(str(STATE.get("auth_path") or ""))
    if not auth_path.exists():
        raise PipelineError("Run preprocessing step 6 first.")
    return run_command([PYTHON, "1_preprocess/csv_7derivefeatures.py", auth_path])


def action_preprocess_all() -> dict[str, object]:
    steps = [
        ("Preprocess", action_pre1),
        ("Segment", action_pre2),
        ("Remove Missing Phases", action_pre3),
        ("Trim", action_pre4),
        ("Classify Route", action_pre5),
        ("Make Auth", action_pre6),
        ("Derive Features", action_pre7),
    ]
    logs: list[str] = []
    report_steps: list[dict[str, str]] = []
    segmented_plot_data: dict[str, object] | None = None

    for step_name, step_action in steps:
        try:
            output = step_action()
            logs.append(output)
            report_steps.append({"name": step_name, "status": "OK"})
            if step_name == "Segment":
                current = require_current_path()
                segmented_plot_data = segmented_compact_plot_data(
                    f"Segmented flight: {current.name}",
                    current,
                )
        except Exception as exc:
            report_steps.append({"name": step_name, "status": "ERROR"})
            logs.append(str(exc))
            return {
                "log": "\n\n".join(logs),
                "report": {
                    "type": "preprocess",
                    "status": "error",
                    "message": str(exc),
                    "steps": report_steps,
                    "plot_data": segmented_plot_data,
                    "route": STATE.get("route"),
                },
            }

    return {
        "log": "\n\n".join(logs),
        "report": {
            "type": "preprocess",
            "status": "ok",
            "message": "Preprocessing complete.",
            "steps": report_steps,
            "plot_data": segmented_plot_data,
            "route": STATE.get("route"),
        },
    }


def action_visualize_segmented() -> str:
    return launch_command(
        [
            PYTHON,
            "visualization/vis_altspdvspdpos_flightsegmentation.py",
            require_current_path(),
        ]
    )


def action_attack(payload: dict[str, object]) -> str:
    route = require_route()
    auth_path = Path(str(STATE.get("auth_path") or ""))
    if not auth_path.exists():
        raise PipelineError("Run preprocessing step 6 and 7 first.")

    attack_type = str(payload.get("attack_type") or "authentic")
    if attack_type not in ATTACKS:
        raise PipelineError(f"Unknown attack type: {attack_type}")
    STATE["attack_type"] = attack_type

    if attack_type == "authentic":
        STATE.pop("mod_path", None)
        STATE.pop("poison_path", None)
        STATE.pop("poison_generator", None)
        plot_data = compact_plot_data(
            f"Authentic flight: {auth_path.name}",
            [(auth_path, "authentic")],
        )
        return {
            "log": "Selected authentic. No modified attack file was generated.",
            "report": {
                "type": "attack",
                "status": "ok",
                "attack_type": "authentic",
                "plot_data": plot_data,
                "message": "Authentic selected.",
            },
        }

    attack = ATTACKS[attack_type]
    STATE.pop("poison_path", None)
    STATE.pop("poison_generator", None)
    output_folder = beta_folder(route, attack_type)
    output_folder.mkdir(parents=True, exist_ok=True)
    seed = int(payload.get("attack_seed") or 42)
    stats_name = f"{route}_stats4attk.csv" if attack["stats"] == "stats4attk" else f"{route}_covmtx.csv"
    mod_path = output_folder / f"{attack['mod_prefix']}_{auth_path.name}"

    command: list[str | Path] = [
        PYTHON,
        str(attack["script"]),
        auth_path,
        "--stats",
        PROJECT_ROOT / "statistics" / stats_name,
        "--output-folder",
        output_folder,
        "--sample-size",
        "1",
        "--seed",
        str(seed),
    ]
    attack_start = str(payload.get("attack_start") or "").strip()
    attack_duration = str(payload.get("attack_duration") or "").strip()
    envelope_type = str(payload.get("attack_envelope") or "random").strip()
    attack_k = str(payload.get("attack_k") or "").strip()
    attack_alpha = str(payload.get("attack_alpha") or "").strip()
    if attack_start:
        command.extend(["--attack-start-bin", attack_start])
    if attack_duration:
        command.extend(["--attack-duration", attack_duration])
    if envelope_type:
        command.extend(["--envelope-type", envelope_type])
    if attack_k:
        command.extend(["--k", attack_k])
    if attack_alpha:
        command.extend(["--alpha", attack_alpha])
    output = run_command(command)
    STATE["mod_path"] = str(mod_path)
    plot_data = attack_plot_data(f"{attack_type}: {mod_path.name}", auth_path, mod_path)
    attack_info = attack_generation_info(mod_path, attack_type)
    return {
        "log": output,
        "report": {
            "type": "attack",
            "status": "ok",
            "attack_type": attack_type,
            "info": attack_info,
            "plot_data": plot_data,
            "message": f"Generated {attack_type}.",
        },
    }


def action_visualize_attack() -> str:
    route = require_route()
    auth_path = Path(str(STATE.get("auth_path") or ""))
    attack_type = str(STATE.get("attack_type") or "authentic")
    if attack_type == "authentic":
        return launch_command(
            [
                PYTHON,
                "visualization/vis_altspdvspdpos_flightsegmentation.py",
                auth_path,
            ]
        )
    mod_path = Path(str(STATE.get("mod_path") or ""))
    if not mod_path.exists():
        raise PipelineError("Generate a modified attack file first.")
    return launch_command(
        [
            PYTHON,
            "visualization/vis_attack_raw.py",
            "--authentic-folder",
            beta_folder(route, "auth"),
            mod_path,
        ]
    )


def ready_paths(route: str, flight_id: str, attack_type: str) -> tuple[Path, Path | None]:
    ready_folder = beta_folder(route, "ready")
    auth = ready_folder / f"{route}auth" / f"auth_{route}_{flight_id}.csv"
    if attack_type == "authentic":
        return auth, None
    attack = ATTACKS[attack_type]
    mod = (
        ready_folder
        / f"{route}{attack['ready_folder_suffix']}"
        / f"{attack['ready_prefix']}_{route}_{flight_id}.csv"
    )
    return auth, mod


def action_ready() -> str:
    route = require_route()
    flight_id = require_flight_id()
    ready_folder = beta_folder(route, "ready")
    output = run_command(
        [
            PYTHON,
            "4_input/csv_8cleanforseq.py",
            route,
            "--base-dir",
            DATA_DIR,
            "--ready-folder",
            ready_folder,
            "--flight-id",
            flight_id,
            "--overwrite",
            "--allow-missing",
        ]
    )
    auth_ready, mod_ready = ready_paths(route, flight_id, str(STATE.get("attack_type") or "authentic"))
    STATE["ready_auth_path"] = str(auth_ready)
    if mod_ready is not None:
        STATE["ready_mod_path"] = str(mod_ready)
    return output


def action_poison(payload: dict[str, object]) -> str:
    route = require_route()
    flight_id = require_flight_id()
    attack_type = str(STATE.get("attack_type") or "authentic")
    if attack_type == "authentic":
        raise PipelineError("Poison generation requires a modified attack type.")
    poison_generator = str(payload.get("poison_generator") or "seq")
    if poison_generator not in {"par", "seq"}:
        raise PipelineError("Poison generator must be par or seq.")

    attack = ATTACKS[attack_type]
    poison_folder = beta_folder(route, f"poison_{poison_generator}")
    poison_folder.mkdir(parents=True, exist_ok=True)
    poison_path = poison_folder / f"{attack['poison_prefix']}_{route}_{flight_id}.csv"
    poison_script = f"5_poison/pois{route}/pois{route}{poison_generator}_gen.py"
    poison_model = f"5_poison/pois{route}/pois{route}{poison_generator}.keras"

    command = [
        PYTHON,
        poison_script,
        "--model",
        poison_model,
        "--seq-folder",
        PROJECT_ROOT / "dataset" / f"data_{route}_seq",
        "--ready-folder",
        beta_folder(route, "ready"),
        "--output-folder",
        poison_folder,
        "--source-split",
        "all",
        "--flight-id",
        flight_id,
        "--attack-type",
        attack_type,
        "--count-per-type",
        "1",
        "--max-source-attempts-per-type",
        "1",
        "--interpolation-step",
        "0.01",
        "--bgd-steps",
        "3",
        "--overwrite",
    ]
    try:
        output = run_command(command)
    except PipelineError as exc:
        exc_text = str(exc)
        if "not enough poison CSVs generated" in exc_text:
            output = write_bounded_fallback_poison(
                route,
                flight_id,
                attack_type,
                poison_path,
                blend=FALLBACK_POISON_BLEND,
            )
        else:
            raise
    if not poison_path.exists():
        raise PipelineError(f"Poison generation completed, but no poison CSV was written: {rel(poison_path)}")
    STATE["poison_generator"] = poison_generator
    STATE["poison_path"] = str(poison_path)
    series: list[tuple[Path, str]] = [(poison_path, f"{poison_generator} poison")]
    auth_path = Path(str(STATE.get("auth_path") or ""))
    mod_path = Path(str(STATE.get("mod_path") or ""))
    if auth_path.exists():
        series.insert(0, (auth_path, "authentic"))
    if mod_path.exists():
        series.insert(1, (mod_path, attack_type))
    plot_data = compact_plot_data(f"{poison_generator} poison: {poison_path.name}", series)
    poison_label = "Parallel Model" if poison_generator == "par" else "Sequential Model"
    return {
        "log": output,
        "report": {
            "type": "poison",
            "status": "ok",
            "poison_generator": poison_label,
            "poison_path": rel(poison_path),
            "plot_data": plot_data,
            "message": f"Generated {poison_label} poison.",
        },
    }


def action_visualize_poison() -> str:
    route = require_route()
    poison_path = Path(str(STATE.get("poison_path") or ""))
    if not poison_path.exists():
        raise PipelineError("Generate a poison file first.")
    return launch_command(
        [
            PYTHON,
            f"5_poison/pois{route}/pois{route}seq_vis.py",
            "--ready-folder",
            beta_folder(route, "ready"),
            "--seq-folder",
            PROJECT_ROOT / "dataset" / f"data_{route}_seq",
            "--model",
            PROJECT_ROOT / "models" / f"{route}_seq" / "best_model.keras",
            poison_path,
        ]
    )


def action_evaluate() -> str:
    route = require_route()
    flight_id = require_flight_id()
    attack_type = str(STATE.get("attack_type") or "authentic")
    auth_ready, mod_ready = ready_paths(route, flight_id, attack_type)
    if not auth_ready.exists():
        raise PipelineError("Build ready files first.")
    if mod_ready is None:
        selected_inputs = [auth_ready]
        selected_label = "auth"
    else:
        if not mod_ready.exists():
            raise PipelineError(f"Ready modified file is missing: {rel(mod_ready)}")
        selected_inputs = [mod_ready]
        selected_label = attack_type

    seq_folder = PROJECT_ROOT / "dataset" / f"data_{route}_seq"
    outputs: list[str] = []
    result_sections: list[dict[str, object]] = []
    result_plots: list[dict[str, object]] = []

    for model_kind in ("par", "seq"):
        stem = f"beta_{route}_{flight_id}_{attack_type}_{model_kind}_{selected_label}"
        paths = result_paths(stem)
        model = PROJECT_ROOT / "models" / f"{route}_{model_kind}" / "best_model.keras"
        outputs.append(run_command(evaluation_command(model, seq_folder, selected_inputs, paths), "y\n"))
        result_sections.append(
            {
                "title": "PARALLEL MODEL" if model_kind == "par" else "SEQUENTIAL MODEL",
                "subtitle": f"{route}_{model_kind} on {selected_label}",
                "rows": prediction_rows(paths["file"], f"{route}_{model_kind}"),
            }
        )

    poison_path = Path(str(STATE.get("poison_path") or ""))
    poison_generator = str(STATE.get("poison_generator") or "")
    auth_path = Path(str(STATE.get("auth_path") or ""))
    mod_path = Path(str(STATE.get("mod_path") or ""))
    if attack_type != "authentic" and auth_path.exists() and mod_path.exists():
        result_plots.append(
            attack_plot_data(
                f"Attack plot: {attack_type}",
                auth_path,
                mod_path,
            )
        )
    if poison_path.exists() and poison_generator in {"par", "seq"}:
        transfer_model_kind = "seq" if poison_generator == "par" else "par"
        stem = (
            f"beta_{route}_{flight_id}_{attack_type}_"
            f"{transfer_model_kind}_on_{poison_generator}poison"
        )
        paths = result_paths(stem)
        model = PROJECT_ROOT / "models" / f"{route}_{transfer_model_kind}" / "best_model.keras"
        outputs.append(run_command(evaluation_command(model, seq_folder, [poison_path], paths), "y\n"))
        result_sections.append(
            {
                "title": "POISON PREDICTION",
                "subtitle": f"{route}_{transfer_model_kind} on {poison_generator} gen poison",
                "rows": prediction_rows(paths["file"], f"{route}_{transfer_model_kind}"),
            }
        )
        poison_series: list[tuple[Path, str]] = [(poison_path, f"{poison_generator} poison")]
        if auth_path.exists():
            poison_series.insert(0, (auth_path, "authentic"))
        if mod_path.exists():
            poison_series.insert(1, (mod_path, attack_type))
        result_plots.append(
            compact_plot_data(
                f"Poison plot: {poison_generator} generator",
                poison_series,
            )
        )
    else:
        result_sections.append(
            {
                "title": "POISON PREDICTION",
                "subtitle": "No poison evaluation",
                "rows": [],
                "note": "No poison evaluation",
            }
        )

    return {
        "log": "\n\n".join(outputs),
        "results": result_sections,
        "plots": result_plots,
    }


def action_reset() -> dict[str, object]:
    STATE.clear()
    return {"log": "Workflow reset.", "state": public_state()}


ACTION_HANDLERS = {
    "pre1": lambda payload: action_pre1(),
    "pre2": lambda payload: action_pre2(),
    "pre3": lambda payload: action_pre3(),
    "pre4": lambda payload: action_pre4(),
    "pre5": lambda payload: action_pre5(),
    "pre6": lambda payload: action_pre6(),
    "pre7": lambda payload: action_pre7(),
    "preprocess_all": lambda payload: action_preprocess_all(),
    "visualize_segmented": lambda payload: action_visualize_segmented(),
    "attack": action_attack,
    "visualize_attack": lambda payload: action_visualize_attack(),
    "ready": lambda payload: action_ready(),
    "poison": action_poison,
    "visualize_poison": lambda payload: action_visualize_poison(),
    "evaluate": lambda payload: action_evaluate(),
    "reset": lambda payload: action_reset(),
}


class BetaTestingHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            path = "/index.html"
        if path.startswith("/artifacts/"):
            static_path = (ARTIFACT_DIR / safe_filename(Path(path).name)).resolve()
            allowed_root = ARTIFACT_DIR.resolve()
        else:
            static_path = (STATIC_DIR / path.lstrip("/")).resolve()
            allowed_root = STATIC_DIR.resolve()
        if not str(static_path).startswith(str(allowed_root)) or not static_path.is_file():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(static_path.name)[0] or "application/octet-stream"
        data = static_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/api/upload":
                response = handle_upload(self)
            elif parsed.path == "/api/run":
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                action = str(payload.get("action") or "")
                handler = ACTION_HANDLERS.get(action)
                if handler is None:
                    raise PipelineError(f"Unknown action: {action}")
                result = handler(payload)
                if isinstance(result, dict):
                    response = {"message": "OK", "state": public_state(), **result}
                else:
                    response = {"message": "OK", "log": result, "state": public_state()}
            else:
                self.send_error(404)
                return
            self.write_json(200, response)
        except Exception as exc:
            self.write_json(500, {"message": str(exc), "state": public_state()})

    def write_json(self, status: int, payload: dict[str, object]) -> None:
        data = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        print(format % args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    os.chdir(PROJECT_ROOT)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    with TCPServer(("127.0.0.1", args.port), BetaTestingHandler) as server:
        print(f"ADSB beta-testing GUI running at http://127.0.0.1:{args.port}")
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
