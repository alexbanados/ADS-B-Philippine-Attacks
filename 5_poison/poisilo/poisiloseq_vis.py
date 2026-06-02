from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ======================
# Tunable visual settings
# ======================

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
DEFAULT_MODEL_PATH = SCRIPT_DIR / "poisiloseq.keras"
DEFAULT_SEQ_FOLDER = ROOT_DIR / "dataset" / "data_ilo_seq"
DEFAULT_READY_FOLDER = ROOT_DIR / "dataset" / "data_ilo_ready"
DEFAULT_INPUT = ROOT_DIR / "dataset" / "poison" / "ilo" / "seq"
KERAS_BACKEND = "tensorflow"
BATCH_SIZE = 128

ATTACK_COLOR = "#dc2626"
POISON_COLOR = "#2563eb"
AUTH_COLOR = "#6b7280"
MOD_COLOR = "#16a34a"

LABEL_NAMES = {
    0: "authentic",
    1: "modified_altitude",
    2: "modified_speed",
    3: "modified_position",
}

POISON_SPECS = {
    "poisalt_ilo_": {
        "label": 1,
        "auth": ("iloauth", "auth_ilo_"),
        "mod": ("iloalt", "modalt_ilo_"),
    },
    "poisspd_ilo_": {
        "label": 2,
        "auth": ("iloauth", "auth_ilo_"),
        "mod": ("ilospd", "modspd_ilo_"),
    },
    "poispos_ilo_": {
        "label": 3,
        "auth": ("iloauth", "auth_ilo_"),
        "mod": ("ilopos", "modpos_ilo_"),
    },
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize generated ILO poison CSVs and optional model window scores."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[DEFAULT_INPUT],
        help="Poison CSV files or folders. Defaults to dataset/poison/ilo/seq.",
    )
    parser.add_argument("--ready-folder", type=Path, default=DEFAULT_READY_FOLDER)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--seq-folder", type=Path, default=DEFAULT_SEQ_FOLDER)
    parser.add_argument("--keras-backend", choices=("tensorflow", "jax", "torch"), default=KERAS_BACKEND)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--no-model", action="store_true", help="Skip model loading and probability plot.")
    parser.add_argument("--no-baselines", action="store_true", help="Do not overlay matching auth/mod CSVs.")
    parser.add_argument("--recursive", action="store_true", help="Search folders recursively for CSVs.")
    return parser.parse_args(argv)


def iter_csv_paths(inputs: list[Path], recursive: bool) -> list[Path]:
    csv_paths: list[Path] = []
    for input_path in inputs:
        if input_path.is_file():
            if input_path.suffix.lower() == ".csv":
                csv_paths.append(input_path)
        elif input_path.is_dir():
            pattern = "**/*.csv" if recursive else "*.csv"
            csv_paths.extend(sorted(path for path in input_path.glob(pattern) if path.is_file()))
        else:
            raise FileNotFoundError(f"input path does not exist: {input_path}")
    csv_paths = sorted(dict.fromkeys(csv_paths))
    if not csv_paths:
        raise ValueError("no CSV files found")
    return csv_paths


def import_pyplot():
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "matplotlib"))
    import matplotlib.pyplot as plt

    return plt


def import_model(model_path: Path, backend: str):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "matplotlib"))
    os.environ["KERAS_BACKEND"] = backend
    import keras

    return keras.models.load_model(model_path, compile=False)


def load_metadata(seq_folder: Path) -> dict:
    with (seq_folder / "metadata.json").open() as metadata_file:
        return json.load(metadata_file)


def load_scaler(seq_folder: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    scaler = np.load(seq_folder / "scaler.npz")
    mean = scaler["mean"].astype(np.float32)
    scale = scaler["scale"].astype(np.float32)
    scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
    feature_columns = [str(value) for value in scaler["feature_columns"].tolist()]
    return mean, scale, feature_columns


def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for column in df.columns:
        converted = pd.to_numeric(df[column], errors="coerce")
        if not converted.isna().any() or df[column].isna().all():
            df[column] = converted
    return df


def poison_source_paths(csv_path: Path, ready_folder: Path) -> tuple[Path | None, Path | None, int | None]:
    stem = csv_path.stem
    for poison_prefix, spec in POISON_SPECS.items():
        if not stem.startswith(poison_prefix):
            continue
        flight_id = stem[len(poison_prefix):]
        auth_folder, auth_prefix = spec["auth"]
        mod_folder, mod_prefix = spec["mod"]
        auth_path = ready_folder / auth_folder / f"{auth_prefix}{flight_id}.csv"
        mod_path = ready_folder / mod_folder / f"{mod_prefix}{flight_id}.csv"
        return (
            auth_path if auth_path.is_file() else None,
            mod_path if mod_path.is_file() else None,
            int(spec["label"]),
        )
    return None, None, None


def make_windows_with_midpoints(
    df: pd.DataFrame,
    feature_columns: list[str],
    mean: np.ndarray,
    scale: np.ndarray,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise ValueError(f"missing model feature columns: {missing}")
    values = df[feature_columns].copy()
    for column in feature_columns:
        values[column] = pd.to_numeric(values[column], errors="coerce")
    if values.isna().any().any():
        bad_columns = values.columns[values.isna().any()].tolist()
        raise ValueError(f"non-numeric or missing model feature values in {bad_columns}")

    scaled = ((values.to_numpy(dtype=np.float32) - mean) / scale).astype(np.float32)
    windows = []
    midpoints = []
    for start in range(0, len(df) - window_size + 1, stride):
        end = start + window_size
        windows.append(scaled[start:end])
        midpoints.append((start + end - 1) / 2)
    if not windows:
        return (
            np.empty((0, window_size, len(feature_columns)), dtype=np.float32),
            np.empty((0,), dtype=float),
        )
    return np.stack(windows), np.array(midpoints, dtype=float)


def predict_windows(
    model,
    df: pd.DataFrame,
    feature_columns: list[str],
    mean: np.ndarray,
    scale: np.ndarray,
    window_size: int,
    stride: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    windows, midpoints = make_windows_with_midpoints(
        df=df,
        feature_columns=feature_columns,
        mean=mean,
        scale=scale,
        window_size=window_size,
        stride=stride,
    )
    if len(windows) == 0:
        return np.empty((0, 4), dtype=float), midpoints
    probabilities = model.predict(windows, batch_size=batch_size, verbose=0)
    return probabilities, midpoints


def attacked_mask(df: pd.DataFrame) -> pd.Series:
    if "label" not in df.columns:
        return pd.Series(False, index=df.index)
    labels = pd.to_numeric(df["label"], errors="coerce").fillna(0)
    return labels.gt(0)


def shade_attacks(ax, mask: pd.Series) -> None:
    active = mask.to_numpy(dtype=bool)
    if not active.any():
        return
    start = None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        elif start is not None and not value:
            ax.axvspan(start, index - 1, color=ATTACK_COLOR, alpha=0.12)
            start = None
    if start is not None:
        ax.axvspan(start, len(active) - 1, color=ATTACK_COLOR, alpha=0.12)


def plot_series(ax, df: pd.DataFrame, column: str, label: str, color: str, style: str = "-") -> None:
    if column in df.columns:
        ax.plot(df.index, pd.to_numeric(df[column], errors="coerce"), style, color=color, linewidth=1.5, label=label)


def plot_csv(
    plt,
    csv_path: Path,
    ready_folder: Path,
    model_bundle: tuple[object, dict, np.ndarray, np.ndarray, list[str]] | None,
    batch_size: int,
    show_baselines: bool,
) -> None:
    poison_df = load_csv(csv_path)
    auth_path, mod_path, inferred_label = poison_source_paths(csv_path, ready_folder)
    auth_df = load_csv(auth_path) if show_baselines and auth_path is not None else None
    mod_df = load_csv(mod_path) if show_baselines and mod_path is not None else None
    mask = attacked_mask(poison_df)

    fig, axes = plt.subplots(4, 1, figsize=(13, 10))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.08, hspace=0.48)
    fig.suptitle(f"Poison CSV: {csv_path.name}", fontsize=14)

    plot_series(axes[0], poison_df, "altitude_meters", "poison", POISON_COLOR)
    if auth_df is not None:
        plot_series(axes[0], auth_df, "altitude_meters", "auth", AUTH_COLOR, "--")
    if mod_df is not None:
        plot_series(axes[0], mod_df, "altitude_meters", "mod", MOD_COLOR, "--")
    shade_attacks(axes[0], mask)
    axes[0].set_ylabel("Altitude m")
    axes[0].legend(loc="upper right")

    plot_series(axes[1], poison_df, "speed_kmh", "poison", POISON_COLOR)
    if auth_df is not None:
        plot_series(axes[1], auth_df, "speed_kmh", "auth", AUTH_COLOR, "--")
    if mod_df is not None:
        plot_series(axes[1], mod_df, "speed_kmh", "mod", MOD_COLOR, "--")
    shade_attacks(axes[1], mask)
    axes[1].set_ylabel("Speed km/h")
    axes[1].legend(loc="upper right")

    axes[2].plot(poison_df["longitude"], poison_df["latitude"], color=POISON_COLOR, linewidth=1.5, label="poison")
    if mask.any():
        axes[2].scatter(
            poison_df.loc[mask, "longitude"],
            poison_df.loc[mask, "latitude"],
            color=ATTACK_COLOR,
            s=10,
            label="poison rows",
        )
    if auth_df is not None:
        axes[2].plot(auth_df["longitude"], auth_df["latitude"], "--", color=AUTH_COLOR, linewidth=1.2, label="auth")
    if mod_df is not None:
        axes[2].plot(mod_df["longitude"], mod_df["latitude"], "--", color=MOD_COLOR, linewidth=1.2, label="mod")
    axes[2].set_ylabel("Latitude")
    axes[2].set_xlabel("Longitude")
    axes[2].legend(loc="best")

    if model_bundle is None:
        axes[3].text(0.5, 0.5, "Model probabilities disabled", ha="center", va="center", transform=axes[3].transAxes)
    else:
        model, metadata, mean, scale, feature_columns = model_bundle
        probabilities, midpoints = predict_windows(
            model=model,
            df=poison_df,
            feature_columns=feature_columns,
            mean=mean,
            scale=scale,
            window_size=int(metadata["window_size"]),
            stride=int(metadata["stride"]),
            batch_size=batch_size,
        )
        if len(probabilities) == 0:
            axes[3].text(0.5, 0.5, "CSV shorter than model window", ha="center", va="center", transform=axes[3].transAxes)
        else:
            for label_id, label_name in LABEL_NAMES.items():
                linewidth = 2.2 if inferred_label == label_id else 1.1
                axes[3].plot(midpoints, probabilities[:, label_id], linewidth=linewidth, label=label_name)
            axes[3].axhline(0.50, color="black", linewidth=1, linestyle=":")
            axes[3].set_ylim(0, 1)
            axes[3].set_ylabel("Window probability")
            axes[3].set_xlabel("Row")
            axes[3].legend(loc="upper right")

    print(f"  Plotted {csv_path.name} with {len(poison_df)} rows")
    print(f"  Poison rows: {int(mask.sum())}")


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv)
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be greater than 0")
        csv_paths = iter_csv_paths(args.inputs, args.recursive)
        plt = import_pyplot()

        model_bundle = None
        if not args.no_model:
            if not args.model.is_file():
                raise FileNotFoundError(f"missing model: {args.model}")
            metadata = load_metadata(args.seq_folder)
            mean, scale, feature_columns = load_scaler(args.seq_folder)
            model = import_model(args.model, args.keras_backend)
            model_bundle = (model, metadata, mean, scale, feature_columns)

        for csv_path in csv_paths:
            print(f"Processing {csv_path.name}...")
            try:
                plot_csv(
                    plt=plt,
                    csv_path=csv_path,
                    ready_folder=args.ready_folder,
                    model_bundle=model_bundle,
                    batch_size=args.batch_size,
                    show_baselines=not args.no_baselines,
                )
            except Exception as exc:
                print(f"  Error processing {csv_path.name}: {exc}", file=sys.stderr)

        plt.show()
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
