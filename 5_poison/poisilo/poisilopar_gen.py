from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# Tunable generation settings
# =========================

ROUTE = "ilo"
AUTH_OUTPUT_PREFIX = f"poisauth_{ROUTE}_"
OUTPUT_PER_ATTACK_TYPE: int | None = None
RANDOM_SEED = 42
MAX_SOURCE_ATTEMPTS_PER_TYPE: int | None = None

ATTACK_THRESHOLD = 0.50
TARGET_SCORE_MIN = 0.00  # Inactive by default; probabilities are already >= 0.
TARGET_SCORE_MAX = 1.00  # Inactive by default; probabilities are already <= 1.
TARGET_MARGIN_MAX = 0.10
TARGET_AUTH_GAP_MIN = 0.00
TARGET_AUTH_GAP_MAX = 1.00
PREDICTION_RULE = "argmax4"

INTERPOLATION_START = 0.60
INTERPOLATION_STOP = 1.00
INTERPOLATION_STEP = 0.02
ACCEPT_BLEND_MIN = 0.60
POISALT_ACCEPT_BLEND_MIN: float | None = None
POISSPD_ACCEPT_BLEND_MIN: float | None = None
POISPOS_ACCEPT_BLEND_MIN: float | None = 0.80

RECOMPUTE_DERIVED_FEATURES = False

USE_BGD = True
BGD_STEPS_PER_INTERPOLATION = 1
BGD_LEARNING_RATE = 0.005
BGD_BLEND_MIN = 0.0
BGD_BLEND_MAX = 1.00
BGD_GRADIENT_PERCENTILE = 95.0

BATCH_SIZE = 128
KERAS_BACKEND = "tensorflow"
OVERWRITE_OUTPUTS = False
PROGRESS_EVERY_ATTEMPTS = 1
CANDIDATE_PROGRESS_EVERY = 10
SOURCE_SPLIT = "test"
SPLIT_MANIFEST_NAME = "split_manifest.csv"

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
MODEL_PATH = SCRIPT_DIR / "poisilopar.keras"
SEQ_FOLDER = ROOT_DIR / "dataset" / "data_ilo_seq"
READY_FOLDER = ROOT_DIR / "dataset" / "data_ilo_ready"
OUTPUT_FOLDER = ROOT_DIR / "dataset" / "poison" / "ilo" / "par"

EARTH_RADIUS_M = 6_371_008.8

READY_COLUMNS = [
    "sequence_index",
    "t_elapsed_sec",
    "dt",
    "t_norm",
    "latitude",
    "longitude",
    "altitude_meters",
    "speed_kmh",
    "verticalSpeed_ms",
    "heading",
    "x_wrt0",
    "y_wrt0",
    "delta_altitude",
    "delta_speed",
    "delta_heading",
    "turn_rate",
    "acceleration",
    "distance_per_timestep",
    "route_progress",
    "label",
]

LABEL_NAMES = {
    0: "authentic",
    1: "modified_altitude",
    2: "modified_speed",
    3: "modified_position",
}

@dataclass(frozen=True)
class AttackSpec:
    name: str
    label: int
    mod_folder: str
    mod_prefix: str
    output_prefix: str
    blend_columns: tuple[str, ...]
    recompute_xy_from_latlon: bool


@dataclass(frozen=True)
class PredictionResult:
    predicted_label: int
    target_score: float
    auth_score: float
    target_auth_gap: float
    second_best_score: float
    winning_margin: float
    best_attack_score: float
    authentic_score_mean: float
    class_score_max: tuple[float, float, float, float]


ATTACK_SPECS = (
    AttackSpec(
        name="poisalt",
        label=1,
        mod_folder="iloalt",
        mod_prefix="modalt_ilo_",
        output_prefix="poisalt_ilo_",
        blend_columns=("altitude_meters",),
        recompute_xy_from_latlon=True,
    ),
    AttackSpec(
        name="poisspd",
        label=2,
        mod_folder="ilospd",
        mod_prefix="modspd_ilo_",
        output_prefix="poisspd_ilo_",
        blend_columns=("speed_kmh",),
        recompute_xy_from_latlon=True,
    ),
    AttackSpec(
        name="poispos",
        label=3,
        mod_folder="ilopos",
        mod_prefix="modpos_ilo_",
        output_prefix="poispos_ilo_",
        blend_columns=("latitude", "longitude", "x_wrt0", "y_wrt0"),
        recompute_xy_from_latlon=False,
    ),
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate ILO poison CSVs with poisilopar.keras from dataset/data_ilo_ready "
            "without modifying the clean source dataset."
        )
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--seq-folder", type=Path, default=SEQ_FOLDER)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="CSV manifest with split and flight_id columns. Default: <seq-folder>/split_manifest.csv.",
    )
    parser.add_argument(
        "--source-split",
        type=str,
        default=SOURCE_SPLIT,
        help='Source split to use from split_manifest.csv. Use "all" to disable split filtering. Default: test.',
    )
    parser.add_argument(
        "--flight-id",
        help="Only use this flight ID from the ready folder.",
    )
    parser.add_argument(
        "--attack-type",
        choices=("all", "poisalt", "poisspd", "poispos", "modalt", "modspd", "modpos", "alt", "spd", "pos"),
        default="all",
        help="Only generate this poison attack type. Default: all.",
    )
    parser.add_argument("--ready-folder", type=Path, default=READY_FOLDER)
    parser.add_argument("--output-folder", type=Path, default=OUTPUT_FOLDER)
    parser.add_argument(
        "--count-per-type",
        type=int,
        default=OUTPUT_PER_ATTACK_TYPE,
        help="Maximum poison CSVs to write per attack type. Default: all paired source flights.",
    )
    parser.add_argument(
        "--max-source-attempts-per-type",
        type=int,
        default=MAX_SOURCE_ATTEMPTS_PER_TYPE,
        help="Maximum source flights to try per attack type. Default: all paired source flights.",
    )
    parser.add_argument(
        "--target-score-min",
        type=float,
        default=TARGET_SCORE_MIN,
        help="Optional target-score floor. Default: 0.0, so no practical floor.",
    )
    parser.add_argument(
        "--target-score-max",
        type=float,
        default=TARGET_SCORE_MAX,
        help="Optional target-score cap. Default: 1.0, so no practical cap.",
    )
    parser.add_argument(
        "--target-margin-max",
        type=float,
        default=TARGET_MARGIN_MAX,
        help=(
            "Deprecated compatibility option. The active rule uses --target-auth-gap-max. "
            "Maximum argmax4 winning margin over the runner-up class. "
            f"Default: {TARGET_MARGIN_MAX}."
        ),
    )
    parser.add_argument(
        "--target-auth-gap-max",
        type=float,
        default=TARGET_AUTH_GAP_MAX,
        help=(
            "Maximum allowed target score minus authentic score. "
            f"Default: {TARGET_AUTH_GAP_MAX}."
        ),
    )
    parser.add_argument(
        "--target-auth-gap-min",
        type=float,
        default=TARGET_AUTH_GAP_MIN,
        help=(
            "Minimum required target score minus authentic score. "
            f"Default: {TARGET_AUTH_GAP_MIN}."
        ),
    )
    parser.add_argument(
        "--attack-threshold",
        type=float,
        default=ATTACK_THRESHOLD,
        help=(
            "Deprecated compatibility option. Poison generation now uses argmax4 "
            "to match 6_evaluate/predict_unseen_balanced.py."
        ),
    )
    parser.add_argument("--interpolation-start", type=float, default=INTERPOLATION_START)
    parser.add_argument("--interpolation-stop", type=float, default=INTERPOLATION_STOP)
    parser.add_argument("--interpolation-step", type=float, default=INTERPOLATION_STEP)
    parser.add_argument(
        "--accept-blend-min",
        type=float,
        default=ACCEPT_BLEND_MIN,
        help=(
            "Minimum mean auth-to-mod blend over attacked rows before a candidate can be accepted. "
            f"Raise this for more obvious, more transferable attacks. Default: {ACCEPT_BLEND_MIN}."
        ),
    )
    parser.add_argument(
        "--poisalt-accept-blend-min",
        type=float,
        default=POISALT_ACCEPT_BLEND_MIN,
        help="Override --accept-blend-min for poisalt only.",
    )
    parser.add_argument(
        "--poisspd-accept-blend-min",
        type=float,
        default=POISSPD_ACCEPT_BLEND_MIN,
        help="Override --accept-blend-min for poisspd only.",
    )
    parser.add_argument(
        "--poispos-accept-blend-min",
        type=float,
        default=POISPOS_ACCEPT_BLEND_MIN,
        help="Override --accept-blend-min for poispos only.",
    )
    parser.add_argument(
        "--recompute-derived",
        action="store_true",
        default=RECOMPUTE_DERIVED_FEATURES,
        help=(
            "Recompute derived physical features after blending. Default: off, "
            "so poison CSVs mimic the original mod_alt/mod_spd/mod_pos attack files."
        ),
    )
    parser.add_argument("--bgd-steps", type=int, default=BGD_STEPS_PER_INTERPOLATION)
    parser.add_argument("--bgd-learning-rate", type=float, default=BGD_LEARNING_RATE)
    parser.add_argument("--bgd-blend-max", type=float, default=BGD_BLEND_MAX)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--keras-backend", choices=("tensorflow", "jax", "torch"), default=KERAS_BACKEND)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=PROGRESS_EVERY_ATTEMPTS,
        help="Print one source-flight progress line every N attempts. Default: 1.",
    )
    parser.add_argument(
        "--candidate-progress-every",
        type=int,
        default=CANDIDATE_PROGRESS_EVERY,
        help="Print in-flight search progress every N interpolation levels. Default: 10.",
    )
    parser.add_argument("--no-bgd", action="store_true", help="Use interpolation only.")
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE_OUTPUTS)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.count_per_type is not None and args.count_per_type <= 0:
        raise ValueError("--count-per-type must be greater than 0")
    if args.max_source_attempts_per_type is not None and args.max_source_attempts_per_type <= 0:
        raise ValueError("--max-source-attempts-per-type must be greater than 0")
    if not 0 <= args.attack_threshold <= 1:
        raise ValueError("--attack-threshold must be between 0 and 1")
    if not 0 <= args.target_score_min <= args.target_score_max <= 1:
        raise ValueError("--target-score-min/max must satisfy 0 <= min <= max <= 1")
    if args.target_margin_max < 0:
        raise ValueError("--target-margin-max must be >= 0")
    if args.target_auth_gap_min < 0 or args.target_auth_gap_max < args.target_auth_gap_min:
        raise ValueError("--target-auth-gap-min/max must satisfy 0 <= min <= max")
    if args.interpolation_step <= 0:
        raise ValueError("--interpolation-step must be greater than 0")
    if args.interpolation_start < 0 or args.interpolation_stop <= args.interpolation_start:
        raise ValueError("--interpolation range must satisfy 0 <= start < stop")
    if args.accept_blend_min < 0:
        raise ValueError("--accept-blend-min must be >= 0")
    for option_name in (
        "poisalt_accept_blend_min",
        "poisspd_accept_blend_min",
        "poispos_accept_blend_min",
    ):
        value = getattr(args, option_name)
        if value is not None and value < 0:
            raise ValueError(f"--{option_name.replace('_', '-')} must be >= 0")
    if args.bgd_steps < 0:
        raise ValueError("--bgd-steps must be >= 0")
    if args.bgd_learning_rate < 0:
        raise ValueError("--bgd-learning-rate must be >= 0")
    if args.bgd_blend_max <= 0:
        raise ValueError("--bgd-blend-max must be greater than 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")
    if not args.source_split.strip():
        raise ValueError("--source-split must not be empty")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be greater than 0")
    if args.candidate_progress_every <= 0:
        raise ValueError("--candidate-progress-every must be greater than 0")
    if not args.model.is_file():
        raise FileNotFoundError(f"missing model: {args.model}")
    if not args.seq_folder.is_dir():
        raise FileNotFoundError(f"missing sequence folder: {args.seq_folder}")
    if not args.ready_folder.is_dir():
        raise FileNotFoundError(f"missing ready folder: {args.ready_folder}")


def import_model(model_path: Path, backend: str):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "matplotlib"))
    os.environ["KERAS_BACKEND"] = backend
    if backend != "tensorflow":
        raise ValueError("BGD generation currently requires --keras-backend tensorflow")

    import tensorflow as tf
    import keras

    return keras.models.load_model(model_path, compile=False), tf


def load_metadata(seq_folder: Path) -> dict:
    metadata_path = seq_folder / "metadata.json"
    with metadata_path.open() as metadata_file:
        return json.load(metadata_file)


def load_scaler(seq_folder: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    scaler_path = seq_folder / "scaler.npz"
    scaler = np.load(scaler_path)
    mean = scaler["mean"].astype(np.float32)
    scale = scaler["scale"].astype(np.float32)
    scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
    feature_columns = [str(value) for value in scaler["feature_columns"].tolist()]
    return mean, scale, feature_columns


def load_split_flight_ids(
    seq_folder: Path,
    split_manifest: Path | None,
    source_split: str,
) -> set[str] | None:
    split_name = source_split.strip().lower()
    if split_name == "all":
        return None

    manifest_path = split_manifest if split_manifest is not None else seq_folder / SPLIT_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing split manifest: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    required_columns = {"split", "flight_id"}
    missing = sorted(required_columns - set(manifest.columns))
    if missing:
        raise ValueError(f"{manifest_path} is missing columns: {missing}")

    split_values = manifest["split"].astype(str).str.lower()
    flight_ids = set(manifest.loc[split_values.eq(split_name), "flight_id"].astype(str))
    if not flight_ids:
        available = sorted(manifest["split"].astype(str).unique())
        raise ValueError(
            f"no flight IDs found for split {source_split!r} in {manifest_path}; "
            f"available splits: {available}"
        )
    return flight_ids


def collect_ready_files(folder: Path, prefix: str) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for csv_path in sorted(folder.glob("*.csv")):
        if not csv_path.name.startswith(prefix):
            continue
        flight_id = csv_path.stem[len(prefix):]
        if flight_id:
            files[flight_id] = csv_path
    return files


def limited_target_count(requested_count: int | None, available_count: int) -> int:
    return available_count if requested_count is None else min(requested_count, available_count)


def limited_attempt_count(requested_count: int | None, available_count: int) -> int:
    return available_count if requested_count is None else min(requested_count, available_count)


def count_text(requested_count: int | None, available_count: int) -> str:
    if requested_count is None:
        return f"all {available_count}"
    return str(min(requested_count, available_count))


def read_ready_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [column for column in READY_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {missing}")
    df = df[READY_COLUMNS].copy()
    for column in READY_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[READY_COLUMNS].isna().any().any():
        bad_columns = df.columns[df.isna().any()].tolist()
        raise ValueError(f"{csv_path} has missing/non-numeric values in {bad_columns}")
    return df


def haversine_m(latitude: pd.Series, longitude: pd.Series) -> pd.Series:
    lat1 = np.radians(latitude.shift())
    lon1 = np.radians(longitude.shift())
    lat2 = np.radians(latitude)
    lon2 = np.radians(longitude)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    distance = 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))
    return pd.Series(distance, index=latitude.index).fillna(0.0)


def relative_xy_m(latitude: pd.Series, longitude: pd.Series) -> tuple[pd.Series, pd.Series]:
    if latitude.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    start_lat = np.radians(float(latitude.iloc[0]))
    start_lon = np.radians(float(longitude.iloc[0]))
    lat = np.radians(latitude)
    lon = np.radians(longitude)
    x = (lon - start_lon) * np.cos((lat + start_lat) / 2) * EARTH_RADIUS_M
    y = (lat - start_lat) * EARTH_RADIUS_M
    return pd.Series(x, index=latitude.index), pd.Series(y, index=latitude.index)


def recompute_derived_columns(df: pd.DataFrame, recompute_xy_from_latlon: bool) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_numeric(df["dt"], errors="coerce").fillna(0.0)
    dt_safe = dt.mask(dt == 0, np.nan)
    altitude = pd.to_numeric(df["altitude_meters"], errors="coerce")
    speed = pd.to_numeric(df["speed_kmh"], errors="coerce")
    heading = pd.to_numeric(df["heading"], errors="coerce")
    latitude = pd.to_numeric(df["latitude"], errors="coerce")
    longitude = pd.to_numeric(df["longitude"], errors="coerce")

    df["altitude_meters"] = altitude.clip(lower=0.0)
    df["speed_kmh"] = speed.clip(lower=0.0)
    altitude = df["altitude_meters"]
    speed = df["speed_kmh"]

    df["delta_altitude"] = altitude.diff().fillna(0.0)
    df["verticalSpeed_ms"] = (df["delta_altitude"] / dt_safe).fillna(0.0)
    df["delta_speed"] = speed.diff().fillna(0.0)
    df["delta_heading"] = ((heading.diff() + 180) % 360 - 180).fillna(0.0)
    df["turn_rate"] = (df["delta_heading"] / dt_safe).fillna(0.0)
    df["acceleration"] = (df["delta_speed"] / dt_safe).fillna(0.0)
    df["distance_per_timestep"] = haversine_m(latitude, longitude)

    cumulative_distance = df["distance_per_timestep"].cumsum()
    total_distance = float(cumulative_distance.iloc[-1]) if len(cumulative_distance) else 0.0
    df["route_progress"] = cumulative_distance / total_distance if total_distance > 0 else 0.0

    if recompute_xy_from_latlon:
        df["x_wrt0"], df["y_wrt0"] = relative_xy_m(latitude, longitude)

    return df[READY_COLUMNS]


def make_windows_with_spans(
    df: pd.DataFrame,
    feature_columns: list[str],
    mean: np.ndarray,
    scale: np.ndarray,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = df[feature_columns].to_numpy(dtype=np.float32)
    values = ((values - mean) / scale).astype(np.float32)

    windows = []
    spans = []
    for start in range(0, len(df) - window_size + 1, stride):
        end = start + window_size
        windows.append(values[start:end])
        spans.append((start, end))

    if not windows:
        return (
            np.empty((0, window_size, len(feature_columns)), dtype=np.float32),
            np.empty((0, 2), dtype=np.int64),
        )
    return np.stack(windows), np.array(spans, dtype=np.int64)


def score_candidate(
    model,
    df: pd.DataFrame,
    target_label: int,
    feature_columns: list[str],
    mean: np.ndarray,
    scale: np.ndarray,
    window_size: int,
    stride: int,
    batch_size: int,
    attack_threshold: float,
) -> PredictionResult:
    _ = attack_threshold  # Kept for CLI compatibility; argmax4 ignores thresholds.
    windows, _ = make_windows_with_spans(df, feature_columns, mean, scale, window_size, stride)
    if len(windows) == 0:
        raise ValueError(f"candidate has fewer than {window_size} rows")

    probabilities = model.predict(windows, batch_size=batch_size, verbose=0)
    class_score_max = tuple(float(probabilities[:, index].max()) for index in range(4))
    max_non_auth_by_class = probabilities[:, 1:].max(axis=0)
    best_attack_offset = int(np.argmax(max_non_auth_by_class))
    best_attack_score = float(max_non_auth_by_class[best_attack_offset])
    predicted_label = int(np.argmax(class_score_max))
    predicted_score = class_score_max[predicted_label]
    auth_score = class_score_max[0]
    second_best_score = max(
        score for index, score in enumerate(class_score_max) if index != predicted_label
    )
    winning_margin = predicted_score - second_best_score
    target_auth_gap = class_score_max[target_label] - auth_score
    return PredictionResult(
        predicted_label=predicted_label,
        target_score=class_score_max[target_label],
        auth_score=float(auth_score),
        target_auth_gap=float(target_auth_gap),
        second_best_score=float(second_best_score),
        winning_margin=float(winning_margin),
        best_attack_score=best_attack_score,
        authentic_score_mean=float(probabilities[:, 0].mean()),
        class_score_max=class_score_max,
    )


def accepted_prediction(
    result: PredictionResult,
    target_label: int,
    min_score: float,
    max_score: float,
    min_auth_gap: float,
    max_auth_gap: float,
    mean_attack_blend: float,
    min_attack_blend: float,
) -> bool:
    return (
        result.predicted_label == target_label
        and min_score <= result.target_score <= max_score
        and min_auth_gap <= result.target_auth_gap <= max_auth_gap
        and mean_attack_blend >= min_attack_blend
    )


def mean_attack_blend(blend: np.ndarray, attack_mask: np.ndarray) -> float:
    active_blend = blend[attack_mask]
    if len(active_blend) == 0:
        return 0.0
    return float(np.mean(active_blend))


def accept_blend_min_for_spec(args: argparse.Namespace, spec: AttackSpec) -> float:
    override = getattr(args, f"{spec.name}_accept_blend_min")
    return float(args.accept_blend_min if override is None else override)


def attack_mask_from_mod(mod_df: pd.DataFrame, auth_df: pd.DataFrame, spec: AttackSpec) -> np.ndarray:
    mask = mod_df["label"].astype(int).eq(spec.label).to_numpy()
    if mask.any():
        return mask

    diff = np.zeros(len(mod_df), dtype=bool)
    for column in spec.blend_columns:
        diff |= np.abs(mod_df[column].to_numpy(dtype=float) - auth_df[column].to_numpy(dtype=float)) > 1e-9
    return diff


def build_candidate(
    auth_df: pd.DataFrame,
    mod_df: pd.DataFrame,
    spec: AttackSpec,
    attack_mask: np.ndarray,
    blend: np.ndarray,
    blend_max: float,
    recompute_derived: bool,
) -> pd.DataFrame:
    candidate = auth_df[READY_COLUMNS].copy()
    clipped_blend = np.clip(blend.astype(float), BGD_BLEND_MIN, blend_max)
    clipped_blend = np.where(attack_mask, clipped_blend, 0.0)

    for column in spec.blend_columns:
        auth_values = auth_df[column].to_numpy(dtype=float)
        mod_values = mod_df[column].to_numpy(dtype=float)
        candidate[column] = auth_values + clipped_blend * (mod_values - auth_values)

    if recompute_derived:
        candidate = recompute_derived_columns(
            candidate,
            recompute_xy_from_latlon=spec.recompute_xy_from_latlon,
        )
    candidate["label"] = 0
    candidate.loc[attack_mask, "label"] = spec.label
    return candidate[READY_COLUMNS]


def blend_gradient_step(
    model,
    tf,
    candidate: pd.DataFrame,
    auth_df: pd.DataFrame,
    mod_df: pd.DataFrame,
    spec: AttackSpec,
    attack_mask: np.ndarray,
    blend: np.ndarray,
    feature_columns: list[str],
    mean: np.ndarray,
    scale: np.ndarray,
    window_size: int,
    stride: int,
    learning_rate: float,
    blend_max: float,
) -> np.ndarray:
    windows, spans = make_windows_with_spans(candidate, feature_columns, mean, scale, window_size, stride)
    if len(windows) == 0:
        return blend

    x_var = tf.Variable(windows)
    with tf.GradientTape() as tape:
        probabilities = model(x_var, training=False)
        target_scores = probabilities[:, spec.label]
        objective = tf.reduce_max(target_scores)

    gradients = tape.gradient(objective, x_var)
    if gradients is None:
        return blend

    gradients_np = gradients.numpy()
    feature_index = {column: index for index, column in enumerate(feature_columns)}
    row_gradient = np.zeros(len(candidate), dtype=float)
    row_counts = np.zeros(len(candidate), dtype=float)

    for column in spec.blend_columns:
        if column not in feature_index:
            continue
        feature_idx = feature_index[column]
        column_delta = mod_df[column].to_numpy(dtype=float) - auth_df[column].to_numpy(dtype=float)
        feature_scale = float(scale[feature_idx]) if float(scale[feature_idx]) != 0 else 1.0
        for window_idx, (start, end) in enumerate(spans):
            row_gradient[start:end] += (
                gradients_np[window_idx, :, feature_idx]
                * column_delta[start:end]
                / feature_scale
            )
            row_counts[start:end] += 1.0

    row_counts = np.where(row_counts == 0, 1.0, row_counts)
    row_gradient = row_gradient / row_counts
    row_gradient = np.where(attack_mask, row_gradient, 0.0)
    active = np.abs(row_gradient[attack_mask])
    if len(active) == 0 or float(active.max()) == 0.0:
        return blend

    normalizer = float(np.percentile(active, BGD_GRADIENT_PERCENTILE))
    if not np.isfinite(normalizer) or normalizer <= 0:
        normalizer = float(active.max())
    if normalizer <= 0:
        return blend

    next_blend = blend + learning_rate * (row_gradient / normalizer)
    next_blend = np.clip(next_blend, BGD_BLEND_MIN, blend_max)
    next_blend = np.where(attack_mask, next_blend, 0.0)
    return next_blend


def interpolation_values(start: float, stop: float, step: float) -> np.ndarray:
    values = np.arange(start, stop + step * 0.5, step, dtype=float)
    return np.clip(values, 0.0, stop)


def log_progress(message: str) -> None:
    print(message, flush=True)


def generate_one_poison(
    model,
    tf,
    auth_path: Path,
    mod_path: Path,
    spec: AttackSpec,
    args: argparse.Namespace,
    feature_columns: list[str],
    mean: np.ndarray,
    scale: np.ndarray,
    window_size: int,
    stride: int,
    progress_prefix: str,
) -> tuple[pd.DataFrame | None, PredictionResult | None, str]:
    auth_df = read_ready_csv(auth_path)
    mod_df = read_ready_csv(mod_path)
    if len(auth_df) != len(mod_df):
        return None, None, f"row count mismatch auth={len(auth_df)} mod={len(mod_df)}"

    attack_mask = attack_mask_from_mod(mod_df, auth_df, spec)
    if not attack_mask.any():
        return None, None, "no attack rows found"

    accept_blend_min = accept_blend_min_for_spec(args, spec)
    best_score = 0.0
    blend_values = interpolation_values(
        args.interpolation_start,
        args.interpolation_stop,
        args.interpolation_step,
    )
    total_blend_values = len(blend_values)
    for blend_index, blend_value in enumerate(blend_values, start=1):
        blend = np.zeros(len(auth_df), dtype=float)
        blend[attack_mask] = blend_value
        candidate = build_candidate(
            auth_df,
            mod_df,
            spec,
            attack_mask,
            blend,
            args.bgd_blend_max,
            args.recompute_derived,
        )
        result = score_candidate(
            model=model,
            df=candidate,
            target_label=spec.label,
            feature_columns=feature_columns,
            mean=mean,
            scale=scale,
            window_size=window_size,
            stride=stride,
            batch_size=args.batch_size,
            attack_threshold=args.attack_threshold,
        )
        best_score = max(best_score, result.target_score)
        effective_blend = mean_attack_blend(blend, attack_mask)
        if blend_index == 1 or blend_index % args.candidate_progress_every == 0:
            log_progress(
                f"{progress_prefix} search {blend_index}/{total_blend_values} "
                f"blend={blend_value:.3f} mean_blend={effective_blend:.3f} "
                f"target={result.target_score:.4f} "
                f"auth={result.auth_score:.4f} "
                f"auth_gap={result.target_auth_gap:.4f} "
                f"second={result.second_best_score:.4f} "
                f"margin={result.winning_margin:.4f} "
                f"best={best_score:.4f} pred={LABEL_NAMES.get(result.predicted_label, result.predicted_label)}"
            )
        if accepted_prediction(
            result,
            spec.label,
            args.target_score_min,
            args.target_score_max,
            args.target_auth_gap_min,
            args.target_auth_gap_max,
            effective_blend,
            accept_blend_min,
        ):
            return candidate, result, f"accepted interpolation={blend_value:.3f} mean_blend={effective_blend:.3f}"
        if result.predicted_label == spec.label and result.target_score > args.target_score_max:
            return None, result, f"overshot target score {result.target_score:.4f}"
        if args.no_bgd or not USE_BGD:
            continue

        for step_index in range(args.bgd_steps):
            blend = blend_gradient_step(
                model=model,
                tf=tf,
                candidate=candidate,
                auth_df=auth_df,
                mod_df=mod_df,
                spec=spec,
                attack_mask=attack_mask,
                blend=blend,
                feature_columns=feature_columns,
                mean=mean,
                scale=scale,
                window_size=window_size,
                stride=stride,
                learning_rate=args.bgd_learning_rate,
                blend_max=args.bgd_blend_max,
            )
            candidate = build_candidate(
                auth_df,
                mod_df,
                spec,
                attack_mask,
                blend,
                args.bgd_blend_max,
                args.recompute_derived,
            )
            result = score_candidate(
                model=model,
                df=candidate,
                target_label=spec.label,
                feature_columns=feature_columns,
                mean=mean,
                scale=scale,
                window_size=window_size,
                stride=stride,
                batch_size=args.batch_size,
                attack_threshold=args.attack_threshold,
            )
            best_score = max(best_score, result.target_score)
            effective_blend = mean_attack_blend(blend, attack_mask)
            if accepted_prediction(
                result,
                spec.label,
                args.target_score_min,
                args.target_score_max,
                args.target_auth_gap_min,
                args.target_auth_gap_max,
                effective_blend,
                accept_blend_min,
            ):
                return (
                    candidate,
                    result,
                    f"accepted interpolation={blend_value:.3f} bgd_step={step_index + 1} "
                    f"mean_blend={effective_blend:.3f}",
                )
            if result.predicted_label == spec.label and result.target_score > args.target_score_max:
                break

    return None, None, f"no barely-threshold candidate found; best target score={best_score:.4f}"


def prepare_output_folder(
    output_folder: Path,
    overwrite: bool,
    flight_id: str | None = None,
) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)
    pattern = f"*_{flight_id}.csv" if flight_id else "*.csv"
    existing = sorted(output_folder.glob(pattern))
    if existing and not overwrite:
        preview = ", ".join(path.name for path in existing[:5])
        suffix = "..." if len(existing) > 5 else ""
        raise FileExistsError(
            f"{output_folder} already has CSV files: {preview}{suffix}; rerun with --overwrite"
        )
    if overwrite:
        for path in existing:
            path.unlink()


def copy_auth_files(args: argparse.Namespace, allowed_flight_ids: set[str] | None) -> int:
    auth_files = collect_ready_files(args.ready_folder / "iloauth", "auth_ilo_")
    flight_ids = set(auth_files)
    if allowed_flight_ids is not None:
        flight_ids &= allowed_flight_ids
    ordered_flight_ids = sorted(flight_ids)
    if not ordered_flight_ids:
        split_text = args.source_split if allowed_flight_ids is not None else "all"
        raise ValueError(f"no auth files found for source split {split_text}")

    for flight_id in ordered_flight_ids:
        output_path = args.output_folder / f"{AUTH_OUTPUT_PREFIX}{flight_id}.csv"
        shutil.copyfile(auth_files[flight_id], output_path)

    log_progress(
        f"\n=== poisauth ({LABEL_NAMES[0]}) ===\n"
        f"Copied auth source flights: {len(ordered_flight_ids)} | source split: {args.source_split}"
    )
    return len(ordered_flight_ids)


def generate_for_attack_type(
    model,
    tf,
    spec: AttackSpec,
    args: argparse.Namespace,
    rng: np.random.Generator,
    feature_columns: list[str],
    mean: np.ndarray,
    scale: np.ndarray,
    window_size: int,
    stride: int,
    allowed_flight_ids: set[str] | None,
) -> tuple[int, int]:
    auth_files = collect_ready_files(args.ready_folder / "iloauth", "auth_ilo_")
    mod_files = collect_ready_files(args.ready_folder / spec.mod_folder, spec.mod_prefix)
    paired_flight_ids = set(auth_files) & set(mod_files)
    if allowed_flight_ids is not None:
        paired_flight_ids &= allowed_flight_ids
    flight_ids = np.array(sorted(paired_flight_ids), dtype=object)
    if len(flight_ids) == 0:
        split_text = args.source_split if allowed_flight_ids is not None else "all"
        raise ValueError(f"no paired auth/{spec.name} files found for source split {split_text}")
    rng.shuffle(flight_ids)
    target_count = limited_target_count(args.count_per_type, len(flight_ids))
    attempt_limit = limited_attempt_count(args.max_source_attempts_per_type, len(flight_ids))
    log_progress(
        f"\n=== {spec.name} ({LABEL_NAMES[spec.label]}) ===\n"
        f"Paired source flights: {len(flight_ids)} | source split: {args.source_split} | "
        f"target outputs: {count_text(args.count_per_type, len(flight_ids))} | "
        f"max attempts: {count_text(args.max_source_attempts_per_type, len(flight_ids))}"
    )

    written = 0
    attempts = 0
    for flight_id_obj in flight_ids:
        if written >= target_count or attempts >= attempt_limit:
            break
        flight_id = str(flight_id_obj)
        attempts += 1
        output_path = args.output_folder / f"{spec.output_prefix}{flight_id}.csv"
        if attempts == 1 or attempts % args.progress_every == 0:
            remaining = target_count - written
            log_progress(
                f"TRY {spec.name} attempt={attempts}/{attempt_limit} "
                f"written={written}/{target_count} remaining={remaining} flight_id={flight_id}"
            )
        candidate, result, reason = generate_one_poison(
            model=model,
            tf=tf,
            auth_path=auth_files[flight_id],
            mod_path=mod_files[flight_id],
            spec=spec,
            args=args,
            feature_columns=feature_columns,
            mean=mean,
            scale=scale,
            window_size=window_size,
            stride=stride,
            progress_prefix=f"{spec.name} {flight_id}",
        )
        if candidate is None:
            log_progress(f"SKIP {spec.name} {flight_id}: {reason}")
            continue

        candidate.to_csv(output_path, index=False)
        written += 1
        assert result is not None
        score_text = ", ".join(
            f"{LABEL_NAMES[index]}={score:.4f}"
            for index, score in enumerate(result.class_score_max)
        )
        log_progress(
            f"WROTE {output_path} "
            f"({written}/{target_count} {spec.name}, {reason}, "
            f"auth_gap={result.target_auth_gap:.4f}, margin={result.winning_margin:.4f}, "
            f"{score_text})"
        )

    log_progress(f"{spec.name}: wrote {written}/{target_count} after {attempts} attempts")
    return written, target_count


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv)
        validate_args(args)

        metadata = load_metadata(args.seq_folder)
        mean, scale, feature_columns = load_scaler(args.seq_folder)
        allowed_flight_ids = load_split_flight_ids(
            seq_folder=args.seq_folder,
            split_manifest=args.split_manifest,
            source_split=args.source_split,
        )
        if args.flight_id:
            requested_flight_ids = {args.flight_id}
            allowed_flight_ids = (
                requested_flight_ids
                if allowed_flight_ids is None
                else allowed_flight_ids & requested_flight_ids
            )
        window_size = int(metadata["window_size"])
        stride = int(metadata["stride"])
        if tuple(feature_columns) != tuple(metadata["feature_columns"]):
            raise ValueError("scaler feature columns do not match metadata feature columns")

        model, tf = import_model(args.model, args.keras_backend)
        prepare_output_folder(args.output_folder, args.overwrite, args.flight_id)
        rng = np.random.default_rng(args.seed)
        log_progress(
            "Poison generation starting\n"
            f"Model: {args.model}\n"
            f"Ready folder: {args.ready_folder}\n"
            f"Source split: {args.source_split} "
            f"({len(allowed_flight_ids) if allowed_flight_ids is not None else 'all'} allowed flight IDs)\n"
            f"Output folder: {args.output_folder}\n"
            f"Output target: {'all paired source flights' if args.count_per_type is None else str(args.count_per_type)} per attack type\n"
            f"Prediction rule: {PREDICTION_RULE} (target class must be max over all 4 classes)\n"
            f"Acceptance rule: target_score >= {args.target_score_min:.3f} (optional floor), "
            f"target_score <= {args.target_score_max:.3f}, "
            f"{args.target_auth_gap_min:.3f} <= target_score - auth_score <= "
            f"{args.target_auth_gap_max:.3f}, "
            f"mean_blend >= {args.accept_blend_min:.3f} "
            f"(poisalt={accept_blend_min_for_spec(args, ATTACK_SPECS[0]):.3f}, "
            f"poisspd={accept_blend_min_for_spec(args, ATTACK_SPECS[1]):.3f}, "
            f"poispos={accept_blend_min_for_spec(args, ATTACK_SPECS[2]):.3f})\n"
            f"Derived features: {'recomputed' if args.recompute_derived else 'kept from auth, matching original attack scripts'}\n"
            f"Interpolation: {args.interpolation_start:.3f}-{args.interpolation_stop:.3f} "
            f"step {args.interpolation_step:.3f}\n"
            f"BGD: {'off' if args.no_bgd or not USE_BGD else 'on'} "
            f"steps={args.bgd_steps} lr={args.bgd_learning_rate} blend_max={args.bgd_blend_max}"
        )

        attack_type_aliases = {
            "all": None,
            "poisalt": "poisalt",
            "poisspd": "poisspd",
            "poispos": "poispos",
            "modalt": "poisalt",
            "modspd": "poisspd",
            "modpos": "poispos",
            "alt": "poisalt",
            "spd": "poisspd",
            "pos": "poispos",
        }
        selected_attack_type = attack_type_aliases[args.attack_type]
        selected_attack_specs = (
            ATTACK_SPECS
            if selected_attack_type is None
            else tuple(spec for spec in ATTACK_SPECS if spec.name == selected_attack_type)
        )
        if not selected_attack_specs:
            raise ValueError(f"unknown poison attack type: {args.attack_type}")

        auth_written = copy_auth_files(args, allowed_flight_ids)
        poison_written_total = 0
        failures = []
        for spec in selected_attack_specs:
            written, target_count = generate_for_attack_type(
                model=model,
                tf=tf,
                spec=spec,
                args=args,
                rng=rng,
                feature_columns=feature_columns,
                mean=mean,
                scale=scale,
                window_size=window_size,
                stride=stride,
                allowed_flight_ids=allowed_flight_ids,
            )
            poison_written_total += written
            if written != target_count:
                failures.append(f"{spec.name}: {written}/{target_count}")

        if failures:
            print("ERROR: not enough poison CSVs generated: " + "; ".join(failures), file=sys.stderr)
            return 1

        print(
            f"Done. Wrote {auth_written} auth CSVs and {poison_written_total} poison CSVs "
            f"to {args.output_folder}"
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
