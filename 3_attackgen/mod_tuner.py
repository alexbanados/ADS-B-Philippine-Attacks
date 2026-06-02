from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_paths import resolve_dataset_path


ROUTE_PROGRESS_COLUMN = "route_progress"

# =========================
# Adjustable attack settings
# =========================

ADJUSTABLE_BIN_COUNT = 2000 #How many bins to divide the route into for attack placement. 
                            #More bins = finer control but more chance of missing stats for some bins. 
                            #Should be >= max attack_start_bin + attack_duration.
ADJUSTABLE_SAMPLE_SIZE = 1272 #how many files to process
ADJUSTABLE_RANDOM_SEED = 42

ADJUSTABLE_K = 3 #intensity of attack
ADJUSTABLE_ALPHA = 0.3 #Controls randomness around the sampled offset. 
                       #Bigger = more variable, sometimes less smooth/less predictable. 
                       #Smaller = cleaner and more consistent.

ADJUSTABLE_DIRECTION = None  # None=random per CSV, or use -1 / +1.

ADJUSTABLE_ATTACK_START_BIN = None  # None=random per CSV.
ADJUSTABLE_ATTACK_START_BIN_MIN = 0
ADJUSTABLE_ATTACK_START_BIN_MAX = 1900

ADJUSTABLE_ATTACK_DURATION = None  # None=random per CSV.
ADJUSTABLE_ATTACK_DURATION_MIN = 100
ADJUSTABLE_ATTACK_DURATION_MAX = 500

ADJUSTABLE_ENVELOPE_TYPE = "random"
ADJUSTABLE_RANDOM_ENVELOPE_TYPES = (
    "random_spline",
    "hann",
    "asymmetric_hann",
    "beta",
    "raised_cosine",
)
ADJUSTABLE_SPLINE_CONTROL_POINTS_MIN = 4
ADJUSTABLE_SPLINE_CONTROL_POINTS_MAX = 7
ADJUSTABLE_SPLINE_MIN_CONTROL_VALUE = 0.50
ADJUSTABLE_ASYMMETRIC_HANN_PEAK_MIN = 0.30
ADJUSTABLE_ASYMMETRIC_HANN_PEAK_MAX = 0.70
ADJUSTABLE_BETA_SHAPE_MIN = 2.0
ADJUSTABLE_BETA_SHAPE_MAX = 6.0
ADJUSTABLE_RAISED_COSINE_TAPER_MIN = 0.15
ADJUSTABLE_RAISED_COSINE_TAPER_MAX = 0.35

ADJUSTABLE_DURATION_AMPLITUDE_BETA = 1  #Balances short vs long attacks for Hann. 
                                          #Bigger beta makes short attacks stealthier and long attacks more obvious.


ATTACK_DURATION_REFERENCE = (
    ADJUSTABLE_ATTACK_DURATION_MIN + ADJUSTABLE_ATTACK_DURATION_MAX
) / 2

ENVELOPE_CHOICES = (
    "random",
    "random_spline",
    "hann",
    "asymmetric_hann",
    "beta",
    "raised_cosine",
)


def assign_bin_id(route_progress: pd.Series, bins: int) -> pd.Series:
    bin_id = np.floor(route_progress * bins)
    bin_id = bin_id.clip(0, bins - 1)
    return bin_id.astype("Int64")


def normalize_envelope(envelope: np.ndarray) -> np.ndarray:
    envelope = np.clip(envelope.astype(float), 0.0, None)
    peak = envelope.max() if len(envelope) else 0.0
    if peak > 0:
        envelope = envelope / peak
    if len(envelope) > 0:
        envelope[0] = 0.0
        envelope[-1] = 0.0
    return envelope


def hann_envelope(row_count: int) -> tuple[np.ndarray, str]:
    """Return a bell-shaped attack envelope over the actual attacked rows."""
    if row_count <= 0:
        return np.array([], dtype=float), "window=hann"
    if row_count == 1:
        return np.array([1.0], dtype=float), "window=hann"
    return np.hanning(row_count), "window=hann"


def asymmetric_hann_envelope(
    row_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    """Return a Hann-like pulse with a random asymmetric peak location."""
    if row_count <= 0:
        return np.array([], dtype=float), "peak_fraction=none"
    if row_count == 1:
        return np.array([1.0], dtype=float), "peak_fraction=1.000"
    if row_count == 2:
        return np.array([0.0, 0.0], dtype=float), "peak_fraction=0.500"

    peak_fraction = float(
        rng.uniform(
            ADJUSTABLE_ASYMMETRIC_HANN_PEAK_MIN,
            ADJUSTABLE_ASYMMETRIC_HANN_PEAK_MAX,
        )
    )
    peak_index = int(round(peak_fraction * (row_count - 1)))
    peak_index = int(np.clip(peak_index, 1, row_count - 2))

    envelope = np.zeros(row_count, dtype=float)
    rise_x = np.linspace(0, 1, peak_index + 1)
    fall_x = np.linspace(0, 1, row_count - peak_index)
    envelope[: peak_index + 1] = 0.5 - 0.5 * np.cos(np.pi * rise_x)
    envelope[peak_index:] = 0.5 + 0.5 * np.cos(np.pi * fall_x)

    params = (
        f"peak_fraction={peak_index / (row_count - 1):.3f};"
        f"rise_rows={peak_index + 1};fall_rows={row_count - peak_index}"
    )
    return normalize_envelope(envelope), params


def beta_curve_envelope(
    row_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    """Return a non-symmetric beta-shaped pulse with zero endpoints."""
    if row_count <= 0:
        return np.array([], dtype=float), "alpha=none;beta=none"
    if row_count == 1:
        return np.array([1.0], dtype=float), "alpha=none;beta=none"
    if row_count == 2:
        return np.array([0.0, 0.0], dtype=float), "alpha=none;beta=none"

    alpha_shape = float(
        rng.uniform(ADJUSTABLE_BETA_SHAPE_MIN, ADJUSTABLE_BETA_SHAPE_MAX)
    )
    beta_shape = float(
        rng.uniform(ADJUSTABLE_BETA_SHAPE_MIN, ADJUSTABLE_BETA_SHAPE_MAX)
    )
    if abs(alpha_shape - beta_shape) < 0.60:
        if rng.random() < 0.5:
            alpha_shape = max(ADJUSTABLE_BETA_SHAPE_MIN, beta_shape - 0.80)
        else:
            beta_shape = max(ADJUSTABLE_BETA_SHAPE_MIN, alpha_shape - 0.80)

    x = np.linspace(0, 1, row_count)
    envelope = np.zeros(row_count, dtype=float)
    internal = (x > 0) & (x < 1)
    envelope[internal] = (
        x[internal] ** (alpha_shape - 1)
        * (1 - x[internal]) ** (beta_shape - 1)
    )
    params = f"alpha={alpha_shape:.2f};beta={beta_shape:.2f}"
    return normalize_envelope(envelope), params


def raised_cosine_envelope(
    row_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    """Return a raised-cosine pulse with random cosine taper lengths."""
    if row_count <= 0:
        return np.array([], dtype=float), "rise_fraction=none;fall_fraction=none"
    if row_count == 1:
        return np.array([1.0], dtype=float), "rise_fraction=none;fall_fraction=none"
    if row_count == 2:
        return np.array([0.0, 0.0], dtype=float), "rise_fraction=0.500;fall_fraction=0.500"

    rise_fraction = float(
        rng.uniform(
            ADJUSTABLE_RAISED_COSINE_TAPER_MIN,
            ADJUSTABLE_RAISED_COSINE_TAPER_MAX,
        )
    )
    fall_fraction = float(
        rng.uniform(
            ADJUSTABLE_RAISED_COSINE_TAPER_MIN,
            ADJUSTABLE_RAISED_COSINE_TAPER_MAX,
        )
    )
    total_taper = rise_fraction + fall_fraction
    if total_taper > 0.85:
        scale = 0.85 / total_taper
        rise_fraction *= scale
        fall_fraction *= scale

    x = np.linspace(0, 1, row_count)
    envelope = np.ones(row_count, dtype=float)
    rising = x < rise_fraction
    falling = x > (1 - fall_fraction)
    envelope[rising] = 0.5 - 0.5 * np.cos(np.pi * x[rising] / rise_fraction)
    envelope[falling] = 0.5 + 0.5 * np.cos(
        np.pi * (x[falling] - (1 - fall_fraction)) / fall_fraction
    )

    params = (
        f"rise_fraction={rise_fraction:.3f};"
        f"fall_fraction={fall_fraction:.3f};"
        f"plateau_fraction={max(0.0, 1 - rise_fraction - fall_fraction):.3f}"
    )
    return normalize_envelope(envelope), params


def random_control_point_spline_envelope(
    row_count: int,
    rng: np.random.Generator,
    control_points_min: int = ADJUSTABLE_SPLINE_CONTROL_POINTS_MIN,
    control_points_max: int = ADJUSTABLE_SPLINE_CONTROL_POINTS_MAX,
    min_control_value: float = ADJUSTABLE_SPLINE_MIN_CONTROL_VALUE,
) -> tuple[np.ndarray, str]:
    """Return a smooth random envelope with exact zero endpoints."""
    if row_count <= 0:
        return np.array([], dtype=float), "control_points=0"
    if row_count == 1:
        return np.array([1.0], dtype=float), "control_points=0"
    if row_count == 2:
        return np.array([0.0, 0.0], dtype=float), "control_points=0"

    control_points_min = max(1, int(control_points_min))
    control_points_max = max(control_points_min, int(control_points_max))
    internal_count = int(rng.integers(control_points_min, control_points_max + 1))

    base_x = np.linspace(0, 1, internal_count + 2)[1:-1]
    spacing = 1 / (internal_count + 1)
    jitter = rng.uniform(-0.30 * spacing, 0.30 * spacing, size=internal_count)
    internal_x = np.sort(np.clip(base_x + jitter, spacing * 0.35, 1 - spacing * 0.35))

    internal_y = rng.uniform(min_control_value, 1.0, size=internal_count)
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
        if span <= 0:
            envelope[in_segment] = control_y[right]
            continue

        position = (x[in_segment] - control_x[left]) / span
        eased_position = 0.5 - 0.5 * np.cos(np.pi * position)
        envelope[in_segment] = (
            control_y[left]
            + (control_y[right] - control_y[left]) * eased_position
        )

    params = (
        f"control_points={internal_count};"
        f"min_control_value={min_control_value:.2f};"
        f"peak_control_index={peak_rank + 1}"
    )
    return normalize_envelope(envelope), params


def build_attack_envelope(
    row_count: int,
    envelope_type: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str, str]:
    if envelope_type == "random":
        envelope_type = str(rng.choice(ADJUSTABLE_RANDOM_ENVELOPE_TYPES))
    if envelope_type == "hann":
        envelope, params = hann_envelope(row_count)
        return envelope, envelope_type, params
    if envelope_type == "asymmetric_hann":
        envelope, params = asymmetric_hann_envelope(row_count, rng)
        return envelope, envelope_type, params
    if envelope_type == "beta":
        envelope, params = beta_curve_envelope(row_count, rng)
        return envelope, envelope_type, params
    if envelope_type == "raised_cosine":
        envelope, params = raised_cosine_envelope(row_count, rng)
        return envelope, envelope_type, params
    if envelope_type == "random_spline":
        envelope, params = random_control_point_spline_envelope(row_count, rng)
        return envelope, envelope_type, params
    raise ValueError(f"Unsupported envelope type: {envelope_type}")


def duration_amplitude_scale(
    attack_start_bin: int,
    attack_end_bin: int,
    bins: int,
    beta: float,
) -> tuple[int, float]:
    effective_duration = max(1, min(attack_end_bin, bins) - attack_start_bin)
    scale = (effective_duration / ATTACK_DURATION_REFERENCE) ** beta
    return effective_duration, float(scale)


def load_feature_std_lookup(stats_csv: Path, feature_name: str) -> dict[int, float]:
    stats = pd.read_csv(stats_csv)
    required_columns = ["bin_id", "feature", "std"]
    missing = [column for column in required_columns if column not in stats.columns]
    if missing:
        raise ValueError(f"{stats_csv} is missing columns: {missing}")

    feature_stats = stats[stats["feature"] == feature_name].copy()
    if feature_stats.empty:
        raise ValueError(f"{stats_csv} has no rows where feature == {feature_name!r}")

    feature_stats["bin_id"] = pd.to_numeric(
        feature_stats["bin_id"],
        errors="coerce",
    )
    feature_stats["std"] = pd.to_numeric(feature_stats["std"], errors="coerce")
    feature_stats = feature_stats.dropna(subset=["bin_id", "std"])
    feature_stats["bin_id"] = feature_stats["bin_id"].astype(int)

    return dict(zip(feature_stats["bin_id"], feature_stats["std"]))


def prepare_scalar_attack_metadata(
    df: pd.DataFrame,
    delta_column: str,
    std_column: str,
    offset_column: str,
) -> pd.DataFrame:
    df = df.copy()
    df["is_attacked"] = 0
    df["attack_type"] = "authentic"
    df["label"] = 0
    df["attack_k"] = 0.0
    df["attack_alpha"] = 0.0
    df["attack_direction"] = 0
    df["attack_start_bin"] = -1
    df["attack_duration"] = 0
    df["attack_end_bin"] = -1
    df["attack_effective_duration"] = 0
    df["attack_duration_scale"] = 0.0
    df["attack_envelope_type"] = "none"
    df["attack_envelope_params"] = ""
    df["attack_envelope"] = 0.0
    df[delta_column] = 0.0
    df[std_column] = 0.0
    df[offset_column] = 0.0
    return df


def generate_scalar_feature_attack(
    input_csv: Path,
    output_csv: Path,
    feature_name: str,
    attack_type: str,
    delta_column: str,
    std_column: str,
    offset_column: str,
    std_lookup: dict[int, float],
    bins: int,
    attack_start_bin: int,
    attack_duration: int,
    k: float,
    alpha: float,
    direction: int,
    label: int,
    duration_amplitude_beta: float,
    envelope_type: str,
    rng: np.random.Generator,
) -> tuple[int, str]:
    df = pd.read_csv(input_csv)
    required_columns = [ROUTE_PROGRESS_COLUMN, feature_name]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{input_csv} is missing columns: {missing}")

    df[ROUTE_PROGRESS_COLUMN] = pd.to_numeric(
        df[ROUTE_PROGRESS_COLUMN],
        errors="coerce",
    )
    df[feature_name] = pd.to_numeric(
        df[feature_name],
        errors="coerce",
    ).astype(float)
    if df[ROUTE_PROGRESS_COLUMN].isna().any():
        raise ValueError(f"{input_csv} has blank or non-numeric route_progress values")
    if df[feature_name].isna().any():
        raise ValueError(f"{input_csv} has blank or non-numeric {feature_name} values")

    attack_end_bin = attack_start_bin + attack_duration
    df = prepare_scalar_attack_metadata(
        df,
        delta_column=delta_column,
        std_column=std_column,
        offset_column=offset_column,
    )
    df["bin_id"] = assign_bin_id(df[ROUTE_PROGRESS_COLUMN], bins)

    in_attack_window = (
        (df["bin_id"] >= attack_start_bin)
        & (df["bin_id"] < attack_end_bin)
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not in_attack_window.any():
        df.to_csv(output_csv, index=False)
        return 0, "none"

    attack_bin_ids = df.loc[in_attack_window, "bin_id"].astype(int)
    missing_bins = sorted(set(attack_bin_ids) - set(std_lookup))
    if missing_bins:
        preview = ", ".join(str(bin_id) for bin_id in missing_bins[:10])
        suffix = "..." if len(missing_bins) > 10 else ""
        raise ValueError(f"{input_csv} has attack bins missing from stats: {preview}{suffix}")

    feature_std = attack_bin_ids.map(std_lookup).astype(float).to_numpy()
    envelope, actual_envelope_type, envelope_params = build_attack_envelope(
        len(attack_bin_ids),
        envelope_type=envelope_type,
        rng=rng,
    )
    peak_index = int(np.argmax(envelope))
    attack_std = float(feature_std[peak_index])
    effective_duration, duration_scale = duration_amplitude_scale(
        attack_start_bin,
        attack_end_bin,
        bins,
        duration_amplitude_beta,
    )
    scaled_std = attack_std * duration_scale
    delta_value = float(
        rng.normal(
            loc=direction * k * scaled_std,
            scale=alpha * scaled_std,
        )
    )
    applied_offset = envelope * delta_value
    attacked_index = df.index[in_attack_window]

    df.loc[attacked_index, feature_name] = (
        df.loc[attacked_index, feature_name].to_numpy() + applied_offset
    )
    df.loc[attacked_index, "is_attacked"] = 1
    df.loc[attacked_index, "attack_type"] = attack_type
    df.loc[attacked_index, "label"] = label
    df.loc[attacked_index, "attack_k"] = k
    df.loc[attacked_index, "attack_alpha"] = alpha
    df.loc[attacked_index, "attack_direction"] = direction
    df.loc[attacked_index, "attack_start_bin"] = attack_start_bin
    df.loc[attacked_index, "attack_duration"] = attack_duration
    df.loc[attacked_index, "attack_end_bin"] = attack_end_bin
    df.loc[attacked_index, "attack_effective_duration"] = effective_duration
    df.loc[attacked_index, "attack_duration_scale"] = duration_scale
    df.loc[attacked_index, "attack_envelope_type"] = actual_envelope_type
    df.loc[attacked_index, "attack_envelope_params"] = envelope_params
    df.loc[attacked_index, "attack_envelope"] = envelope
    df.loc[attacked_index, delta_column] = delta_value
    df.loc[attacked_index, std_column] = attack_std
    df.loc[attacked_index, offset_column] = applied_offset

    df.to_csv(output_csv, index=False)
    return int(in_attack_window.sum()), actual_envelope_type


def select_input_csvs(input_folder: Path, sample_size: int, rng: np.random.Generator) -> list[Path]:
    input_folder = resolve_dataset_path(input_folder)
    if input_folder.is_file():
        if input_folder.suffix.lower() != ".csv":
            raise ValueError(f"Input file is not a CSV: {input_folder}")
        return [input_folder]

    csv_paths = sorted(path for path in input_folder.glob("*.csv") if path.is_file())
    if not csv_paths:
        raise ValueError(f"No CSV files found in {input_folder}")

    sample_count = min(sample_size, len(csv_paths))
    selected_indices = rng.choice(len(csv_paths), size=sample_count, replace=False)
    return [csv_paths[int(index)] for index in selected_indices]


def prefixed_output_name(input_csv: Path, output_filename_prefix: str) -> str:
    if input_csv.name.startswith(output_filename_prefix):
        return input_csv.name
    return f"{output_filename_prefix}{input_csv.name}"


def random_attack_start_bin(bins: int, rng: np.random.Generator) -> int:
    max_start_bin = min(ADJUSTABLE_ATTACK_START_BIN_MAX, bins - 1)
    return int(rng.integers(ADJUSTABLE_ATTACK_START_BIN_MIN, max_start_bin + 1))


def random_attack_duration(rng: np.random.Generator) -> int:
    return int(
        rng.integers(
            ADJUSTABLE_ATTACK_DURATION_MIN,
            ADJUSTABLE_ATTACK_DURATION_MAX + 1,
        )
    )


def random_attack_direction(rng: np.random.Generator) -> int:
    return int(rng.choice([-1, 1]))


def validate_common_attack_settings(
    input_folder: Path,
    output_folder: Path,
    stats_csv: Path,
    bins: int,
    attack_start_bin: int | None,
    attack_duration: int | None,
    k: float,
    alpha: float,
    sample_size: int,
    duration_amplitude_beta: float | None = None,
) -> None:
    if not input_folder.exists():
        raise ValueError(f"Input path does not exist: {input_folder}")
    if not input_folder.is_dir() and input_folder.suffix.lower() != ".csv":
        raise ValueError(f"Input path must be a folder or CSV file: {input_folder}")
    if not stats_csv.is_file():
        raise ValueError(f"Statistics CSV does not exist: {stats_csv}")
    if output_folder.exists() and not output_folder.is_dir():
        raise ValueError(f"Output folder path is not a directory: {output_folder}")
    if bins <= 0:
        raise ValueError("--bins must be greater than 0")
    if attack_start_bin is not None and attack_start_bin < 0:
        raise ValueError("--attack-start-bin must be >= 0")
    if attack_start_bin is not None and attack_start_bin >= bins:
        raise ValueError("--attack-start-bin must be less than --bins")
    if attack_duration is not None and attack_duration <= 0:
        raise ValueError("--attack-duration must be greater than 0")
    if k < 0:
        raise ValueError("--k must be >= 0")
    if alpha < 0:
        raise ValueError("--alpha must be >= 0")
    if sample_size <= 0:
        raise ValueError("--sample-size must be greater than 0")
    if duration_amplitude_beta is not None and duration_amplitude_beta < 0:
        raise ValueError("--duration-amplitude-beta must be >= 0")
    if ADJUSTABLE_SPLINE_CONTROL_POINTS_MIN <= 0:
        raise ValueError("ADJUSTABLE_SPLINE_CONTROL_POINTS_MIN must be > 0")
    if ADJUSTABLE_SPLINE_CONTROL_POINTS_MAX < ADJUSTABLE_SPLINE_CONTROL_POINTS_MIN:
        raise ValueError(
            "ADJUSTABLE_SPLINE_CONTROL_POINTS_MAX must be >= "
            "ADJUSTABLE_SPLINE_CONTROL_POINTS_MIN"
        )
    if ADJUSTABLE_SPLINE_MIN_CONTROL_VALUE < 0:
        raise ValueError("ADJUSTABLE_SPLINE_MIN_CONTROL_VALUE must be >= 0")
    if not ADJUSTABLE_RANDOM_ENVELOPE_TYPES:
        raise ValueError("ADJUSTABLE_RANDOM_ENVELOPE_TYPES must not be empty")
    if ADJUSTABLE_ASYMMETRIC_HANN_PEAK_MIN <= 0:
        raise ValueError("ADJUSTABLE_ASYMMETRIC_HANN_PEAK_MIN must be > 0")
    if ADJUSTABLE_ASYMMETRIC_HANN_PEAK_MAX >= 1:
        raise ValueError("ADJUSTABLE_ASYMMETRIC_HANN_PEAK_MAX must be < 1")
    if ADJUSTABLE_ASYMMETRIC_HANN_PEAK_MAX < ADJUSTABLE_ASYMMETRIC_HANN_PEAK_MIN:
        raise ValueError(
            "ADJUSTABLE_ASYMMETRIC_HANN_PEAK_MAX must be >= "
            "ADJUSTABLE_ASYMMETRIC_HANN_PEAK_MIN"
        )
    if ADJUSTABLE_BETA_SHAPE_MIN <= 1:
        raise ValueError("ADJUSTABLE_BETA_SHAPE_MIN must be > 1")
    if ADJUSTABLE_BETA_SHAPE_MAX < ADJUSTABLE_BETA_SHAPE_MIN:
        raise ValueError("ADJUSTABLE_BETA_SHAPE_MAX must be >= ADJUSTABLE_BETA_SHAPE_MIN")
    if ADJUSTABLE_RAISED_COSINE_TAPER_MIN <= 0:
        raise ValueError("ADJUSTABLE_RAISED_COSINE_TAPER_MIN must be > 0")
    if ADJUSTABLE_RAISED_COSINE_TAPER_MAX < ADJUSTABLE_RAISED_COSINE_TAPER_MIN:
        raise ValueError(
            "ADJUSTABLE_RAISED_COSINE_TAPER_MAX must be >= "
            "ADJUSTABLE_RAISED_COSINE_TAPER_MIN"
        )
