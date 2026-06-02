from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mod_tuner import ADJUSTABLE_ALPHA
from mod_tuner import ADJUSTABLE_ATTACK_DURATION
from mod_tuner import ADJUSTABLE_ATTACK_DURATION_MAX
from mod_tuner import ADJUSTABLE_ATTACK_DURATION_MIN
from mod_tuner import ADJUSTABLE_ATTACK_START_BIN
from mod_tuner import ADJUSTABLE_ATTACK_START_BIN_MAX
from mod_tuner import ADJUSTABLE_ATTACK_START_BIN_MIN
from mod_tuner import ADJUSTABLE_BIN_COUNT
from mod_tuner import ADJUSTABLE_ENVELOPE_TYPE
from mod_tuner import ADJUSTABLE_K
from mod_tuner import ADJUSTABLE_RANDOM_SEED
from mod_tuner import ADJUSTABLE_SAMPLE_SIZE
from mod_tuner import ENVELOPE_CHOICES
from mod_tuner import assign_bin_id
from mod_tuner import build_attack_envelope
from mod_tuner import prefixed_output_name
from mod_tuner import random_attack_direction
from mod_tuner import random_attack_duration
from mod_tuner import random_attack_start_bin
from mod_tuner import select_input_csvs
from mod_tuner import validate_common_attack_settings


X_FEATURE_NAME = "x_wrt0"
Y_FEATURE_NAME = "y_wrt0"
LATITUDE_COLUMN = "latitude"
LONGITUDE_COLUMN = "longitude"
ROUTE_PROGRESS_COLUMN = "route_progress"
ATTACK_TYPE = "modified_position"
OUTPUT_FILENAME_PREFIX = "modpos_"
EARTH_RADIUS_M = 6_371_008.8

# =========================
# Adjustable attack settings
# =========================

ADJUSTABLE_LABEL_MOD_POSITION = 3
ADJUSTABLE_DIRECTION_X = None  # None=random per CSV, or use -1 / +1.
ADJUSTABLE_DIRECTION_Y = None  # None=random per CSV, or use -1 / +1.


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate modified-position attack CSVs from a random sample of "
            "authentic flight CSVs."
        )
    )
    parser.add_argument(
        "input_folder",
        type=Path,
        help="Authentic one-flight CSV file or folder containing CSV files.",
    )
    parser.add_argument(
        "--covariance-stats",
        "--covariance",
        "--stats",
        dest="covariance_stats",
        type=Path,
        required=True,
        help=(
            "Position covariance CSV with bin_id,var_x,var_y,cov_xy to use "
            "for this run."
        ),
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        required=True,
        help="Folder where modified-position CSVs will be written.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=ADJUSTABLE_BIN_COUNT,
        help=f"Number of route-progress bins. Default: {ADJUSTABLE_BIN_COUNT}.",
    )
    parser.add_argument(
        "--attack-start-bin",
        type=int,
        default=ADJUSTABLE_ATTACK_START_BIN,
        help=(
            "Optional fixed first route-progress bin in the attack window. "
            "Default: random integer from "
            f"{ADJUSTABLE_ATTACK_START_BIN_MIN} to "
            f"{ADJUSTABLE_ATTACK_START_BIN_MAX} for each CSV."
        ),
    )
    parser.add_argument(
        "--attack-duration",
        type=int,
        default=ADJUSTABLE_ATTACK_DURATION,
        help=(
            "Optional fixed number of bins in the attack window. "
            "Default: random integer from "
            f"{ADJUSTABLE_ATTACK_DURATION_MIN} to "
            f"{ADJUSTABLE_ATTACK_DURATION_MAX} for each CSV."
        ),
    )
    parser.add_argument(
        "--k",
        type=float,
        default=ADJUSTABLE_K,
        help=f"Attack intensity multiplier. Default: {ADJUSTABLE_K}.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=ADJUSTABLE_ALPHA,
        help=f"2D Gaussian covariance multiplier. Default: {ADJUSTABLE_ALPHA}.",
    )
    parser.add_argument(
        "--direction-x",
        type=int,
        choices=(-1, 1),
        default=ADJUSTABLE_DIRECTION_X,
        help=(
            "Optional fixed x attack direction. Default: random choice of -1 "
            "or +1 for each CSV."
        ),
    )
    parser.add_argument(
        "--direction-y",
        type=int,
        choices=(-1, 1),
        default=ADJUSTABLE_DIRECTION_Y,
        help=(
            "Optional fixed y attack direction. Default: random choice of -1 "
            "or +1 for each CSV."
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=ADJUSTABLE_SAMPLE_SIZE,
        help=f"Number of random CSVs to process. Default: {ADJUSTABLE_SAMPLE_SIZE}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=ADJUSTABLE_RANDOM_SEED,
        help=(
            "Random seed for file sampling and 2D Gaussian offsets. "
            f"Default: {ADJUSTABLE_RANDOM_SEED}."
        ),
    )
    parser.add_argument(
        "--label",
        type=int,
        default=ADJUSTABLE_LABEL_MOD_POSITION,
        help=(
            "Class label for modified position rows. "
            f"Default: {ADJUSTABLE_LABEL_MOD_POSITION}."
        ),
    )
    parser.add_argument(
        "--envelope-type",
        choices=ENVELOPE_CHOICES,
        default=ADJUSTABLE_ENVELOPE_TYPE,
        help=(
            "Attack envelope shape. random chooses one envelope per CSV. "
            f"Default: {ADJUSTABLE_ENVELOPE_TYPE}."
        ),
    )
    return parser.parse_args(argv)


def load_position_covariance_lookup(
    covariance_csv: Path,
) -> dict[int, tuple[float, float, float]]:
    covariance = pd.read_csv(covariance_csv)
    required_columns = ["bin_id", "var_x", "var_y", "cov_xy"]
    missing = [column for column in required_columns if column not in covariance.columns]
    if missing:
        raise ValueError(f"{covariance_csv} is missing columns: {missing}")

    covariance = covariance[required_columns].copy()
    for column in required_columns:
        covariance[column] = pd.to_numeric(covariance[column], errors="coerce")
    covariance = covariance.dropna(subset=required_columns)
    covariance["bin_id"] = covariance["bin_id"].astype(int)

    duplicated = covariance[covariance["bin_id"].duplicated(keep=False)]
    if not duplicated.empty:
        duplicate_bins = sorted(duplicated["bin_id"].unique())
        preview = ", ".join(str(bin_id) for bin_id in duplicate_bins[:10])
        suffix = "..." if len(duplicate_bins) > 10 else ""
        raise ValueError(f"{covariance_csv} has duplicate bin_id values: {preview}{suffix}")

    return {
        int(row.bin_id): (float(row.var_x), float(row.var_y), float(row.cov_xy))
        for row in covariance.itertuples(index=False)
    }


def prepare_attack_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_attacked"] = 0
    df["attack_type"] = "authentic"
    df["label"] = 0
    df["attack_k"] = 0.0
    df["attack_alpha"] = 0.0
    df["attack_direction"] = 0
    df["attack_direction_x"] = 0
    df["attack_direction_y"] = 0
    df["attack_start_bin"] = -1
    df["attack_duration"] = 0
    df["attack_end_bin"] = -1
    df["attack_effective_duration"] = 0
    df["attack_duration_scale"] = 1.0
    df["attack_center_bin"] = -1
    df["attack_envelope_type"] = "none"
    df["attack_envelope_params"] = ""
    df["attack_envelope"] = 0.0
    df["attack_var_x"] = 0.0
    df["attack_var_y"] = 0.0
    df["attack_cov_xy"] = 0.0
    df["attack_mean_delta_x"] = 0.0
    df["attack_mean_delta_y"] = 0.0
    df["attack_delta_x"] = 0.0
    df["attack_delta_y"] = 0.0
    df["x_wrt0_offset"] = 0.0
    df["y_wrt0_offset"] = 0.0
    return df


def effective_attack_duration(
    attack_start_bin: int,
    attack_end_bin: int,
    bins: int,
) -> int:
    return max(1, min(attack_end_bin, bins) - attack_start_bin)


def attack_reference_bin(
    attack_start_bin: int,
    attack_end_bin: int,
    bins: int,
) -> int:
    effective_end_bin = min(attack_end_bin, bins)
    return int((attack_start_bin + effective_end_bin - 1) // 2)


def validate_covariance_matrix(
    var_x: float,
    var_y: float,
    cov_xy: float,
) -> np.ndarray:
    if not np.isfinite([var_x, var_y, cov_xy]).all():
        raise ValueError("position covariance values must be finite")
    if var_x < 0 or var_y < 0:
        raise ValueError(
            f"position variances must be non-negative, got var_x={var_x}, var_y={var_y}"
        )

    covariance = np.array([[var_x, cov_xy], [cov_xy, var_y]], dtype=float)
    min_eigenvalue = float(np.linalg.eigvalsh(covariance).min())
    if min_eigenvalue < -1e-8:
        raise ValueError(
            "position covariance matrix is not positive semidefinite "
            f"(minimum eigenvalue {min_eigenvalue})"
        )
    if min_eigenvalue < 0:
        covariance = covariance + np.eye(2) * abs(min_eigenvalue)
    return covariance


def sample_position_offset(
    var_x: float,
    var_y: float,
    cov_xy: float,
    k: float,
    alpha: float,
    direction_x: int,
    direction_y: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    covariance = validate_covariance_matrix(var_x, var_y, cov_xy)
    mean = k * np.array(
        [
            direction_x * np.sqrt(var_x),
            direction_y * np.sqrt(var_y),
        ],
        dtype=float,
    )
    sampled_offset = rng.multivariate_normal(
        mean=mean,
        cov=(alpha ** 2) * covariance,
        check_valid="raise",
    )
    return mean, covariance, sampled_offset.astype(float)


def latitude_longitude_from_relative_xy(
    x: np.ndarray,
    y: np.ndarray,
    origin_latitude: float,
    origin_longitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    start_lat_rad = np.radians(origin_latitude)
    start_lon_rad = np.radians(origin_longitude)

    lat_rad = start_lat_rad + (y / EARTH_RADIUS_M)
    lon_scale = np.cos((lat_rad + start_lat_rad) / 2) * EARTH_RADIUS_M
    if np.any(np.abs(lon_scale) < 1e-12):
        raise ValueError("longitude scale is too close to zero for relative x conversion")
    lon_rad = start_lon_rad + (x / lon_scale)

    return np.degrees(lat_rad), np.degrees(lon_rad)


def generate_modified_position_attack(
    input_csv: Path,
    output_csv: Path,
    covariance_lookup: dict[int, tuple[float, float, float]],
    bins: int,
    attack_start_bin: int,
    attack_duration: int,
    k: float,
    alpha: float,
    direction_x: int,
    direction_y: int,
    label: int,
    envelope_type: str,
    rng: np.random.Generator,
) -> tuple[int, str]:
    df = pd.read_csv(input_csv)
    required_columns = [
        ROUTE_PROGRESS_COLUMN,
        X_FEATURE_NAME,
        Y_FEATURE_NAME,
        LATITUDE_COLUMN,
        LONGITUDE_COLUMN,
    ]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{input_csv} is missing columns: {missing}")

    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[ROUTE_PROGRESS_COLUMN].isna().any():
        raise ValueError(f"{input_csv} has blank or non-numeric route_progress values")
    for column in [X_FEATURE_NAME, Y_FEATURE_NAME, LATITUDE_COLUMN, LONGITUDE_COLUMN]:
        if df[column].isna().any():
            raise ValueError(f"{input_csv} has blank or non-numeric {column} values")

    origin_latitude = float(df[LATITUDE_COLUMN].iloc[0])
    origin_longitude = float(df[LONGITUDE_COLUMN].iloc[0])
    attack_end_bin = attack_start_bin + attack_duration

    df = prepare_attack_metadata(df)
    df["bin_id"] = assign_bin_id(df[ROUTE_PROGRESS_COLUMN], bins)

    in_attack_window = (
        (df["bin_id"] >= attack_start_bin)
        & (df["bin_id"] < attack_end_bin)
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not in_attack_window.any():
        df.to_csv(output_csv, index=False)
        return 0, "none"

    center_bin = attack_reference_bin(attack_start_bin, attack_end_bin, bins)
    if center_bin not in covariance_lookup:
        raise ValueError(f"{input_csv} attack center bin {center_bin} is missing from covariance stats")

    var_x, var_y, cov_xy = covariance_lookup[center_bin]
    mean_offset, covariance, sampled_offset = sample_position_offset(
        var_x=var_x,
        var_y=var_y,
        cov_xy=cov_xy,
        k=k,
        alpha=alpha,
        direction_x=direction_x,
        direction_y=direction_y,
        rng=rng,
    )

    attacked_index = df.index[in_attack_window]
    envelope, actual_envelope_type, envelope_params = build_attack_envelope(
        len(attacked_index),
        envelope_type=envelope_type,
        rng=rng,
    )

    x_offsets = envelope * float(sampled_offset[0])
    y_offsets = envelope * float(sampled_offset[1])
    df.loc[attacked_index, X_FEATURE_NAME] = (
        df.loc[attacked_index, X_FEATURE_NAME].to_numpy(dtype=float) + x_offsets
    )
    df.loc[attacked_index, Y_FEATURE_NAME] = (
        df.loc[attacked_index, Y_FEATURE_NAME].to_numpy(dtype=float) + y_offsets
    )

    attacked_x = df.loc[attacked_index, X_FEATURE_NAME].to_numpy(dtype=float)
    attacked_y = df.loc[attacked_index, Y_FEATURE_NAME].to_numpy(dtype=float)
    attacked_latitude, attacked_longitude = latitude_longitude_from_relative_xy(
        attacked_x,
        attacked_y,
        origin_latitude=origin_latitude,
        origin_longitude=origin_longitude,
    )
    df.loc[attacked_index, LATITUDE_COLUMN] = attacked_latitude
    df.loc[attacked_index, LONGITUDE_COLUMN] = attacked_longitude

    effective_duration = effective_attack_duration(attack_start_bin, attack_end_bin, bins)
    df.loc[attacked_index, "is_attacked"] = 1
    df.loc[attacked_index, "attack_type"] = ATTACK_TYPE
    df.loc[attacked_index, "label"] = label
    df.loc[attacked_index, "attack_k"] = k
    df.loc[attacked_index, "attack_alpha"] = alpha
    df.loc[attacked_index, "attack_direction_x"] = direction_x
    df.loc[attacked_index, "attack_direction_y"] = direction_y
    df.loc[attacked_index, "attack_start_bin"] = attack_start_bin
    df.loc[attacked_index, "attack_duration"] = attack_duration
    df.loc[attacked_index, "attack_end_bin"] = attack_end_bin
    df.loc[attacked_index, "attack_effective_duration"] = effective_duration
    df.loc[attacked_index, "attack_center_bin"] = center_bin
    df.loc[attacked_index, "attack_envelope_type"] = actual_envelope_type
    df.loc[attacked_index, "attack_envelope_params"] = envelope_params
    df.loc[attacked_index, "attack_envelope"] = envelope
    df.loc[attacked_index, "attack_var_x"] = float(covariance[0, 0])
    df.loc[attacked_index, "attack_var_y"] = float(covariance[1, 1])
    df.loc[attacked_index, "attack_cov_xy"] = float(covariance[0, 1])
    df.loc[attacked_index, "attack_mean_delta_x"] = float(mean_offset[0])
    df.loc[attacked_index, "attack_mean_delta_y"] = float(mean_offset[1])
    df.loc[attacked_index, "attack_delta_x"] = float(sampled_offset[0])
    df.loc[attacked_index, "attack_delta_y"] = float(sampled_offset[1])
    df.loc[attacked_index, "x_wrt0_offset"] = x_offsets
    df.loc[attacked_index, "y_wrt0_offset"] = y_offsets

    df.to_csv(output_csv, index=False)
    return int(in_attack_window.sum()), actual_envelope_type


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv)
        validate_common_attack_settings(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            stats_csv=args.covariance_stats,
            bins=args.bins,
            attack_start_bin=args.attack_start_bin,
            attack_duration=args.attack_duration,
            k=args.k,
            alpha=args.alpha,
            sample_size=args.sample_size,
        )

        file_rng = np.random.default_rng(args.seed)
        attack_rng = np.random.default_rng(args.seed + 1)
        input_csvs = select_input_csvs(args.input_folder, args.sample_size, file_rng)
        covariance_lookup = load_position_covariance_lookup(args.covariance_stats)

        args.output_folder.mkdir(parents=True, exist_ok=True)
        errors = 0
        total_attacked_rows = 0
        for input_csv in input_csvs:
            output_csv = args.output_folder / prefixed_output_name(
                input_csv,
                OUTPUT_FILENAME_PREFIX,
            )
            attack_start_bin = (
                args.attack_start_bin
                if args.attack_start_bin is not None
                else random_attack_start_bin(args.bins, attack_rng)
            )
            attack_duration = (
                args.attack_duration
                if args.attack_duration is not None
                else random_attack_duration(attack_rng)
            )
            direction_x = (
                args.direction_x
                if args.direction_x is not None
                else random_attack_direction(attack_rng)
            )
            direction_y = (
                args.direction_y
                if args.direction_y is not None
                else random_attack_direction(attack_rng)
            )
            try:
                attacked_rows, actual_envelope_type = generate_modified_position_attack(
                    input_csv=input_csv,
                    output_csv=output_csv,
                    covariance_lookup=covariance_lookup,
                    bins=args.bins,
                    attack_start_bin=attack_start_bin,
                    attack_duration=attack_duration,
                    k=args.k,
                    alpha=args.alpha,
                    direction_x=direction_x,
                    direction_y=direction_y,
                    label=args.label,
                    envelope_type=args.envelope_type,
                    rng=attack_rng,
                )
                total_attacked_rows += attacked_rows
                print(
                    f"Wrote {output_csv} ({attacked_rows} attacked rows, "
                    f"attack bins {attack_start_bin}-{attack_start_bin + attack_duration}, "
                    f"directions x={direction_x}, y={direction_y}, "
                    f"envelope {actual_envelope_type})"
                )
            except Exception as exc:
                errors += 1
                print(f"ERROR {input_csv}: {exc}", file=sys.stderr)

        print(f"Input folder: {args.input_folder}")
        print(f"Covariance CSV: {args.covariance_stats}")
        print(f"Output folder: {args.output_folder}")
        print(f"Selected CSVs: {len(input_csvs)}")
        print(f"Total attacked rows: {total_attacked_rows}")
        print(f"Errors: {errors}")
        return 1 if errors else 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
