from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flight_statistics import FLIGHT_ID_COLUMN
from flight_statistics import STAT_COLUMNS
from flight_statistics import describe_numeric_values
from flight_statistics import resolve_csv_paths


DEFAULT_BINS = 2000
DEFAULT_OUTPUT_PATH = Path("attack_tuning_statistics.csv")
ROUTE_PROGRESS_COLUMN = "route_progress"

ATTACK_FEATURES = [
    "x_wrt0",
    "y_wrt0",
    "altitude_meters",
    "speed_kmh",
    "delta_altitude",
    "delta_speed",
    "delta_heading",
    "turn_rate",
    "acceleration",
    "distance_per_timestep",
    "route_progress",
]

ALL_FLIGHT_BIN_COLUMNS = [
    "flight_id",
    "bin_id",
    "feature",
    "flight_bin_value",
]
POSITION_COVARIANCE_COLUMNS = ["bin_id", "var_x", "var_y", "cov_xy"]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate unsmoothed attack tuning statistics by route_progress bin. "
            "Each flight contributes one median value per feature per bin."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="CSV files or directories containing one-flight-per-CSV route data.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_BINS,
        help=f"Number of route_progress bins. Default: {DEFAULT_BINS}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Attack tuning output CSV path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--covariance-output",
        type=Path,
        help="Optional output CSV path for x_wrt0/y_wrt0 covariance by bin.",
    )
    return parser.parse_args(argv)


def flight_id_from_csv(df: pd.DataFrame, csv_path: Path) -> str:
    if FLIGHT_ID_COLUMN not in df.columns:
        return csv_path.stem

    flight_ids = df[FLIGHT_ID_COLUMN].dropna().astype(str).str.strip()
    flight_ids = flight_ids[flight_ids != ""].unique()
    if len(flight_ids) == 0:
        return csv_path.stem
    return str(flight_ids[0])


def assign_route_progress_bins(route_progress: pd.Series, bins: int) -> pd.Series:
    bin_ids = np.floor(route_progress * bins)
    bin_ids = bin_ids.clip(0, bins - 1)
    return bin_ids.astype("Int64")


def load_flight_bin_values(
    csv_path: Path,
    bins: int,
    features: list[str] | None = None,
) -> pd.DataFrame:
    features = ATTACK_FEATURES if features is None else features
    df = pd.read_csv(csv_path)
    required_columns = list(dict.fromkeys([ROUTE_PROGRESS_COLUMN, *features]))
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}")

    df = df.copy()
    flight_id = flight_id_from_csv(df, csv_path)
    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=[ROUTE_PROGRESS_COLUMN])
    if df.empty:
        return pd.DataFrame(columns=ALL_FLIGHT_BIN_COLUMNS)

    df["bin_id"] = assign_route_progress_bins(df[ROUTE_PROGRESS_COLUMN], bins)
    df = df.dropna(subset=["bin_id"])
    if df.empty:
        return pd.DataFrame(columns=ALL_FLIGHT_BIN_COLUMNS)

    flight_bin_medians = (
        df.groupby("bin_id", sort=True)[features]
        .median()
        .reset_index()
    )
    flight_bin_medians["bin_id"] = flight_bin_medians["bin_id"].astype(int)

    flight_bin_values = flight_bin_medians.melt(
        id_vars="bin_id",
        value_vars=features,
        var_name="feature",
        value_name="flight_bin_value",
    )
    flight_bin_values.insert(0, "flight_id", flight_id)
    return flight_bin_values[ALL_FLIGHT_BIN_COLUMNS]


def build_attack_tuning_statistics(
    all_flight_bin_values: pd.DataFrame,
    bins: int,
) -> pd.DataFrame:
    grouped_statistics = {
        key: describe_numeric_values(values)
        for key, values in all_flight_bin_values.groupby(
            ["bin_id", "feature"]
        )["flight_bin_value"]
    }
    empty_statistics = describe_numeric_values(pd.Series(dtype=float))

    rows = []
    for bin_id in range(bins):
        for feature in ATTACK_FEATURES:
            row = {"bin_id": bin_id, "feature": feature}
            row.update(grouped_statistics.get((bin_id, feature), empty_statistics))
            rows.append(row)

    return pd.DataFrame(rows, columns=["bin_id", "feature", *STAT_COLUMNS])


def build_position_covariance_statistics(
    all_flight_bin_values: pd.DataFrame,
    bins: int,
) -> pd.DataFrame:
    position_values = all_flight_bin_values[
        all_flight_bin_values["feature"].isin(["x_wrt0", "y_wrt0"])
    ]
    if position_values.empty:
        return pd.DataFrame(
            [{"bin_id": bin_id, "var_x": np.nan, "var_y": np.nan, "cov_xy": np.nan}
             for bin_id in range(bins)],
            columns=POSITION_COVARIANCE_COLUMNS,
        )

    paired_positions = (
        position_values.pivot_table(
            index=["flight_id", "bin_id"],
            columns="feature",
            values="flight_bin_value",
            aggfunc="first",
        )
        .reset_index()
    )
    for column in ["x_wrt0", "y_wrt0"]:
        if column not in paired_positions.columns:
            paired_positions[column] = np.nan

    rows = []
    for bin_id in range(bins):
        group = paired_positions[paired_positions["bin_id"] == bin_id]
        group = group.dropna(subset=["x_wrt0", "y_wrt0"])
        rows.append(
            {
                "bin_id": bin_id,
                "var_x": group["x_wrt0"].var(),
                "var_y": group["y_wrt0"].var(),
                "cov_xy": group["x_wrt0"].cov(group["y_wrt0"]),
            }
        )
    return pd.DataFrame(rows, columns=POSITION_COVARIANCE_COLUMNS)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv)
        if args.bins <= 0:
            raise ValueError("--bins must be greater than 0")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    csv_paths = resolve_csv_paths(args.paths)
    if not csv_paths:
        print("No CSV files found.", file=sys.stderr)
        return 1

    flight_bin_frames = []
    errors = 0
    for csv_path in csv_paths:
        try:
            flight_bin_values = load_flight_bin_values(csv_path, args.bins)
            if not flight_bin_values.empty:
                flight_bin_frames.append(flight_bin_values)
        except Exception as exc:
            errors += 1
            print(f"ERROR {csv_path}: {exc}", file=sys.stderr)

    if not flight_bin_frames:
        print("No valid flight-bin values found.", file=sys.stderr)
        return 1

    all_flight_bin_values = pd.concat(flight_bin_frames, ignore_index=True)
    attack_statistics = build_attack_tuning_statistics(
        all_flight_bin_values,
        args.bins,
    )
    attack_statistics.to_csv(args.output, index=False)

    if args.covariance_output is not None:
        covariance_statistics = build_position_covariance_statistics(
            all_flight_bin_values,
            args.bins,
        )
        covariance_statistics.to_csv(args.covariance_output, index=False)
        print(
            f"Wrote {args.covariance_output}: "
            f"{len(covariance_statistics)} covariance rows"
        )

    print(f"Wrote {args.output}: {len(attack_statistics)} bin-feature rows")
    print(f"Flights read: {all_flight_bin_values['flight_id'].nunique()}")
    print(f"Flight-bin-feature values: {len(all_flight_bin_values)}")
    print(f"Errors: {errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
