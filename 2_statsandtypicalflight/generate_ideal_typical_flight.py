from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flight_statistics import resolve_csv_paths
from generate_attack_tuning_statistics import ATTACK_FEATURES
from generate_attack_tuning_statistics import DEFAULT_BINS
from generate_attack_tuning_statistics import ROUTE_PROGRESS_COLUMN
from generate_attack_tuning_statistics import load_flight_bin_values


DEFAULT_OUTPUT_PATH = Path("ideal_typical_flight.csv")
VISUALIZATION_FEATURES = ["latitude", "longitude", "verticalSpeed_ms"]
IDEAL_FEATURES = [*VISUALIZATION_FEATURES, *ATTACK_FEATURES]
OUTPUT_FEATURES = [
    feature for feature in IDEAL_FEATURES if feature != ROUTE_PROGRESS_COLUMN
]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an unsmoothed ideal typical flight by route_progress bin. "
            "Each flight contributes one median value per feature per bin, then "
            "the typical value is the across-flight median."
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
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    return parser.parse_args(argv)


def build_ideal_typical_flight_long(
    all_flight_bin_values: pd.DataFrame,
    bins: int,
) -> pd.DataFrame:
    grouped_medians = (
        all_flight_bin_values.groupby(["bin_id", "feature"])["flight_bin_value"]
        .median()
        .to_dict()
    )

    rows = []
    for bin_id in range(bins):
        for feature in IDEAL_FEATURES:
            rows.append(
                {
                    "bin_id": bin_id,
                    "feature": feature,
                    "ideal_value": grouped_medians.get((bin_id, feature), np.nan),
                }
            )
    return pd.DataFrame(rows, columns=["bin_id", "feature", "ideal_value"])


def build_ideal_typical_flight_wide(
    ideal_typical_flight_long: pd.DataFrame,
    bins: int,
) -> pd.DataFrame:
    wide = ideal_typical_flight_long.pivot(
        index="bin_id",
        columns="feature",
        values="ideal_value",
    ).reset_index()

    for feature in OUTPUT_FEATURES:
        if feature not in wide.columns:
            wide[feature] = np.nan

    wide["route_progress_center"] = (wide["bin_id"] + 0.5) / bins
    wide["route_progress_bin_center"] = wide["route_progress_center"]
    columns = [
        "bin_id",
        "route_progress_center",
        "route_progress_bin_center",
        *OUTPUT_FEATURES,
    ]
    return wide[columns]


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
            flight_bin_values = load_flight_bin_values(
                csv_path,
                args.bins,
                features=IDEAL_FEATURES,
            )
            if not flight_bin_values.empty:
                flight_bin_frames.append(flight_bin_values)
        except Exception as exc:
            errors += 1
            print(f"ERROR {csv_path}: {exc}", file=sys.stderr)

    if not flight_bin_frames:
        print("No valid flight-bin values found.", file=sys.stderr)
        return 1

    all_flight_bin_values = pd.concat(flight_bin_frames, ignore_index=True)
    ideal_long = build_ideal_typical_flight_long(all_flight_bin_values, args.bins)
    ideal_wide = build_ideal_typical_flight_wide(ideal_long, args.bins)
    ideal_wide.to_csv(args.output, index=False)

    print(f"Wrote {args.output}: {len(ideal_wide)} route-progress bin rows")
    print(f"Flights read: {all_flight_bin_values['flight_id'].nunique()}")
    print(f"Flight-bin-feature values: {len(all_flight_bin_values)}")
    print(f"Errors: {errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
