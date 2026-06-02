from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_paths import resolve_dataset_path


DEFAULT_OUTPUT_PATH = Path("flight_statistics_summary.csv")

FLIGHT_ID_COLUMN = "flight_id"
ROUTE_COLUMN = "route"
PHASE_COLUMN = "phase"

ROUTE_CODES = {
    1: "MNL-CEB",
    2: "MNL-DVO",
    3: "MNL-ILO",
    4: "MNL-MPH",
    5: "MNL-PPS",
}
ROUTE_FOLDERS = {
    "data_nolvl_ceb": "MNL-CEB",
    "data_nolvl_dvo": "MNL-DVO",
    "data_nolvl_ilo": "MNL-ILO",
    "data_nolvl_mph": "MNL-MPH",
    "data_nolvl_pps": "MNL-PPS",
}
ROUTE_TO_CODE = {route: code for code, route in ROUTE_CODES.items()}

PHASE_CODES = {
    1: "TKF",
    3: "CLB",
    4: "CRZ",
    5: "DSC",
    6: "APP",
}
PHASE_NAMES = {
    "takeoff": "TKF",
    "tkf": "TKF",
    "climb": "CLB",
    "clb": "CLB",
    "cruise": "CRZ",
    "crz": "CRZ",
    "descent": "DSC",
    "dsc": "DSC",
    "approach": "APP",
    "app": "APP",
}
REQUESTED_PHASES = ["TKF", "CLB", "CRZ", "DSC", "APP"]

FEATURES_FOR_STATS = [
    "altitude_meters",
    "speed_kmh",
    "verticalSpeed_ms",
    "x_wrt0",
    "y_wrt0",
    "delta_altitude",
    "delta_speed",
    "delta_heading",
    "turn_rate",
    "acceleration",
    "distance_per_timestep",
    "route_progress",
]

STAT_COLUMNS = ["mean", "std", "median", "iqr", "min", "max", "p05", "p95"]


def describe_numeric_values(values: pd.Series) -> dict[str, float]:
    values = values.dropna()
    quantile_25 = values.quantile(0.25)
    quantile_75 = values.quantile(0.75)
    return {
        "mean": values.mean(),
        "std": values.std(),
        "median": values.median(),
        "iqr": quantile_75 - quantile_25,
        "min": values.min(),
        "max": values.max(),
        "p05": values.quantile(0.05),
        "p95": values.quantile(0.95),
    }


def resolve_csv_paths(args: list[str]) -> list[Path]:
    input_paths = [Path(arg) for arg in args]
    csv_paths: list[Path] = []
    for path in (resolve_dataset_path(path) for path in input_paths):
        if path.is_dir():
            csv_paths.extend(sorted(path.glob("*.csv")))
        else:
            csv_paths.append(path)
    return csv_paths


def route_from_folder(csv_path: Path) -> str | None:
    for parent in [csv_path.parent, *csv_path.parents]:
        route = ROUTE_FOLDERS.get(parent.name)
        if route is not None:
            return route
    return None


def normalize_route(value, csv_path: Path) -> str | None:
    route_from_path = route_from_folder(csv_path)
    if pd.isna(value) or str(value).strip() == "":
        return route_from_path

    route_text = str(value).strip()
    try:
        route_code = int(float(route_text))
    except ValueError:
        route_code = None

    if route_code in ROUTE_CODES:
        return ROUTE_CODES[route_code]

    route_text_upper = route_text.upper()
    if route_text_upper in ROUTE_TO_CODE:
        return route_text_upper

    return route_from_path


def normalize_phase(value) -> str | None:
    if pd.isna(value) or str(value).strip() == "":
        return None

    phase_text = str(value).strip()
    try:
        phase_code = int(float(phase_text))
    except ValueError:
        phase_code = None

    if phase_code in PHASE_CODES:
        return PHASE_CODES[phase_code]

    return PHASE_NAMES.get(phase_text.lower())


def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_columns = [FLIGHT_ID_COLUMN, ROUTE_COLUMN, PHASE_COLUMN]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}")

    missing_features = [
        column for column in FEATURES_FOR_STATS if column not in df.columns
    ]
    if missing_features:
        raise ValueError(f"missing feature columns {missing_features}")

    df = df.copy()
    df["source_file"] = csv_path.name
    df["route_label"] = df[ROUTE_COLUMN].apply(lambda value: normalize_route(value, csv_path))
    df["route_code"] = df["route_label"].map(ROUTE_TO_CODE)
    df["phase_label"] = df[PHASE_COLUMN].apply(normalize_phase)
    for feature in FEATURES_FOR_STATS:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df[
        df["route_label"].isin(ROUTE_TO_CODE.keys())
        & df["phase_label"].isin(REQUESTED_PHASES)
    ].copy()

    rows = []
    for route_code, route_label in ROUTE_CODES.items():
        route_rows = filtered[filtered["route_label"] == route_label]
        for phase_label in REQUESTED_PHASES:
            group = route_rows[route_rows["phase_label"] == phase_label]
            if group.empty:
                continue
            flight_count = group[FLIGHT_ID_COLUMN].nunique()
            for feature in FEATURES_FOR_STATS:
                values = group[feature].dropna()
                row = {
                    "route_code": route_code,
                    "route": route_label,
                    "phase": phase_label,
                    "feature": feature,
                    "flight_count": flight_count,
                    "row_count": len(values),
                }
                row.update(describe_numeric_values(values))
                rows.append(row)

    columns = [
        "route_code",
        "route",
        "phase",
        "feature",
        "flight_count",
        "row_count",
    ] + STAT_COLUMNS
    return pd.DataFrame(rows, columns=columns)


def main() -> int:
    args = sys.argv[1:]
    output_path = DEFAULT_OUTPUT_PATH
    if "--output" in args:
        output_index = args.index("--output")
        try:
            output_path = Path(args[output_index + 1])
        except IndexError:
            print("ERROR: --output requires a path", file=sys.stderr)
            return 1
        del args[output_index : output_index + 2]

    if not args:
        print(
            "Usage: python3 2_statsandtypicalflight/flight_statistics.py "
            "<csv-file-or-directory> [...] "
            "[--output flight_statistics_summary.csv]",
            file=sys.stderr,
        )
        return 1

    csv_paths = resolve_csv_paths(args)
    if not csv_paths:
        print("No CSV files found.", file=sys.stderr)
        return 1

    frames = []
    errors = 0
    for csv_path in csv_paths:
        try:
            frames.append(load_csv(csv_path))
        except Exception as exc:
            errors += 1
            print(f"ERROR {csv_path}: {exc}", file=sys.stderr)

    if not frames:
        print("No valid CSV files found.", file=sys.stderr)
        return 1

    all_rows = pd.concat(frames, ignore_index=True)
    summary = build_summary(all_rows)
    summary.to_csv(output_path, index=False)

    print(f"Wrote {output_path}: {len(summary)} route-phase-feature rows")
    print(f"Flights read: {all_rows[FLIGHT_ID_COLUMN].nunique()}")
    print(f"Rows read: {len(all_rows)}")
    print(f"Errors: {errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
