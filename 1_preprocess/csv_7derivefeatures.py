from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_paths import resolve_dataset_path


EARTH_RADIUS_M = 6_371_008.8
LATITUDE_COLUMN = "latitude"
LONGITUDE_COLUMN = "longitude"
TIMESTAMP_COLUMN = "timestamp"
ALTITUDE_COLUMN = "altitude_meters"
SPEED_COLUMN = "speed_kmh"
HEADING_COLUMN = "heading"
PHASE_COLUMN = "phase"
PH_TIME_COLUMN = "ph_time"

TIME_FEATURE_COLUMNS = [
    "sequence_index",
    "t_elapsed_sec",
    "dt",
    "t_norm",
]

FEATURE_COLUMNS = [
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
ROUTE_PROGRESS_COLUMN = "route_progress"
DERIVED_COLUMNS = TIME_FEATURE_COLUMNS + FEATURE_COLUMNS

REQUIRED_COLUMNS = [
    TIMESTAMP_COLUMN,
    LATITUDE_COLUMN,
    LONGITUDE_COLUMN,
    ALTITUDE_COLUMN,
    SPEED_COLUMN,
    HEADING_COLUMN,
]


def haversine_m(latitude: pd.Series, longitude: pd.Series) -> pd.Series:
    """Return per-row ground distance from the previous coordinate in meters."""
    lat1 = np.radians(latitude.shift())
    lon1 = np.radians(longitude.shift())
    lat2 = np.radians(latitude)
    lon2 = np.radians(longitude)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    distance = 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))
    return pd.Series(distance, index=latitude.index).fillna(0)


def relative_xy_m(latitude: pd.Series, longitude: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return east/north meter offsets from the first coordinate."""
    if latitude.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    start_lat = np.radians(latitude.iloc[0])
    start_lon = np.radians(longitude.iloc[0])
    lat = np.radians(latitude)
    lon = np.radians(longitude)

    x = (lon - start_lon) * np.cos((lat + start_lat) / 2) * EARTH_RADIUS_M
    y = (lat - start_lat) * EARTH_RADIUS_M
    return pd.Series(x, index=latitude.index), pd.Series(y, index=latitude.index)


def output_columns(columns: list[str]) -> list[str]:
    """Return columns with time and movement features in their expected positions."""
    fields = [column for column in columns if column not in DERIVED_COLUMNS]
    if PH_TIME_COLUMN in fields:
        time_insert_at = fields.index(PH_TIME_COLUMN) + 1
    elif TIMESTAMP_COLUMN in fields:
        time_insert_at = fields.index(TIMESTAMP_COLUMN) + 1
    else:
        time_insert_at = len(fields)
    fields = (
        fields[:time_insert_at]
        + TIME_FEATURE_COLUMNS
        + fields[time_insert_at:]
    )
    insert_at = fields.index(PHASE_COLUMN) + 1 if PHASE_COLUMN in fields else len(fields)
    return fields[:insert_at] + FEATURE_COLUMNS + fields[insert_at:]


def add_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Fill derived flight-statistic columns from timestamp and movement data."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}")

    df = df.copy()
    timestamp = pd.to_numeric(df[TIMESTAMP_COLUMN], errors="coerce")
    latitude = pd.to_numeric(df[LATITUDE_COLUMN], errors="coerce")
    longitude = pd.to_numeric(df[LONGITUDE_COLUMN], errors="coerce")
    altitude = pd.to_numeric(df[ALTITUDE_COLUMN], errors="coerce")
    speed = pd.to_numeric(df[SPEED_COLUMN], errors="coerce")
    heading = pd.to_numeric(df[HEADING_COLUMN], errors="coerce")

    dt = timestamp.diff().fillna(0)
    dt_safe = dt.mask(dt == 0, np.nan)

    df["sequence_index"] = range(len(df))
    if df.empty:
        df["t_elapsed_sec"] = pd.Series(dtype=float)
    else:
        df["t_elapsed_sec"] = timestamp - timestamp.iloc[0]
    df["dt"] = dt

    total_duration = df["t_elapsed_sec"].iloc[-1] if not df.empty else 0
    if total_duration == 0 or pd.isna(total_duration):
        df["t_norm"] = 0
    else:
        df["t_norm"] = df["t_elapsed_sec"] / total_duration

    df["delta_altitude"] = altitude.diff().fillna(0)
    df["delta_speed"] = speed.diff().fillna(0)
    df["delta_heading"] = ((heading.diff() + 180) % 360 - 180).fillna(0)
    df["turn_rate"] = (df["delta_heading"] / dt_safe).fillna(0)
    df["acceleration"] = (df["delta_speed"] / dt_safe).fillna(0)
    df["distance_per_timestep"] = haversine_m(latitude, longitude)

    cumulative_distance = df["distance_per_timestep"].cumsum()
    total_distance = cumulative_distance.iloc[-1] if not cumulative_distance.empty else 0
    if total_distance > 0:
        df["route_progress"] = cumulative_distance / total_distance
    else:
        df["route_progress"] = 0

    df["x_wrt0"], df["y_wrt0"] = relative_xy_m(latitude, longitude)

    return df[output_columns(list(df.columns))]


def csv_row_number(index) -> int:
    """Return the 1-based CSV file row number, counting the header as row 1."""
    return int(index) + 2


def numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return selected columns coerced to numeric values."""
    return df[columns].apply(pd.to_numeric, errors="coerce")


def input_validation_issues(df: pd.DataFrame) -> list[tuple[int, str]]:
    """Return row-numbered issues in raw numeric input columns."""
    missing_required = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_required:
        raise ValueError(f"missing columns {missing_required}")

    issues = []
    raw_values = df[REQUIRED_COLUMNS]
    numeric_values = numeric_columns(df, REQUIRED_COLUMNS)

    missing_mask = raw_values.isna() | raw_values.astype(str).apply(
        lambda column: column.str.strip().eq("")
    )
    for index, columns in missing_mask[missing_mask.any(axis=1)].iterrows():
        missing_columns = ", ".join(columns[columns].index)
        issues.append((csv_row_number(index), f"missing input: {missing_columns}"))

    non_numeric_mask = numeric_values.isna() & ~missing_mask
    for index, columns in non_numeric_mask[non_numeric_mask.any(axis=1)].iterrows():
        non_numeric_columns = ", ".join(columns[columns].index)
        issues.append(
            (csv_row_number(index), f"non-numeric input: {non_numeric_columns}")
        )

    infinite_mask = np.isinf(numeric_values.to_numpy(dtype=float))
    infinite_df = pd.DataFrame(
        infinite_mask,
        index=numeric_values.index,
        columns=numeric_values.columns,
    )
    for index, columns in infinite_df[infinite_df.any(axis=1)].iterrows():
        infinite_columns = ", ".join(columns[columns].index)
        issues.append((csv_row_number(index), f"infinite input: {infinite_columns}"))

    timestamp = numeric_values[TIMESTAMP_COLUMN]
    dt = timestamp.diff()
    duplicate_timestamps = dt.eq(0)
    duplicate_timestamps.iloc[:1] = False
    for index in duplicate_timestamps[duplicate_timestamps].index:
        issues.append((csv_row_number(index), "duplicate timestamp"))

    negative_timesteps = dt.lt(0)
    for index, value in dt[negative_timesteps].items():
        issues.append((csv_row_number(index), f"negative timestep: {value}"))

    return issues


def validation_issues(df: pd.DataFrame) -> list[tuple[int, str]]:
    """Return row-numbered issues in derived feature columns."""
    issues = []
    feature_values = numeric_columns(df, DERIVED_COLUMNS)

    missing_mask = feature_values.isna()
    for index, columns in missing_mask[missing_mask.any(axis=1)].iterrows():
        missing_columns = ", ".join(columns[columns].index)
        issues.append((csv_row_number(index), f"missing: {missing_columns}"))

    infinite_mask = np.isinf(feature_values.to_numpy(dtype=float))
    infinite_df = pd.DataFrame(
        infinite_mask,
        index=feature_values.index,
        columns=feature_values.columns,
    )
    for index, columns in infinite_df[infinite_df.any(axis=1)].iterrows():
        infinite_columns = ", ".join(columns[columns].index)
        issues.append((csv_row_number(index), f"infinite: {infinite_columns}"))

    route_progress = feature_values[ROUTE_PROGRESS_COLUMN]
    out_of_range = route_progress.notna() & (
        (route_progress < 0) | (route_progress > 1)
    )
    for index, value in route_progress[out_of_range].items():
        issues.append(
            (
                csv_row_number(index),
                f"{ROUTE_PROGRESS_COLUMN} out of range: {value}",
            )
        )

    return issues


def process_csv(csv_path: Path) -> tuple[int, list[tuple[int, str]]]:
    df = pd.read_csv(csv_path)
    input_issues = input_validation_issues(df)
    df = add_statistics(df)
    issues = input_issues + validation_issues(df)

    temp_path = csv_path.with_suffix(".tmp")
    df.to_csv(temp_path, index=False)
    temp_path.replace(csv_path)
    return len(df), issues


def resolve_csv_paths(args: list[str]) -> list[Path]:
    csv_paths: list[Path] = []
    for arg in args:
        path = resolve_dataset_path(Path(arg))
        if path.is_dir():
            csv_paths.extend(sorted(path.glob("*.csv")))
        else:
            csv_paths.append(path)
    return csv_paths


def main() -> int:
    if len(sys.argv) == 1:
        print(
            "Usage: python3 1_preprocess/csv_7derivefeatures.py <csv-file-or-directory> [...]",
            file=sys.stderr,
        )
        return 1

    csv_paths = resolve_csv_paths(sys.argv[1:])
    if not csv_paths:
        print("No CSV files found.", file=sys.stderr)
        return 1

    processed = 0
    total_rows = 0
    total_validation_issues = 0
    errors = 0
    for csv_path in csv_paths:
        try:
            rows, issues = process_csv(csv_path)
        except Exception as exc:
            errors += 1
            print(f"ERROR {csv_path}: {exc}", file=sys.stderr)
            continue

        processed += 1
        total_rows += rows
        total_validation_issues += len(issues)
        print(f"Processed {csv_path}: {rows} rows")
        for row_number, issue in issues:
            print(f"  VALIDATION {csv_path}: row {row_number}: {issue}")

    print("Summary:")
    print(f"  processed: {processed}")
    print(f"  rows updated: {total_rows}")
    print(f"  validation issues: {total_validation_issues}")
    print(f"  errors: {errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
