import re
import shutil
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_paths import resolve_dataset_path

PH_TIMEZONE = ZoneInfo("Asia/Manila")
INVALID_DATA_DIR = Path("dataset/data_XinvalidX")

METADATA_COLUMNS = [
    "aircraft_type",
    "flight_id",
]

OUTPUT_COLUMNS = [
    "timestamp",
    "ph_time",
    "latitude",
    "longitude",
    "altitude_meters",
    "speed_kmh",
    "verticalSpeed_ms",
    "heading",
]


def invalid_destination(csv_path, prefix):
    """Return a collision-safe invalid-data path with a reason prefix."""
    destination = INVALID_DATA_DIR / f"{prefix}_{csv_path.name}"
    if not destination.exists():
        return destination

    stem = f"{prefix}_{csv_path.stem}"
    suffix = csv_path.suffix
    counter = 1
    while True:
        destination = INVALID_DATA_DIR / f"{stem}_{counter}{suffix}"
        if not destination.exists():
            return destination
        counter += 1


def move_if_non_airbus(csv_path):
    """Move files whose filename aircraft type does not start with A."""
    metadata = flight_metadata(csv_path)
    if metadata["aircraft_type"].upper().startswith("A"):
        return None

    INVALID_DATA_DIR.mkdir(parents=True, exist_ok=True)
    destination = invalid_destination(csv_path, "notairbus")
    shutil.move(str(csv_path), destination)
    return destination


def renamed_flight_path(csv_path):
    """
    Convert names like 3ec8c787_A21N_RPLL_to_RPVM.csv
    to A21N_3ec8c787_RPLL_to_RPVM.csv.
    """
    first_token = csv_path.stem.split("_", 1)[0]
    if not first_token[:1].isdigit():
        return csv_path

    match = re.fullmatch(
        r"(?P<flight_id>[^_]+)_(?P<aircraft>[^_]+)_(?P<route>.+)\.csv",
        csv_path.name,
    )
    if not match:
        return csv_path

    new_name = (
        f"{match.group('aircraft')}_"
        f"{match.group('flight_id')}_"
        f"{match.group('route')}.csv"
    )
    return csv_path.with_name(new_name)


def flight_metadata(csv_path):
    """Return aircraft type and flight ID from the CSV name."""
    canonical_path = renamed_flight_path(csv_path)
    match = re.fullmatch(
        r"(?P<aircraft>[^_]+)_(?P<flight_id>[^_]+)_(?P<route>.+)\.csv",
        canonical_path.name,
    )
    if not match:
        raise ValueError(f"cannot parse flight metadata from filename: {csv_path.name}")

    return {
        "aircraft_type": match.group("aircraft"),
        "flight_id": match.group("flight_id"),
    }


def add_ph_time(df):
    """Add Manila-local time with millisecond detail from the raw Unix timestamp."""
    ph_time = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    ph_time = ph_time.dt.tz_convert(PH_TIMEZONE)
    df["ph_time"] = ph_time.dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3] + " PHT"
    return df


def process_csv(csv_path):
    """Rewrite one CSV with SI-unit columns only, then rename the file."""
    invalid_path = move_if_non_airbus(csv_path)
    if invalid_path is not None:
        return invalid_path

    df = pd.read_csv(csv_path)

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df = add_ph_time(df)

    missing = [column for column in OUTPUT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}")

    new_path = renamed_flight_path(csv_path)
    metadata = flight_metadata(csv_path)
    for column, value in metadata.items():
        df[column] = value

    df = df[METADATA_COLUMNS + OUTPUT_COLUMNS]

    temp_path = new_path.with_suffix(".tmp")
    df.to_csv(temp_path, index=False)
    temp_path.replace(new_path)

    if new_path != csv_path:
        csv_path.unlink()

    return new_path


def resolve_csv_paths(args):
    csv_paths = []
    for arg in args:
        path = resolve_dataset_path(Path(arg))
        if path.is_dir():
            csv_paths.extend(sorted(path.glob("*.csv")))
        else:
            csv_paths.append(path)
    return csv_paths


def main():
    """Process requested CSV files or directories."""
    if len(sys.argv) == 1:
        print(
            "Usage: python3 1_preprocess/csv_1preprocessing.py <csv-file-or-directory> [...]",
            file=sys.stderr,
        )
        return 1

    csv_files = resolve_csv_paths(sys.argv[1:])
    if not csv_files:
        print("No CSV files found.", file=sys.stderr)
        return 1

    processed = 0
    moved_non_airbus = 0
    errors = 0
    for csv_path in csv_files:
        try:
            new_path = process_csv(csv_path)
            if new_path.parent == INVALID_DATA_DIR:
                moved_non_airbus += 1
                print(f"Moved non-Airbus {csv_path.name} -> {new_path}")
            else:
                processed += 1
                print(f"Processed {csv_path.name} -> {new_path.name}")
        except Exception as exc:
            errors += 1
            print(f"Error processing {csv_path.name}: {exc}", file=sys.stderr)

    print("Summary <preprocessing.py>:")
    print(f"  processed: {processed}")
    print(f"  moved non-Airbus CSV files to {INVALID_DATA_DIR}: {moved_non_airbus}")
    print(f"  files with processing errors: {errors}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
