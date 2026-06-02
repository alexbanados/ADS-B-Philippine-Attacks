import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_paths import resolve_dataset_path
from flight_segmentation import load_flight
from flight_segmentation import phase_summary
from flight_segmentation import segment_flight_phases


INTERNAL_COLUMNS = {"Time", "Time_local"}
SEGMENTATION_COLUMNS = {"phase", "is_level"}


def process_csv(csv_path):
    """Rewrite one CSV with a final numeric flight phase column."""
    df = load_flight(csv_path)
    output_columns = [
        column
        for column in df.columns
        if column not in INTERNAL_COLUMNS and column not in SEGMENTATION_COLUMNS
    ]

    existing_segmentation_columns = [
        column for column in SEGMENTATION_COLUMNS if column in df.columns
    ]
    if existing_segmentation_columns:
        df = df.drop(columns=existing_segmentation_columns)

    df = segment_flight_phases(df)
    phase = df.pop("phase")
    is_level = df.pop("is_level")
    df = df[[column for column in output_columns if column in df.columns]]
    df["phase"] = phase
    df["is_level"] = is_level

    temp_path = csv_path.with_suffix(".tmp")
    df.to_csv(temp_path, index=False)
    temp_path.replace(csv_path)

    return df


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
            "Usage: python3 1_preprocess/csv_2segment.py <csv-file-or-directory> [...]",
            file=sys.stderr,
        )
        return 1

    csv_files = resolve_csv_paths(sys.argv[1:])
    if not csv_files:
        print("No CSV files found.", file=sys.stderr)
        return 1

    processed = 0
    errors = 0
    for csv_path in csv_files:
        try:
            df = process_csv(csv_path)
            processed += 1
            print(f"Processed {csv_path.name}: {phase_summary(df)}")
        except Exception as exc:
            errors += 1
            print(f"Error processing {csv_path.name}: {exc}", file=sys.stderr)

    print("Summary <segment.py>:")
    print(f"  processed: {processed}")
    print(f"  errors: {errors}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
