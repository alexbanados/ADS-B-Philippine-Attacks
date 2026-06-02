from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_paths import resolve_dataset_path


LEVEL_COLUMN = "is_level"
PHASE_COLUMN = "phase"
PH_TIME_COLUMN = "ph_time"

BLANK_TIME_COLUMNS = [
    "sequence_index",
    "t_elapsed_sec",
    "dt",
    "t_norm",
]

BLANK_MOVEMENT_COLUMNS = [
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
BLANK_FEATURE_COLUMNS = BLANK_TIME_COLUMNS + BLANK_MOVEMENT_COLUMNS
BLANK_FEATURE_COLUMNS_TEXT = ", ".join(BLANK_FEATURE_COLUMNS)


def nolvl_directory(source_dir: Path) -> Path:
    if not source_dir.is_dir():
        raise ValueError(f"{source_dir} is not a directory")
    if not source_dir.name.startswith("data_"):
        raise ValueError(f"{source_dir} does not use the data_xxx folder pattern")
    if source_dir.name.startswith("data_nolvl_"):
        raise ValueError(f"{source_dir} is already a data_nolvl_xxx folder")

    suffix = source_dir.name.removeprefix("data_")
    return source_dir.with_name(f"data_nolvl_{suffix}")


def output_fieldnames(fieldnames: list[str]) -> list[str]:
    fields = [
        field
        for field in fieldnames
        if field != LEVEL_COLUMN and field not in BLANK_FEATURE_COLUMNS
    ]
    if PH_TIME_COLUMN in fields:
        time_insert_at = fields.index(PH_TIME_COLUMN) + 1
    elif "timestamp" in fields:
        time_insert_at = fields.index("timestamp") + 1
    else:
        time_insert_at = len(fields)
    fields = (
        fields[:time_insert_at]
        + BLANK_TIME_COLUMNS
        + fields[time_insert_at:]
    )
    insert_at = fields.index(PHASE_COLUMN) + 1 if PHASE_COLUMN in fields else len(fields)
    return fields[:insert_at] + BLANK_MOVEMENT_COLUMNS + fields[insert_at:]


def remove_level_column(csv_path: Path) -> tuple[bool, bool]:
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("missing CSV header")
        rows = list(reader)

    appended_columns = any(column not in fieldnames for column in BLANK_FEATURE_COLUMNS)
    output_fields = output_fieldnames(fieldnames)
    temp_path = csv_path.with_suffix(".tmp")
    with temp_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=output_fields)
        writer.writeheader()
        for row in rows:
            output_row = {field: row.get(field, "") for field in output_fields}
            for column in BLANK_FEATURE_COLUMNS:
                output_row[column] = ""
            writer.writerow(output_row)
    temp_path.replace(csv_path)
    return LEVEL_COLUMN in fieldnames, appended_columns


def copy_without_level_column(
    source_dir: Path,
    destination_dir: Path | None = None,
) -> tuple[Path, int, int, int]:
    if source_dir.is_file():
        if source_dir.suffix.lower() != ".csv":
            raise ValueError(f"{source_dir} is not a CSV file")
        if destination_dir is None:
            destination_dir = nolvl_directory(source_dir.parent)
        destination_dir.mkdir(exist_ok=True)
        destination_path = destination_dir / source_dir.name
        shutil.copy2(source_dir, destination_path)
        removed_level, appended_columns = remove_level_column(destination_path)
        return destination_dir, 1, int(removed_level), int(appended_columns)

    if not source_dir.is_dir():
        raise ValueError(f"{source_dir} is not a directory or CSV file")
    if destination_dir is None:
        destination_dir = nolvl_directory(source_dir)
    if source_dir.resolve() == destination_dir.resolve():
        raise ValueError("output folder must be different from input folder")

    destination_dir.mkdir(exist_ok=True)

    copied = 0
    removed = 0
    appended = 0
    for source_path in sorted(source_dir.glob("*.csv")):
        destination_path = destination_dir / source_path.name
        shutil.copy2(source_path, destination_path)
        copied += 1
        removed_level, appended_columns = remove_level_column(destination_path)
        if removed_level:
            removed += 1
        if appended_columns:
            appended += 1

    return destination_dir, copied, removed, appended


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy route CSVs, remove the is_level column, and append blank "
            "feature columns for the next preprocessing step."
        )
    )
    parser.add_argument(
        "data_folders",
        nargs="+",
        type=Path,
        help="Input data folder(s) or CSV file(s), such as dataset/data_ceb.",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        help=(
            "Manual output folder. Use this only with one input folder. "
            "If omitted, output defaults to data_nolvl_<suffix>."
        ),
    )
    args = parser.parse_args(argv)
    if args.output_folder is not None and len(args.data_folders) != 1:
        parser.error("--output-folder can only be used with one input folder")
    args.data_folders = [resolve_dataset_path(path) for path in args.data_folders]
    return args


def main() -> int:
    args = parse_args(sys.argv[1:])

    errors = 0
    for source_dir in args.data_folders:
        destination_override = args.output_folder if len(args.data_folders) == 1 else None
        try:
            destination_dir, copied, removed, appended = copy_without_level_column(
                source_dir,
                destination_override,
            )
        except Exception as exc:
            errors += 1
            print(f"ERROR {source_dir}: {exc}", file=sys.stderr)
            continue

        print(f"Processed {source_dir} -> {destination_dir}")
        print(f"  copied CSVs: {copied}")
        print(f"  removed {LEVEL_COLUMN!r}: {removed} files")
        print(f"  appended columns: {BLANK_FEATURE_COLUMNS_TEXT}")
        print(f"  appended to: {appended} files")
        print(f"  errors: {errors}")

    print("Done.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
