from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_paths import resolve_dataset_path

INVALID_DATA_DIR = Path("dataset/data_XinvalidX")
REQUIRED_PHASES = {
    "noclb": {3, "climb"},
    "nocrz": {4, "cruise"},
    "nodsc": {5, "descent"},
}
ALTITUDE_COLUMN = "altitude_meters"
MAX_VALID_END_ALTITUDE_METERS = 1000


def phase_is_ground(value: str | None) -> bool:
    if value is None or value.strip() == "":
        return False

    phase = value.strip().lower()
    if phase == "ground":
        return True

    return float(phase) == 0


def phase_tokens(value: str | None) -> set[int | str]:
    if value is None or value.strip() == "":
        return set()

    phase = value.strip().lower()
    tokens: set[int | str] = {phase}
    try:
        tokens.add(int(float(phase)))
    except ValueError:
        pass
    return tokens


def row_altitude_meters(row: dict[str, str]) -> float:
    return float(row.get(ALTITUDE_COLUMN, ""))


def end_row_is_valid(row: dict[str, str]) -> bool:
    if phase_is_ground(row.get("phase")):
        return True

    return row_altitude_meters(row) < MAX_VALID_END_ALTITUDE_METERS


def invalid_reasons(path: Path) -> list[tuple[str, str]]:
    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        first_row = next(reader, None)
        last_row = first_row
        seen_phases = set()
        if first_row:
            seen_phases.update(phase_tokens(first_row.get("phase")))
        for row in reader:
            last_row = row
            seen_phases.update(phase_tokens(row.get("phase")))

    if not first_row:
        return [("nognd1", "file has no data rows")]

    reasons = []
    if not phase_is_ground(first_row.get("phase")):
        reasons.append(("nognd1", "start is not gnd"))
    try:
        valid_end_row = end_row_is_valid(last_row)
    except ValueError:
        valid_end_row = False
    if not valid_end_row:
        reasons.append(("earlyoff", "last row is above 1km"))

    for prefix, accepted_values in REQUIRED_PHASES.items():
        if seen_phases.isdisjoint(accepted_values):
            phase_name = next(
                value for value in accepted_values if isinstance(value, str)
            )
            reasons.append((prefix, f"missing {phase_name} phase"))

    return reasons


def prefixed_destination(path: Path, prefixes: list[str]) -> Path:
    prefix_text = "_".join(prefixes) + "_"
    destination = INVALID_DATA_DIR / f"{prefix_text}{path.name}"
    if not destination.exists():
        return destination

    stem = f"{prefix_text}{path.stem}"
    suffix = path.suffix
    counter = 1
    while True:
        destination = INVALID_DATA_DIR / f"{stem}_{counter}{suffix}"
        if not destination.exists():
            return destination
        counter += 1


def resolve_csv_paths(args: list[str]) -> list[Path]:
    csv_paths: list[Path] = []
    for arg in args:
        path = resolve_dataset_path(Path(arg))
        if path.is_dir():
            csv_paths.extend(sorted(path.glob("*.csv")))
        else:
            csv_paths.append(path)
    return csv_paths


def parse_args(args: list[str]) -> tuple[bool, list[str]]:
    scan = False
    paths = []
    for arg in args:
        if arg == "--scan":
            scan = True
        else:
            paths.append(arg)
    return scan, paths


def main() -> int:
    if len(sys.argv) == 1:
        print(
            "Usage: python3 1_preprocess/csv_3removemissingphases.py [--scan] "
            "<csv-file-or-directory> [...]",
            file=sys.stderr,
        )
        return 1

    scan, path_args = parse_args(sys.argv[1:])
    if not path_args:
        print("ERROR: provide at least one CSV file or directory.", file=sys.stderr)
        return 1

    csv_paths = resolve_csv_paths(path_args)
    if not csv_paths:
        print("No CSV files found.", file=sys.stderr)
        return 1

    if not scan:
        INVALID_DATA_DIR.mkdir(parents=True, exist_ok=True)

    moved = 0
    would_move = 0
    skipped = 0
    errors = 0

    for path in csv_paths:
        try:
            reasons = invalid_reasons(path)
        except (csv.Error, OSError, ValueError) as error:
            errors += 1
            print(f"ERROR {path}: {error}")
            continue

        if not reasons:
            skipped += 1
            continue

        prefixes = list(dict.fromkeys(prefix for prefix, _ in reasons))
        destination = prefixed_destination(path, prefixes)
        reason_text = "; ".join(reason for _, reason in reasons)
        if scan:
            would_move += 1
            print(f"WOULD MOVE {path} -> {destination} ({reason_text})")
        else:
            shutil.move(str(path), destination)
            moved += 1
            print(f"MOVED {path} -> {destination} ({reason_text})")

    if scan:
        print(
            "Summary <removemissingphases.py> [SCAN ONLY]. "
            f"\n Would move: {would_move} \n Skipped: {skipped} \n Errors: {errors}"
        )
    else:
        print(
            "Summary <removemissingphases.py>. "
            f"\n Moved: {moved} \n Skipped: {skipped} \n Errors: {errors}"
        )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
