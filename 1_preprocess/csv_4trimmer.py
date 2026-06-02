from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_paths import resolve_dataset_path


PHASE_COLUMN = "phase"
GROUND = 0
DEPARTURE_PHASES = (1, 2, 3)  # Takeoff, Initial Climb, Climb.
ARRIVAL_PHASE_PRIORITY = (
    (6,),      # Approach.
    (5,),     # Descent fallback when Approach is missing.
)


PHASE_NAMES = {
    "ground": 0,
    "takeoff": 1,
    "initial climb": 2,
    "climb": 3,
    "cruise": 4,
    "descent": 5,
    "approach": 6,
}


def parse_phase(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return None

    phase = value.strip()
    named_phase = PHASE_NAMES.get(phase.lower())
    if named_phase is not None:
        return named_phase

    return int(float(phase))


def resolve_csv_paths(args: list[str]) -> list[Path]:
    csv_paths: list[Path] = []
    for arg in args:
        path = resolve_dataset_path(Path(arg))
        if path.is_dir():
            csv_paths.extend(sorted(path.glob("*.csv")))
        else:
            csv_paths.append(path)
    return csv_paths


def first_index(phases: list[int | None], targets: tuple[int, ...]) -> int | None:
    for index, phase in enumerate(phases):
        if phase in targets:
            return index
    return None


def last_index(phases: list[int | None], targets: tuple[int, ...]) -> int | None:
    for index in range(len(phases) - 1, -1, -1):
        if phases[index] in targets:
            return index
    return None


def choose_start_ground(phases: list[int | None]) -> int | None:
    first_departure = first_index(phases, DEPARTURE_PHASES)
    ground_indexes = [
        index
        for index, phase in enumerate(phases)
        if phase == GROUND and (first_departure is None or index < first_departure)
    ]
    if ground_indexes:
        return ground_indexes[-1]
    return first_index(phases, (GROUND,))


def choose_end_ground(phases: list[int | None]) -> int | None:
    last_arrival = None
    for targets in ARRIVAL_PHASE_PRIORITY:
        last_arrival = last_index(phases, targets)
        if last_arrival is not None:
            break

    ground_indexes = [
        index
        for index, phase in enumerate(phases)
        if phase == GROUND and (last_arrival is None or index > last_arrival)
    ]
    if ground_indexes:
        return ground_indexes[0]
    return last_index(phases, (GROUND,))


def trim_csv(csv_path: Path) -> tuple[int, int, int]:
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames
        if not fieldnames or PHASE_COLUMN not in fieldnames:
            raise ValueError(f"missing {PHASE_COLUMN!r} column")
        rows = list(reader)

    phases = [parse_phase(row.get(PHASE_COLUMN)) for row in rows]
    start_ground = choose_start_ground(phases)
    end_ground = choose_end_ground(phases)
    keep_ground_indexes = {
        index for index in (start_ground, end_ground) if index is not None
    }

    trimmed_rows = [
        row
        for index, row in enumerate(rows)
        if phases[index] != GROUND or index in keep_ground_indexes
    ]

    temp_path = csv_path.with_suffix(".tmp")
    with temp_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trimmed_rows)
    temp_path.replace(csv_path)

    return len(rows), len(trimmed_rows), len(rows) - len(trimmed_rows)


def main() -> int:
    if len(sys.argv) == 1:
        print(
            "Usage: python3 1_preprocess/csv_4trimmer.py <csv-file-or-directory> [...]",
            file=sys.stderr,
        )
        return 1

    csv_paths = resolve_csv_paths(sys.argv[1:])
    if not csv_paths:
        print("No CSV files found.", file=sys.stderr)
        return 1

    total_removed = 0
    errors = 0
    for csv_path in csv_paths:
        try:
            before, after, removed = trim_csv(csv_path)
        except Exception as exc:
            errors += 1
            print(f"ERROR {csv_path}: {exc}", file=sys.stderr)
            continue

        total_removed += removed
        print(f"Trimmed {csv_path}: {before} -> {after} rows ({removed} removed)")

    print(f"Summary <trimmer.py>: \n Files: {len(csv_paths)} \n Removed rows: {total_removed} \n Errors: {errors}.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
