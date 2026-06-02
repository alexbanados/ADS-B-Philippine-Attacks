import csv
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_paths import resolve_dataset_path

INVALID_DATA_DIR = Path("dataset/data_XinvalidX")
RADIUS_KM = 10
ROUTE_RADII_KM = {3: 50, 4: 25}
EARTH_RADIUS_KM = 6371.0088
LATITUDE_COLUMN = "latitude"
LONGITUDE_COLUMN = "longitude"
ROUTE_COLUMN = "route"
AIRCRAFT_ID_COLUMN = "flight_id"
START_POINT = (14.513627, 121.014090)

ROUTE_POINTS = {
    1: (10.309000, 123.979000),
    2: (7.125633, 125.645990),
    3: (10.832583, 122.493237),
    4: (11.926941, 121.958967),
    5: (9.742127, 118.758210),
}
ROUTE_CODES = {
    "ceb": 1,
    "dvo": 2,
    "ilo": 3,
    "mph": 4,
    "pps": 5,
}
ROUTE_NAMES = {code: name for name, code in ROUTE_CODES.items()}


def path_tokens(path):
    """Return lowercase route-name tokens found in a path's names."""
    tokens = []
    for part in [path.stem, path.parent.name, *(parent.name for parent in path.parents)]:
        for token in part.lower().replace("-", "_").split("_"):
            if token:
                tokens.append(token)
    return tokens


def expected_route_from_path(csv_path):
    """Return the route code implied by folder/file names, if present."""
    for token in path_tokens(csv_path):
        route = ROUTE_CODES.get(token)
        if route is not None:
            return route
    return None


def route_folder_code(csv_path):
    """Return the data folder suffix used in bad-route invalid filenames."""
    for parent in [csv_path.parent, *csv_path.parents]:
        if parent.name.startswith("data_") and parent.name != INVALID_DATA_DIR.name:
            return parent.name.removeprefix("data_")
    return csv_path.parent.name


def invalid_destination(csv_path):
    """Return a collision-safe invalid-data path for a bad-route CSV."""
    folder_code = route_folder_code(csv_path)
    destination = INVALID_DATA_DIR / f"badroute_{folder_code}_{csv_path.name}"
    if not destination.exists():
        return destination

    stem = f"badroute_{folder_code}_{csv_path.stem}"
    suffix = csv_path.suffix
    counter = 1
    while True:
        destination = INVALID_DATA_DIR / f"{stem}_{counter}{suffix}"
        if not destination.exists():
            return destination
        counter += 1


def haversine_km(lat1, lon1, lat2, lon2):
    """Return great-circle distance between two coordinates in kilometers."""
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def classify_route(latitude, longitude):
    """Return the route code for the destination coordinate, or 0 if unmatched."""
    for route, (route_latitude, route_longitude) in ROUTE_POINTS.items():
        distance = haversine_km(latitude, longitude, route_latitude, route_longitude)
        radius = ROUTE_RADII_KM.get(route, RADIUS_KM)
        if distance <= radius:
            return route
    return 0


def is_valid_start(latitude, longitude):
    """Return whether the origin coordinate is close enough to the start point."""
    start_latitude, start_longitude = START_POINT
    distance = haversine_km(latitude, longitude, start_latitude, start_longitude)
    return distance <= RADIUS_KM


def route_fieldnames(fieldnames):
    """Return CSV fieldnames with route placed after the aircraft ID column."""
    if fieldnames is None:
        raise ValueError("missing CSV header")

    output_fields = [field for field in fieldnames if field != ROUTE_COLUMN]
    if AIRCRAFT_ID_COLUMN not in output_fields:
        raise ValueError(f"missing column {AIRCRAFT_ID_COLUMN}")

    insert_index = output_fields.index(AIRCRAFT_ID_COLUMN) + 1
    output_fields.insert(insert_index, ROUTE_COLUMN)
    return output_fields


def process_csv(csv_path):
    """Classify one CSV by its first and final coordinates, then rewrite it."""
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        fieldnames = route_fieldnames(reader.fieldnames)

    if not rows:
        raise ValueError("empty CSV")

    first_row = rows[0]
    last_row = rows[-1]
    try:
        start_latitude = float(first_row[LATITUDE_COLUMN])
        start_longitude = float(first_row[LONGITUDE_COLUMN])
        end_latitude = float(last_row[LATITUDE_COLUMN])
        end_longitude = float(last_row[LONGITUDE_COLUMN])
    except KeyError as exc:
        raise ValueError(f"missing column {exc.args[0]}") from exc
    except ValueError as exc:
        raise ValueError("first or last row has invalid latitude or longitude") from exc

    route = 0
    if is_valid_start(start_latitude, start_longitude):
        route = classify_route(end_latitude, end_longitude)

    for row in rows:
        row[ROUTE_COLUMN] = route

    temp_path = csv_path.with_suffix(".tmp")
    with temp_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(csv_path)

    return route


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
            "Usage: python3 1_preprocess/csv_5routeclassifier.py <csv-file-or-directory> [...]",
            file=sys.stderr,
        )
        return 1

    csv_paths = resolve_csv_paths(sys.argv[1:])
    if not csv_paths:
        print("No CSV files found.", file=sys.stderr)
        return 1

    INVALID_DATA_DIR.mkdir(parents=True, exist_ok=True)

    errors = 0
    route_counts = defaultdict(int)
    unmatched_paths = []
    mismatch_paths = []
    for csv_path in csv_paths:
        expected_route = expected_route_from_path(csv_path)
        try:
            route = process_csv(csv_path)
        except Exception as exc:
            errors += 1
            print(f"ERROR {csv_path}: {exc}", file=sys.stderr)
            continue

        route_counts[route] += 1
        print(f"Processed {csv_path}: route={route}")
        if route == 0:
            destination = invalid_destination(csv_path)
            shutil.move(str(csv_path), destination)
            unmatched_paths.append(destination)
            print(f"Moved bad route {csv_path} -> {destination}")
        elif expected_route is not None and route != expected_route:
            destination = invalid_destination(csv_path)
            shutil.move(str(csv_path), destination)
            mismatch_paths.append((destination, expected_route, route))
            print(
                f"Moved route mismatch {csv_path} -> {destination} "
                f"expected={expected_route}({ROUTE_NAMES.get(expected_route, 'unknown')}) "
                f"classified={route}({ROUTE_NAMES.get(route, 'unknown')})"
            )

    print("Route report:")
    print(f"0: {route_counts[0]}")
    for unmatched_path in unmatched_paths:
        print(f"  {unmatched_path}")
    for route in range(1, 6):
        print(f"{route}: {route_counts[route]}")
        for mismatch_path, expected_route, actual_route in mismatch_paths:
            if expected_route == route:
                expected_name = ROUTE_NAMES.get(expected_route, "unknown")
                actual_name = ROUTE_NAMES.get(actual_route, "unknown")
                print(
                    f"  mismatch: {mismatch_path} "
                    f"expected={expected_route}({expected_name}) "
                    f"classified={actual_route}({actual_name})"
                )

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
