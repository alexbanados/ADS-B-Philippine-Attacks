from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_paths import DATASET_DIR


FINAL_COLUMNS = [
    "sequence_index",
    "t_elapsed_sec",
    "dt",
    "t_norm",
    "latitude",
    "longitude",
    "altitude_meters",
    "speed_kmh",
    "verticalSpeed_ms",
    "heading",
    "x_wrt0",
    "y_wrt0",
    "delta_altitude",
    "delta_speed",
    "delta_heading",
    "turn_rate",
    "acceleration",
    "distance_per_timestep",
    "route_progress",
    "label",
]

FEATURE_COLUMNS = [column for column in FINAL_COLUMNS if column != "label"]

DATASETS = [
    {
        "tag": "auth",
        "label": 0,
        "output_suffix": "auth",
        "source_folders": ("data_nolvl_{route}", "beta_{route}_auth"),
    },
    {
        "tag": "modalt",
        "label": 1,
        "output_suffix": "alt",
        "source_folders": ("data_modalt_{route}", "beta_{route}_modalt"),
    },
    {
        "tag": "modpos",
        "label": 3,
        "output_suffix": "pos",
        "source_folders": ("data_modpos_{route}", "beta_{route}_modpos"),
    },
    {
        "tag": "modspd",
        "label": 2,
        "output_suffix": "spd",
        "source_folders": ("data_modspd_{route}", "beta_{route}_modspd"),
    },
]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build cleaned per-route sequence CSV folders from authentic and "
            "modified attack CSVs."
        )
    )
    parser.add_argument(
        "routes",
        nargs="+",
        help="Route suffixes to process, such as ceb dvo ilo pps mph.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DATASET_DIR,
        help="Directory containing data_nolvl_<route> and data_mod*_<route> folders. Default: dataset.",
    )
    parser.add_argument(
        "--ready-folder",
        type=Path,
        help=(
            "Manual output ready folder. Use this only with one route. "
            "If omitted, output defaults to <base-dir>/data_<route>_ready."
        ),
    )
    parser.add_argument(
        "--flight-id",
        help="Only clean CSVs whose filename contains this flight ID.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite cleaned CSVs if they already exist.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip missing source folders instead of failing.",
    )
    args = parser.parse_args(argv)
    if args.ready_folder is not None and len(args.routes) != 1:
        parser.error("--ready-folder can only be used with one route")
    return args


def safe_filename_part(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._-")
    return text or "unknown"


def source_folder_for_dataset(
    base_dir: Path,
    route: str,
    dataset: dict,
) -> Path | None:
    for pattern in dataset["source_folders"]:
        candidate = base_dir / pattern.format(route=route)
        if candidate.is_dir():
            return candidate
    return None


def flight_id_from_csv(df: pd.DataFrame, csv_path: Path) -> str:
    if "flight_id" in df.columns:
        flight_ids = df["flight_id"].dropna().astype(str).str.strip()
        flight_ids = flight_ids[flight_ids != ""].unique()
        if len(flight_ids) > 0:
            return safe_filename_part(flight_ids[0])
    return safe_filename_part(csv_path.stem)


def cleaned_dataframe(
    csv_path: Path,
    expected_label: int,
) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(csv_path)
    missing_features = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing_features:
        raise ValueError(f"missing required columns: {missing_features}")

    flight_id = flight_id_from_csv(df, csv_path)
    cleaned = df[FEATURE_COLUMNS].copy()
    cleaned["label"] = (
        pd.to_numeric(df["label"], errors="coerce").fillna(expected_label).astype(int)
        if "label" in df.columns
        else expected_label
    )
    return cleaned[FINAL_COLUMNS], flight_id


def unique_output_path(
    output_folder: Path,
    tag: str,
    route: str,
    flight_id: str,
    used_paths: set[Path],
) -> Path:
    base_name = f"{tag}_{route}_{flight_id}"
    output_path = output_folder / f"{base_name}.csv"
    counter = 2
    while output_path in used_paths:
        output_path = output_folder / f"{base_name}_{counter}.csv"
        counter += 1
    used_paths.add(output_path)
    return output_path


def clean_dataset(
    source_folder: Path,
    output_folder: Path,
    route: str,
    dataset: dict,
    overwrite: bool,
    flight_id_filter: str | None = None,
) -> tuple[int, int]:
    output_folder.mkdir(parents=True, exist_ok=True)
    csv_paths = sorted(source_folder.glob("*.csv"))
    if flight_id_filter:
        csv_paths = [path for path in csv_paths if flight_id_filter in path.stem]
    errors = 0
    written = 0
    used_paths: set[Path] = set()

    for csv_path in csv_paths:
        try:
            cleaned, flight_id = cleaned_dataframe(csv_path, dataset["label"])
            output_path = unique_output_path(
                output_folder,
                dataset["tag"],
                route,
                flight_id,
                used_paths,
            )
            if output_path.exists() and not overwrite:
                raise FileExistsError(
                    f"{output_path} already exists; rerun with --overwrite"
                )
            shutil.copy2(csv_path, output_path)
            cleaned.to_csv(output_path, index=False)
            written += 1
        except Exception as exc:
            errors += 1
            print(f"ERROR {csv_path}: {exc}", file=sys.stderr)

    return written, errors


def clean_route(
    base_dir: Path,
    route: str,
    overwrite: bool,
    allow_missing: bool,
    ready_folder_override: Path | None = None,
    flight_id_filter: str | None = None,
) -> tuple[int, int]:
    route = route.lower()
    ready_folder = ready_folder_override or base_dir / f"data_{route}_ready"
    total_written = 0
    total_errors = 0

    for dataset in DATASETS:
        source_folder = source_folder_for_dataset(base_dir, route, dataset)
        output_folder = ready_folder / f"{route}{dataset['output_suffix']}"

        if source_folder is None:
            message = (
                f"Missing source folder for {dataset['tag']} route {route}: "
                + " or ".join(
                    pattern.format(route=route)
                    for pattern in dataset["source_folders"]
                )
            )
            if allow_missing:
                print(f"SKIP: {message}")
                continue
            raise ValueError(message)

        written, errors = clean_dataset(
            source_folder=source_folder,
            output_folder=output_folder,
            route=route,
            dataset=dataset,
            overwrite=overwrite,
            flight_id_filter=flight_id_filter,
        )
        total_written += written
        total_errors += errors
        print(f"{source_folder} -> {output_folder}: {written} written, {errors} errors")

    return total_written, total_errors


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if not args.base_dir.is_dir():
        print(f"ERROR: base directory does not exist: {args.base_dir}", file=sys.stderr)
        return 1

    total_written = 0
    total_errors = 0
    for route in args.routes:
        try:
            written, errors = clean_route(
                base_dir=args.base_dir,
                route=route,
                overwrite=args.overwrite,
                allow_missing=args.allow_missing,
                ready_folder_override=args.ready_folder,
                flight_id_filter=args.flight_id,
            )
            total_written += written
            total_errors += errors
        except Exception as exc:
            total_errors += 1
            print(f"ERROR route {route}: {exc}", file=sys.stderr)

    print(f"Cleaned CSVs written: {total_written}")
    print(f"Errors: {total_errors}")
    return 1 if total_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
