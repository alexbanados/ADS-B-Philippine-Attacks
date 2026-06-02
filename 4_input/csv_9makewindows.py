from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_paths import dataset_path


FEATURE_COLUMNS = [
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
]
TARGET_COLUMN = "label"
LABEL_NAMES = {
    0: "authentic",
    1: "modified_altitude",
    2: "modified_speed",
    3: "modified_position",
}
DATASET_SUFFIXES = ("auth", "alt", "pos", "spd")


@dataclass(frozen=True)
class CsvRecord:
    path: Path
    flight_id: str
    file_label: int


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert cleaned per-flight CSVs from dataset/data_<route>_ready into "
            "scaled train/validation/test sequence windows."
        )
    )
    parser.add_argument(
        "route",
        help="Route suffix to process, such as ceb, dvo, ilo, pps, or mph.",
    )
    parser.add_argument(
        "--ready-folder",
        type=Path,
        help="Cleaned ready folder. Defaults to dataset/data_<route>_ready.",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        help="Folder where window arrays will be written. Defaults to dataset/data_<route>_seq.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=128,
        help="Number of consecutive rows per sequence window. Default: 128.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=16,
        help="Row step between neighboring windows. Default: 16.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Fraction of flight IDs assigned to train. Default: 0.70.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of flight IDs assigned to validation. Default: 0.15.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Fraction of flight IDs assigned to test. Default: 0.15.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting flight IDs. Default: 42.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output folder files if they already exist.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.window_size <= 0:
        raise ValueError("--window-size must be greater than 0")
    if args.stride <= 0:
        raise ValueError("--stride must be greater than 0")
    ratios = [args.train_ratio, args.val_ratio, args.test_ratio]
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("--train-ratio, --val-ratio, and --test-ratio must be > 0")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("--train-ratio + --val-ratio + --test-ratio must equal 1.0")


def parse_cleaned_filename(path: Path, route: str) -> tuple[str, str]:
    stem = path.stem
    prefix_parts = stem.split("_", 2)
    if len(prefix_parts) != 3:
        raise ValueError(f"filename does not match <tag>_<route>_<flight_id>.csv: {path.name}")

    tag, file_route, flight_id = prefix_parts
    if file_route != route:
        raise ValueError(
            f"filename route {file_route!r} does not match requested route {route!r}: {path.name}"
        )
    return tag, flight_id


def file_label(csv_path: Path) -> int:
    labels = pd.read_csv(csv_path, usecols=[TARGET_COLUMN])[TARGET_COLUMN]
    labels = pd.to_numeric(labels, errors="coerce").dropna()
    if labels.empty:
        raise ValueError(f"{csv_path} has no valid label values")
    return int(labels.max())


def collect_csv_records(ready_folder: Path, route: str) -> list[CsvRecord]:
    records: list[CsvRecord] = []
    for suffix in DATASET_SUFFIXES:
        folder = ready_folder / f"{route}{suffix}"
        if not folder.is_dir():
            raise ValueError(f"missing cleaned folder: {folder}")
        for csv_path in sorted(folder.glob("*.csv")):
            _, flight_id = parse_cleaned_filename(csv_path, route)
            records.append(
                CsvRecord(
                    path=csv_path,
                    flight_id=flight_id,
                    file_label=file_label(csv_path),
                )
            )
    if not records:
        raise ValueError(f"no cleaned CSVs found in {ready_folder}")
    return records


def split_flight_ids(
    flight_ids: list[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    shuffled = np.array(sorted(set(flight_ids)), dtype=object)
    rng.shuffle(shuffled)

    total = len(shuffled)
    if total < 3:
        raise ValueError(
            "at least 3 unique flight IDs are required to create train/val/test splits"
        )
    train_count = int(round(total * train_ratio))
    val_count = int(round(total * val_ratio))
    train_count = min(max(train_count, 1), total - 2)
    val_count = min(max(val_count, 1), total - train_count - 1)

    train_ids = set(shuffled[:train_count])
    val_ids = set(shuffled[train_count: train_count + val_count])
    test_ids = set(shuffled[train_count + val_count:])

    split_by_flight = {}
    for flight_id in train_ids:
        split_by_flight[str(flight_id)] = "train"
    for flight_id in val_ids:
        split_by_flight[str(flight_id)] = "val"
    for flight_id in test_ids:
        split_by_flight[str(flight_id)] = "test"
    return split_by_flight


def read_feature_frame(csv_path: Path) -> pd.DataFrame:
    required_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    df = pd.read_csv(csv_path)
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {missing}")

    df = df[required_columns].copy()
    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[required_columns].isna().any().any():
        bad_columns = df.columns[df.isna().any()].tolist()
        raise ValueError(f"{csv_path} has non-numeric or missing values in {bad_columns}")
    return df


def fit_train_scaler(records: list[CsvRecord]) -> StandardScaler:
    train_rows = []
    for record in records:
        df = read_feature_frame(record.path)
        train_rows.append(df[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
    if not train_rows:
        raise ValueError("no training rows available for scaler fitting")

    scaler = StandardScaler()
    scaler.fit(np.vstack(train_rows))
    return scaler


def make_windows(
    df: pd.DataFrame,
    scaler: StandardScaler,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    values = scaler.transform(values).astype(np.float32)
    labels = df[TARGET_COLUMN].to_numpy(dtype=np.int64)

    windows = []
    window_labels = []
    for start in range(0, len(df) - window_size + 1, stride):
        end = start + window_size
        windows.append(values[start:end])
        window_labels.append(int(labels[start:end].max()))

    if not windows:
        return (
            np.empty((0, window_size, len(FEATURE_COLUMNS)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    return np.stack(windows), np.array(window_labels, dtype=np.int64)


def build_split_arrays(
    records: list[CsvRecord],
    scaler: StandardScaler,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_parts = []
    y_parts = []
    for record in records:
        df = read_feature_frame(record.path)
        x, y = make_windows(df, scaler, window_size, stride)
        if len(x) == 0:
            print(
                f"WARNING {record.path}: shorter than window_size={window_size}; skipped",
                file=sys.stderr,
            )
            continue
        x_parts.append(x)
        y_parts.append(y)

    if not x_parts:
        raise ValueError("no windows generated for split")
    return np.vstack(x_parts), np.concatenate(y_parts)


def class_counts(labels: np.ndarray) -> dict[str, int]:
    counts = {}
    for label, count in zip(*np.unique(labels, return_counts=True)):
        label_int = int(label)
        name = LABEL_NAMES.get(label_int, str(label_int))
        counts[f"{label_int}:{name}"] = int(count)
    return counts


def write_manifest(
    output_folder: Path,
    records: list[CsvRecord],
    split_by_flight: dict[str, str],
) -> None:
    rows = [
        {
            "split": split_by_flight[record.flight_id],
            "flight_id": record.flight_id,
            "file_label": record.file_label,
            "csv_path": str(record.path),
        }
        for record in records
    ]
    pd.DataFrame(rows).sort_values(["split", "flight_id", "file_label", "csv_path"]).to_csv(
        output_folder / "split_manifest.csv",
        index=False,
    )


def save_scaler(output_folder: Path, scaler: StandardScaler) -> None:
    np.savez(
        output_folder / "scaler.npz",
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
        var=scaler.var_.astype(np.float32),
        feature_columns=np.array(FEATURE_COLUMNS),
    )


def save_metadata(
    output_folder: Path,
    args: argparse.Namespace,
    ready_folder: Path,
    arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    split_by_flight: dict[str, str],
) -> None:
    metadata = {
        "route": args.route.lower(),
        "ready_folder": str(ready_folder),
        "window_size": args.window_size,
        "stride": args.stride,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "label_names": LABEL_NAMES,
        "split_flight_counts": {
            split: sum(1 for value in split_by_flight.values() if value == split)
            for split in ("train", "val", "test")
        },
        "arrays": {
            split: {
                "x_shape": list(x.shape),
                "y_shape": list(y.shape),
                "class_counts": class_counts(y),
            }
            for split, (x, y) in arrays.items()
        },
    }
    with (output_folder / "metadata.json").open("w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)


def ensure_output_folder(output_folder: Path, overwrite: bool) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)
    existing_files = list(output_folder.glob("*"))
    if existing_files and not overwrite:
        raise FileExistsError(f"{output_folder} is not empty; rerun with --overwrite")


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv)
        validate_args(args)

        route = args.route.lower()
        ready_folder = args.ready_folder or dataset_path(f"data_{route}_ready")
        output_folder = args.output_folder or dataset_path(f"data_{route}_seq")
        if not ready_folder.is_dir():
            raise ValueError(f"ready folder does not exist: {ready_folder}")
        ensure_output_folder(output_folder, args.overwrite)

        records = collect_csv_records(ready_folder, route)
        split_by_flight = split_flight_ids(
            [record.flight_id for record in records],
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        records_by_split = {
            split: [record for record in records if split_by_flight[record.flight_id] == split]
            for split in ("train", "val", "test")
        }

        scaler = fit_train_scaler(records_by_split["train"])
        arrays = {
            split: build_split_arrays(
                split_records,
                scaler=scaler,
                window_size=args.window_size,
                stride=args.stride,
            )
            for split, split_records in records_by_split.items()
        }

        for split, (x, y) in arrays.items():
            np.save(output_folder / f"X_{split}.npy", x)
            np.save(output_folder / f"y_{split}.npy", y)
            print(f"{split}: X{x.shape}, y{y.shape}, classes={class_counts(y)}")

        save_scaler(output_folder, scaler)
        write_manifest(output_folder, records, split_by_flight)
        save_metadata(output_folder, args, ready_folder, arrays, split_by_flight)

        print(f"Ready folder: {ready_folder}")
        print(f"Output folder: {output_folder}")
        print(f"Feature count: {len(FEATURE_COLUMNS)}")
        print(f"Flight IDs: {len(set(record.flight_id for record in records))}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
