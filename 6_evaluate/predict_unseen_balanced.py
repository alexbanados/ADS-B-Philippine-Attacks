from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_paths import resolve_dataset_path


LABEL_NAMES = {
    0: "authentic",
    1: "modified_altitude",
    2: "modified_speed",
    3: "modified_position",
}

SCORE_COLUMNS = (
    ("auth", "authentic_score_max"),
    ("alt", "modified_altitude_score_max"),
    ("spd", "modified_speed_score_max"),
    ("pos", "modified_position_score_max"),
)

BETA_RESULTS_FOLDER = Path("betatesting/betatesting_results")
DEFAULT_FILE_OUTPUT = Path("balanced_unseen_predictions.csv")
DEFAULT_FNR_FPR_OUTPUT = Path("balanced_unseen_fnr_fpr_argmax4.csv")


def relative_display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def confirm_test_run(model_path: Path, targets: list[str]) -> bool:
    model_text = relative_display_path(model_path)
    input_text = ", ".join(targets)
    answer = input(
        f"You are testing model {model_text} on {input_text}. Proceed? [y/n] "
    ).strip().lower()
    return answer in {"y", "yes"}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict labels for unseen per-flight CSVs using a balanced file-level rule. "
            "By default, the whole file is scanned for attack-predicted windows. "
            "If any window is predicted as an attack, the file is treated as an attack "
            "file, so many authentic-looking windows do not dominate the decision."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="One or more CSV files or folders containing unseen CSVs.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/keras_sequence_ceb/best_model.keras"),
        help="Path to trained .keras model.",
    )
    parser.add_argument(
        "--seq-folder",
        type=Path,
        default=Path("dataset/data_ceb_seq"),
        help="Folder containing metadata.json and scaler.npz from 4_input/csv_9makewindows.py.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help=(
            "Where to write one prediction row per input CSV. Defaults to "
            "betatesting/betatesting_results/beta_ceb_<model>_filepred_balanced_argmax4.csv "
            "for beta-testing inputs, otherwise balanced_unseen_predictions.csv."
        ),
    )
    parser.add_argument(
        "--window-output-csv",
        type=Path,
        help=(
            "Optional CSV for one prediction row per generated window. For beta-testing "
            "inputs, defaults to "
            "betatesting/betatesting_results/beta_ceb_<model>_winpred_balanced_argmax4.csv."
        ),
    )
    parser.add_argument(
        "--confusion-matrix-csv",
        type=Path,
        help=(
            "Where to write a labeled file-level confusion matrix. For beta-testing "
            "inputs, defaults to "
            "betatesting/betatesting_results/beta_ceb_<model>_confmat_balanced_argmax4.csv."
        ),
    )
    parser.add_argument(
        "--classification-report-csv",
        type=Path,
        help=(
            "Where to write a labeled file-level classification report. For beta-testing "
            "inputs, defaults to "
            "betatesting/betatesting_results/beta_ceb_<model>_classreport_balanced_argmax4.csv."
        ),
    )
    parser.add_argument(
        "--fnr-fpr-csv",
        type=Path,
        help=(
            "Where to write macro and per-class file-level FNR/FPR. Defaults to "
            "betatesting/betatesting_results/beta_ceb_<model>_fnrfpr_balanced_argmax4.csv for "
            "beta-testing inputs, otherwise balanced_unseen_fnr_fpr_argmax4.csv."
        ),
    )
    parser.add_argument(
        "--keras-backend",
        choices=("tensorflow", "jax", "torch"),
        default="tensorflow",
        help="Keras backend used to load the model. Default: tensorflow.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Prediction batch size. Default: 128.",
    )
    parser.add_argument(
        "--attack-threshold",
        type=float,
        default=0.50,
        help=(
            "Deprecated; accepted for compatibility but ignored. File-level prediction "
            "uses the balanced attack-aware rule."
        ),
    )
    parser.add_argument(
        "--decision-scope",
        choices=("auto-event", "all-windows"),
        default="all-windows",
        help=(
            "Which windows to use for the file-level decision. all-windows scans the "
            "entire file without using labels. auto-event uses only windows overlapping "
            "label > 0 rows when labels exist; otherwise it uses all windows. "
            "Default: all-windows."
        ),
    )
    parser.add_argument(
        "--attack-window-min",
        type=int,
        default=1,
        help=(
            "Minimum number of decision windows that must have an attack class as "
            "argmax before the file is called an attack. Default: 1."
        ),
    )
    parser.add_argument(
        "--attack-window-fraction-min",
        type=float,
        default=0.0,
        help=(
            "Minimum fraction of decision windows that must have the same attack class "
            "as argmax before the file is called that attack. Default: 0.0."
        ),
    )
    parser.add_argument(
        "--attack-score-min",
        type=float,
        default=0.0,
        help=(
            "Minimum max softmax score required for a candidate attack class. Keep this "
            "at 0.0 for pure argmax4-style behavior. Default: 0.0."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input folders recursively for CSV files.",
    )
    parser.add_argument(
        "--manifest-split",
        choices=("train", "val", "test"),
        help=(
            "Read CSV paths from <seq-folder>/split_manifest.csv for this split. "
            "Useful when you want the same balanced predictor on the original "
            "train/val/test split."
        ),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        help=(
            "Optional explicit split manifest path. Defaults to "
            "<seq-folder>/split_manifest.csv when --manifest-split is used."
        ),
    )
    return parser.parse_args(argv)


def is_betatesting_input(path: Path) -> bool:
    return "betatesting_data" in path.parts and any(
        part.startswith("beta_") for part in path.parts
    )


def beta_model_tag(model_path: Path) -> str:
    path_text = "/".join(part.lower() for part in model_path.parts)
    if "ceb_par" in path_text or "partcnlstm" in path_text:
        return "par"
    if "ceb_seq" in path_text or "seqtcnlstm" in path_text or "sequence" in path_text:
        return "seq"
    if "ceb_lstm" in path_text or "lstm" in path_text:
        return "lstm"
    if "ceb_tcn" in path_text or "tcn" in path_text:
        return "tcn"
    return "model"


def beta_output_path(model_path: Path, prediction_level: str) -> Path:
    return BETA_RESULTS_FOLDER / (
        f"beta_ceb_{beta_model_tag(model_path)}_{prediction_level}_balanced_argmax4.csv"
    )


def apply_default_output_paths(args: argparse.Namespace) -> None:
    is_beta_run = any(is_betatesting_input(input_path) for input_path in args.inputs)
    if args.output_csv is None:
        args.output_csv = (
            beta_output_path(args.model, "filepred") if is_beta_run else DEFAULT_FILE_OUTPUT
        )
    if args.window_output_csv is None and is_beta_run:
        args.window_output_csv = beta_output_path(args.model, "winpred")
    if args.confusion_matrix_csv is None and is_beta_run:
        args.confusion_matrix_csv = beta_output_path(args.model, "confmat")
    if args.classification_report_csv is None and is_beta_run:
        args.classification_report_csv = beta_output_path(args.model, "classreport")
    if args.fnr_fpr_csv is None:
        args.fnr_fpr_csv = (
            beta_output_path(args.model, "fnrfpr") if is_beta_run else DEFAULT_FNR_FPR_OUTPUT
        )


def import_keras(backend: str):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "matplotlib"))
    os.environ["KERAS_BACKEND"] = backend
    try:
        import keras
    except Exception as exc:
        raise RuntimeError(f"Keras could not be imported with backend={backend!r}") from exc
    return keras


def load_metadata(seq_folder: Path) -> dict:
    metadata_path = seq_folder / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing metadata file: {metadata_path}")
    with metadata_path.open() as metadata_file:
        return json.load(metadata_file)


def load_scaler(seq_folder: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    scaler_path = seq_folder / "scaler.npz"
    if not scaler_path.is_file():
        raise FileNotFoundError(f"missing scaler file: {scaler_path}")
    scaler = np.load(scaler_path)
    mean = scaler["mean"].astype(np.float32)
    scale = scaler["scale"].astype(np.float32)
    feature_columns = [str(value) for value in scaler["feature_columns"].tolist()]
    return mean, scale, feature_columns


def iter_csv_paths(inputs: list[Path], recursive: bool) -> list[Path]:
    csv_paths: list[Path] = []
    for input_path in inputs:
        input_path = resolve_dataset_path(input_path)
        if input_path.is_file():
            if input_path.suffix.lower() != ".csv":
                raise ValueError(f"input file is not a CSV: {input_path}")
            csv_paths.append(input_path)
        elif input_path.is_dir():
            pattern = "**/*.csv" if recursive else "*.csv"
            csv_paths.extend(sorted(path for path in input_path.glob(pattern) if path.is_file()))
        else:
            raise FileNotFoundError(f"input path does not exist: {input_path}")

    unique_paths = sorted(dict.fromkeys(csv_paths))
    if not unique_paths:
        raise ValueError("no CSV files found in input paths")
    return unique_paths


def iter_manifest_csv_paths(manifest_path: Path, split: str) -> list[Path]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"split manifest does not exist: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    required = ["split", "csv_path"]
    missing = [column for column in required if column not in manifest.columns]
    if missing:
        raise ValueError(f"{manifest_path} is missing columns: {missing}")

    csv_paths = [
        resolve_dataset_path(Path(csv_path))
        for csv_path in manifest.loc[manifest["split"].eq(split), "csv_path"].tolist()
    ]
    if not csv_paths:
        raise ValueError(f"no CSV paths found in {manifest_path} for split={split}")
    return csv_paths


def collect_input_csv_paths(args: argparse.Namespace) -> list[Path]:
    csv_paths: list[Path] = []
    if args.manifest_split is not None:
        manifest_path = args.split_manifest or args.seq_folder / "split_manifest.csv"
        csv_paths.extend(iter_manifest_csv_paths(manifest_path, args.manifest_split))
    if args.inputs:
        csv_paths.extend(iter_csv_paths(args.inputs, args.recursive))

    unique_paths = sorted(dict.fromkeys(csv_paths))
    if not unique_paths:
        raise ValueError("no CSV files found; pass inputs or use --manifest-split")
    return unique_paths


def confirmation_targets(args: argparse.Namespace) -> list[str]:
    targets = []
    if args.manifest_split is not None:
        manifest_path = args.split_manifest or args.seq_folder / "split_manifest.csv"
        targets.append(
            f"{relative_display_path(manifest_path)} split={args.manifest_split}"
        )
    targets.extend(relative_display_path(path) for path in args.inputs)
    return targets


def read_feature_frame(csv_path: Path, feature_columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")

    feature_df = df[feature_columns].copy()
    for column in feature_columns:
        feature_df[column] = pd.to_numeric(feature_df[column], errors="coerce")
    if feature_df.isna().any().any():
        bad_columns = feature_df.columns[feature_df.isna().any()].tolist()
        raise ValueError(f"non-numeric or missing values in columns: {bad_columns}")
    df = df.copy()
    df[feature_columns] = feature_df
    return df


def label_from_path(csv_path: Path) -> int | None:
    path_text = "/".join(part.lower() for part in csv_path.parts)
    if "modalt" in path_text or "modified_altitude" in path_text:
        return 1
    if "modspd" in path_text or "modified_speed" in path_text:
        return 2
    if "modpos" in path_text or "modified_position" in path_text:
        return 3
    if "auth" in path_text or "authentic" in path_text:
        return 0
    return None


def true_label_from_frame(df: pd.DataFrame, csv_path: Path) -> int | None:
    if "label" not in df.columns:
        return label_from_path(csv_path)
    labels = pd.to_numeric(df["label"], errors="coerce").dropna()
    if labels.empty:
        return label_from_path(csv_path)
    return int(labels.max())


def make_windows_with_spans(
    df: pd.DataFrame,
    feature_columns: list[str],
    mean: np.ndarray,
    scale: np.ndarray,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = df[feature_columns].to_numpy(dtype=np.float32)
    values = ((values - mean) / scale).astype(np.float32)

    windows = []
    spans = []
    for start in range(0, len(df) - window_size + 1, stride):
        end = start + window_size
        windows.append(values[start:end])
        spans.append((start, end))

    if not windows:
        return (
            np.empty((0, window_size, len(feature_columns)), dtype=np.float32),
            np.empty((0, 2), dtype=np.int64),
        )
    return np.stack(windows), np.array(spans, dtype=np.int64)


def attack_overlap_window_indexes(df: pd.DataFrame, spans: np.ndarray) -> np.ndarray:
    if "label" not in df.columns or len(spans) == 0:
        return np.empty(0, dtype=np.int64)

    labels = pd.to_numeric(df["label"], errors="coerce").fillna(0).to_numpy()
    attack_rows = labels > 0
    if not attack_rows.any():
        return np.empty(0, dtype=np.int64)

    attack_prefix = np.concatenate(([0], np.cumsum(attack_rows.astype(np.int64))))
    overlaps = [
        attack_prefix[int(end)] - attack_prefix[int(start)] > 0
        for start, end in spans
    ]
    return np.flatnonzero(overlaps)


def decision_window_indexes(
    df: pd.DataFrame,
    spans: np.ndarray,
    decision_scope: str,
) -> tuple[np.ndarray, str]:
    all_indexes = np.arange(len(spans), dtype=np.int64)
    if decision_scope == "auto-event":
        event_indexes = attack_overlap_window_indexes(df, spans)
        if len(event_indexes) > 0:
            return event_indexes, "event_windows"
    return all_indexes, "all_windows"


def balanced_file_prediction(
    probabilities: np.ndarray,
    df: pd.DataFrame,
    spans: np.ndarray,
    decision_scope: str,
    attack_window_min: int,
    attack_window_fraction_min: float,
    attack_score_min: float,
) -> dict[str, object]:
    indexes, resolved_scope = decision_window_indexes(df, spans, decision_scope)
    if len(indexes) == 0:
        indexes = np.arange(len(probabilities), dtype=np.int64)
        resolved_scope = "all_windows"

    decision_probabilities = probabilities[indexes]
    decision_predictions = decision_probabilities.argmax(axis=1)
    decision_max_by_class = decision_probabilities.max(axis=0)
    all_max_by_class = probabilities.max(axis=0)

    attack_counts = np.bincount(decision_predictions, minlength=4)[1:4]
    decision_count = int(len(decision_probabilities))
    qualified_attack_labels = []
    for label in range(1, 4):
        attack_count = int(attack_counts[label - 1])
        attack_fraction = attack_count / decision_count if decision_count else 0.0
        if (
            attack_count >= attack_window_min
            and attack_fraction >= attack_window_fraction_min
            and float(decision_max_by_class[label]) >= attack_score_min
        ):
            qualified_attack_labels.append(label)

    if qualified_attack_labels:
        predicted_label = max(
            qualified_attack_labels,
            key=lambda label: float(decision_max_by_class[label]),
        )
        decision_reason = "qualified_attack_window"
    else:
        predicted_label = 0
        decision_reason = "no_qualified_attack_window"

    score = float(decision_max_by_class[predicted_label])
    attack_window_count = int((decision_predictions > 0).sum())
    attack_window_fraction = (
        attack_window_count / decision_count if decision_count else 0.0
    )

    return {
        "predicted_label": int(predicted_label),
        "score": score,
        "decision_scope": resolved_scope,
        "decision_reason": decision_reason,
        "decision_window_count": decision_count,
        "attack_window_count": attack_window_count,
        "attack_window_fraction": attack_window_fraction,
        "decision_max_by_class": decision_max_by_class,
        "all_max_by_class": all_max_by_class,
    }


def format_prediction_scores(row) -> str:
    scores = []
    for short_name, column in SCORE_COLUMNS:
        value = getattr(row, column)
        scores.append(f"{short_name}={float(value):.6f}")
    return " ".join(scores)


def labeled_confusion_df(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    labels = sorted(set(LABEL_NAMES) | set(map(int, y_true)) | set(map(int, y_pred)))
    names = [LABEL_NAMES.get(label, str(label)) for label in labels]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=names, columns=names)


def fnr_fpr_df(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    labels = sorted(set(LABEL_NAMES) | set(map(int, y_true)) | set(map(int, y_pred)))
    rows = []
    for label in labels:
        actual_positive = y_true == label
        predicted_positive = y_pred == label
        tp = int((actual_positive & predicted_positive).sum())
        fn = int((actual_positive & ~predicted_positive).sum())
        fp = int((~actual_positive & predicted_positive).sum())
        tn = int((~actual_positive & ~predicted_positive).sum())
        positive_count = tp + fn
        negative_count = fp + tn
        rows.append(
            {
                "scope": "class",
                "label": label,
                "class_name": LABEL_NAMES.get(label, str(label)),
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
                "support": positive_count,
                "predicted_positive": tp + fp,
                "fnr": fn / positive_count if positive_count else np.nan,
                "fpr": fp / negative_count if negative_count else np.nan,
            }
        )

    metrics_df = pd.DataFrame(rows)
    macro_fnr = float(metrics_df["fnr"].mean(skipna=True)) if not metrics_df.empty else np.nan
    macro_fpr = float(metrics_df["fpr"].mean(skipna=True)) if not metrics_df.empty else np.nan
    macro_row = {
        "scope": "macro_avg",
        "label": None,
        "class_name": "macro_avg",
        "tp": None,
        "fn": None,
        "fp": None,
        "tn": None,
        "support": int(len(y_true)),
        "predicted_positive": int(len(y_pred)),
        "fnr": macro_fnr,
        "fpr": macro_fpr,
    }
    return pd.concat([metrics_df, pd.DataFrame([macro_row])], ignore_index=True)


def write_labeled_reports(
    labeled_df: pd.DataFrame,
    confusion_matrix_csv: Path | None,
    classification_report_csv: Path | None,
    fnr_fpr_csv: Path | None,
) -> None:
    if (
        confusion_matrix_csv is None
        and classification_report_csv is None
        and fnr_fpr_csv is None
    ):
        return

    y_true = labeled_df["true_label"].to_numpy(dtype=int)
    y_pred = labeled_df["predicted_label"].to_numpy(dtype=int)
    labels = sorted(set(LABEL_NAMES) | set(map(int, y_true)) | set(map(int, y_pred)))
    names = [LABEL_NAMES.get(label, str(label)) for label in labels]

    if confusion_matrix_csv is not None:
        confusion_matrix_csv.parent.mkdir(parents=True, exist_ok=True)
        labeled_confusion_df(y_true, y_pred).to_csv(confusion_matrix_csv)

    if classification_report_csv is not None:
        classification_report_csv.parent.mkdir(parents=True, exist_ok=True)
        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=names,
            output_dict=True,
            zero_division=0,
        )
        pd.DataFrame(report).transpose().to_csv(classification_report_csv)

    if fnr_fpr_csv is not None:
        fnr_fpr_csv.parent.mkdir(parents=True, exist_ok=True)
        fnr_fpr_df(y_true, y_pred).to_csv(fnr_fpr_csv, index=False)


def prediction_row_for_error(csv_path: Path, status: str, error: str) -> dict[str, object]:
    return {
        "csv_path": str(csv_path),
        "status": status,
        "rows": None,
        "window_count": 0,
        "true_label": None,
        "true_label_name": None,
        "predicted_label": None,
        "predicted_label_name": None,
        "score": None,
        "decision_scope": None,
        "decision_reason": None,
        "decision_window_count": None,
        "attack_window_count": None,
        "attack_window_fraction": None,
        "correct_if_labeled": None,
        "authentic_score_mean": None,
        "authentic_score_max": None,
        "modified_altitude_score_max": None,
        "modified_speed_score_max": None,
        "modified_position_score_max": None,
        "all_authentic_score_max": None,
        "all_modified_altitude_score_max": None,
        "all_modified_speed_score_max": None,
        "all_modified_position_score_max": None,
        "error": error,
    }


def predict_csv(
    model,
    csv_path: Path,
    feature_columns: list[str],
    mean: np.ndarray,
    scale: np.ndarray,
    window_size: int,
    stride: int,
    batch_size: int,
    decision_scope: str,
    attack_window_min: int,
    attack_window_fraction_min: float,
    attack_score_min: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    df = read_feature_frame(csv_path, feature_columns)
    true_label = true_label_from_frame(df, csv_path)
    windows, spans = make_windows_with_spans(
        df=df,
        feature_columns=feature_columns,
        mean=mean,
        scale=scale,
        window_size=window_size,
        stride=stride,
    )
    if len(windows) == 0:
        return (
            prediction_row_for_error(
                csv_path,
                status="skipped_short",
                error=f"CSV has {len(df)} rows, fewer than window_size={window_size}",
            ),
            [],
        )

    probabilities = model.predict(windows, batch_size=batch_size, verbose=0)
    window_predictions = probabilities.argmax(axis=1)
    prediction = balanced_file_prediction(
        probabilities=probabilities,
        df=df,
        spans=spans,
        decision_scope=decision_scope,
        attack_window_min=attack_window_min,
        attack_window_fraction_min=attack_window_fraction_min,
        attack_score_min=attack_score_min,
    )
    predicted_label = int(prediction["predicted_label"])
    score = float(prediction["score"])
    decision_max_by_class = prediction["decision_max_by_class"]
    all_max_by_class = prediction["all_max_by_class"]

    row = {
        "csv_path": str(csv_path),
        "status": "ok",
        "rows": len(df),
        "window_count": len(windows),
        "true_label": true_label,
        "true_label_name": LABEL_NAMES.get(true_label) if true_label is not None else None,
        "predicted_label": predicted_label,
        "predicted_label_name": LABEL_NAMES.get(predicted_label, str(predicted_label)),
        "score": score,
        "decision_scope": prediction["decision_scope"],
        "decision_reason": prediction["decision_reason"],
        "decision_window_count": prediction["decision_window_count"],
        "attack_window_count": prediction["attack_window_count"],
        "attack_window_fraction": prediction["attack_window_fraction"],
        "correct_if_labeled": (
            bool(true_label == predicted_label) if true_label is not None else None
        ),
        "authentic_score_mean": float(probabilities[:, 0].mean()),
        "authentic_score_max": float(decision_max_by_class[0]),
        "modified_altitude_score_max": float(decision_max_by_class[1]),
        "modified_speed_score_max": float(decision_max_by_class[2]),
        "modified_position_score_max": float(decision_max_by_class[3]),
        "all_authentic_score_max": float(all_max_by_class[0]),
        "all_modified_altitude_score_max": float(all_max_by_class[1]),
        "all_modified_speed_score_max": float(all_max_by_class[2]),
        "all_modified_position_score_max": float(all_max_by_class[3]),
        "error": None,
    }

    window_rows = []
    for index, ((start, end), prediction, proba) in enumerate(
        zip(spans, window_predictions, probabilities)
    ):
        window_rows.append(
            {
                "csv_path": str(csv_path),
                "window_index": index,
                "start_row": int(start),
                "end_row_exclusive": int(end),
                "predicted_label": int(prediction),
                "predicted_label_name": LABEL_NAMES.get(int(prediction), str(int(prediction))),
                "prob_authentic": float(proba[0]),
                "prob_modified_altitude": float(proba[1]),
                "prob_modified_speed": float(proba[2]),
                "prob_modified_position": float(proba[3]),
            }
        )
    return row, window_rows


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv)
        apply_default_output_paths(args)
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be greater than 0")
        if args.attack_window_min < 0:
            raise ValueError("--attack-window-min must be zero or greater")
        if not 0.0 <= args.attack_window_fraction_min <= 1.0:
            raise ValueError("--attack-window-fraction-min must be between 0 and 1")
        if not 0.0 <= args.attack_score_min <= 1.0:
            raise ValueError("--attack-score-min must be between 0 and 1")
        if not args.model.is_file():
            raise FileNotFoundError(f"model does not exist: {args.model}")
        if not confirm_test_run(args.model, confirmation_targets(args)):
            print("Aborted.")
            return 1

        metadata = load_metadata(args.seq_folder)
        mean, scale, feature_columns = load_scaler(args.seq_folder)
        window_size = int(metadata["window_size"])
        stride = int(metadata["stride"])
        csv_paths = collect_input_csv_paths(args)

        keras = import_keras(args.keras_backend)
        model = keras.models.load_model(args.model, compile=False)

        prediction_rows = []
        all_window_rows = []
        for csv_path in csv_paths:
            try:
                prediction_row, window_rows = predict_csv(
                    model=model,
                    csv_path=csv_path,
                    feature_columns=feature_columns,
                    mean=mean,
                    scale=scale,
                    window_size=window_size,
                    stride=stride,
                    batch_size=args.batch_size,
                    decision_scope=args.decision_scope,
                    attack_window_min=args.attack_window_min,
                    attack_window_fraction_min=args.attack_window_fraction_min,
                    attack_score_min=args.attack_score_min,
                )
                prediction_rows.append(prediction_row)
                all_window_rows.extend(window_rows)
            except Exception as exc:
                prediction_rows.append(
                    prediction_row_for_error(csv_path, status="error", error=str(exc))
                )

        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        prediction_df = pd.DataFrame(prediction_rows)
        prediction_df.to_csv(args.output_csv, index=False)

        if args.window_output_csv is not None:
            args.window_output_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(all_window_rows).to_csv(args.window_output_csv, index=False)

        ok_df = prediction_df[prediction_df["status"].eq("ok")]
        print(f"CSV files found: {len(csv_paths)}")
        print(f"Prediction rows generated: {len(ok_df)}")
        print(f"File-level predictions written to: {args.output_csv}")
        if args.window_output_csv is not None:
            print(f"Window-level predictions written to: {args.window_output_csv}")

        if not ok_df.empty:
            print("Per-file balanced prediction scores:")
            for row in ok_df.itertuples(index=False):
                print(
                    "  "
                    f"{row.csv_path}: "
                    f"scores[{format_prediction_scores(row)}], "
                    f"decision={row.decision_scope}, "
                    f"attack_windows={row.attack_window_count}/{row.decision_window_count}, "
                    f"predicted={row.predicted_label_name}"
                )

        labeled_df = ok_df[ok_df["true_label"].notna()]
        if not labeled_df.empty:
            accuracy = float(labeled_df["correct_if_labeled"].mean())
            correct_df = labeled_df[labeled_df["correct_if_labeled"].eq(True)]
            wrong_df = labeled_df[labeled_df["correct_if_labeled"].eq(False)]
            print(f"Correct Classification: {len(correct_df)}")
            print(f"Wrong Classification: {len(wrong_df)}")
            print(f"Accuracy on labeled unseen files: {accuracy:.4f}")
            write_labeled_reports(
                labeled_df=labeled_df,
                confusion_matrix_csv=args.confusion_matrix_csv,
                classification_report_csv=args.classification_report_csv,
                fnr_fpr_csv=args.fnr_fpr_csv,
            )
            if args.confusion_matrix_csv is not None:
                print(f"Confusion matrix written to: {args.confusion_matrix_csv}")
            if args.classification_report_csv is not None:
                print(f"Classification report written to: {args.classification_report_csv}")
            if args.fnr_fpr_csv is not None:
                print(f"FNR/FPR report written to: {args.fnr_fpr_csv}")
            if not wrong_df.empty:
                print("Misclassified files:")
                for row in wrong_df.itertuples(index=False):
                    print(
                        "  "
                        f"{row.csv_path} "
                        f"(true={row.true_label_name}, "
                        f"predicted={row.predicted_label_name}, "
                        f"scores[{format_prediction_scores(row)}])"
                    )

        error_count = int((prediction_df["status"] == "error").sum())
        skipped_count = int((prediction_df["status"] == "skipped_short").sum())
        if error_count or skipped_count:
            print(
                f"Warnings: {error_count} errors, {skipped_count} shorter-than-window files",
                file=sys.stderr,
            )
        return 0 if error_count == 0 else 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
