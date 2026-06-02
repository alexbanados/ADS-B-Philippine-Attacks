from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = (
    (
        "ilo_par on auth/mods/seq gen poisons",
        "beta_ilo_par_on_seqpoison_filepred_argmax4.csv",
    ),
    (
        "ilo_seq on auth/mods/par gen poisons",
        "beta_ilo_seq_on_parpoison_filepred_argmax4.csv",
    ),
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print readable beta-testing predictions from file-level prediction CSVs."
        )
    )
    parser.add_argument(
        "prediction_csvs",
        nargs="*",
        type=Path,
        help=(
            "File-level prediction CSVs. If omitted, prints only the default "
            "ILO poison-transfer beta-testing reports."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "betatesting" / "betatesting_results",
        help="Folder to search when no CSV paths are passed.",
    )
    parser.add_argument(
        "--path-style",
        choices=("name", "path"),
        default="name",
        help="Print only the CSV file name or the full stored path. Default: name.",
    )
    return parser.parse_args(argv)


def infer_model_name(prediction_csv: Path) -> str:
    parts = prediction_csv.stem.split("_")
    if (
        len(parts) >= 4
        and parts[0] == "beta"
        and parts[2].lower().startswith("3")
        and parts[3] in {"par", "seq", "lstm", "tcn"}
    ):
        return f"{parts[1]}_{parts[3]}"
    if len(parts) >= 3 and parts[0] == "beta":
        return f"{parts[1]}_{parts[2]}"
    return prediction_csv.stem


def display_file_name(csv_path_text: str, path_style: str) -> str:
    if path_style == "path":
        return csv_path_text
    return Path(csv_path_text).name


def correctness_tag(row: dict[str, str]) -> str:
    correct_text = row.get("correct_if_labeled", "").strip().lower()
    if correct_text in {"true", "1", "yes"}:
        return "[CORRECT]"
    if correct_text in {"false", "0", "no"}:
        return "[WRONG]"
    if row.get("predicted_label_name") == row.get("true_label_name"):
        return "[CORRECT]"
    return "[WRONG]"


def iter_prediction_csvs(args: argparse.Namespace) -> list[tuple[str | None, Path]]:
    if args.prediction_csvs:
        return [(None, path) for path in args.prediction_csvs]
    return [
        (title, args.results_dir / filename)
        for title, filename in DEFAULT_RUNS
    ]


def print_prediction_rows(prediction_csv: Path, path_style: str) -> int:
    model_name = infer_model_name(prediction_csv)
    printed = 0

    with prediction_csv.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"csv_path", "status", "predicted_label_name", "true_label_name"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{prediction_csv} is missing required columns: {sorted(missing)}"
            )

        for row in reader:
            file_name = display_file_name(row["csv_path"], path_style)
            if row["status"] != "ok":
                error = row.get("error", "unknown error")
                print(
                    f'[WRONG] the prediction of "{model_name}" on "{file_name}" '
                    f"failed: {error}"
                )
                printed += 1
                continue

            prediction = row["predicted_label_name"]
            true_class = row["true_label_name"]
            tag = correctness_tag(row)
            print(
                f'{tag} the prediction of "{model_name}" on "{file_name}" is: '
                f'{prediction} (true class: "{true_class}")'
            )
            printed += 1

    return printed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    prediction_csvs = iter_prediction_csvs(args)
    if not prediction_csvs:
        print(
            f"ERROR: no prediction CSVs found in {args.results_dir} "
            f"matching {args.pattern!r}",
            file=sys.stderr,
        )
        return 1

    errors = 0
    for index, (title, prediction_csv) in enumerate(prediction_csvs):
        if index:
            print()
        try:
            if title is not None:
                print(title)
            print_prediction_rows(prediction_csv, args.path_style)
        except Exception as exc:
            errors += 1
            print(f"ERROR {prediction_csv}: {exc}", file=sys.stderr)

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
