from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import zipfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import h5py


DEFAULT_BACKEND = "tensorflow"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a Keras .keras model: architecture summary, parameter "
            "counts, compile config, and nearby training metadata when present."
        )
    )
    parser.add_argument(
        "model_path",
        type=Path,
        help="Path to a .keras model file.",
    )
    parser.add_argument(
        "--keras-backend",
        choices=("jax", "torch", "tensorflow"),
        default=DEFAULT_BACKEND,
        help=f"Keras backend to use when loading the model. Default: {DEFAULT_BACKEND}.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a text report.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Load the model with Keras and include model.summary() output.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Deprecated compatibility flag; summary is skipped by default.",
    )
    return parser.parse_args(argv)


def load_keras_model(model_path: Path, backend: str):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    os.environ["KERAS_BACKEND"] = backend

    import keras

    return keras.models.load_model(model_path)


def keras_summary_text(model) -> str:
    buffer = StringIO()
    with redirect_stdout(buffer):
        model.summary()
    return buffer.getvalue().rstrip()


def read_keras_archive(model_path: Path) -> tuple[dict, dict]:
    with zipfile.ZipFile(model_path) as archive:
        config = json.loads(archive.read("config.json").decode("utf-8"))
        metadata = json.loads(archive.read("metadata.json").decode("utf-8"))
    return config, metadata


def layer_configs(config: dict) -> list[dict]:
    layers = config.get("config", {}).get("layers", [])
    extracted = []
    for index, layer in enumerate(layers):
        layer_config = layer.get("config", {})
        extracted.append(
            {
                "index": index,
                "name": layer_config.get("name", layer.get("name")),
                "class_name": layer.get("class_name"),
                "activation": layer_config.get("activation"),
                "units": layer_config.get("units"),
                "filters": layer_config.get("filters"),
                "kernel_size": layer_config.get("kernel_size"),
                "dilation_rate": layer_config.get("dilation_rate"),
                "dropout_rate": layer_config.get("rate"),
                "return_sequences": layer_config.get("return_sequences"),
            }
        )
    return extracted


def compile_config(config: dict) -> dict:
    compile_cfg = config.get("compile_config") or {}
    optimizer = compile_cfg.get("optimizer")
    loss = compile_cfg.get("loss")
    metrics = compile_cfg.get("metrics")

    optimizer_name = None
    learning_rate = None
    if isinstance(optimizer, dict):
        optimizer_name = optimizer.get("class_name")
        optimizer_config = optimizer.get("config", {})
        learning_rate = optimizer_config.get("learning_rate")
    elif optimizer is not None:
        optimizer_name = str(optimizer)

    return {
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "loss": loss,
        "metrics": metrics,
    }


def count_saved_values(model_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with zipfile.ZipFile(model_path) as archive:
        weight_names = [
            name for name in archive.namelist()
            if name.endswith(".weights.h5") or name.endswith("model.weights.h5")
        ]
        if not weight_names:
            return counts
        weights_path = weight_names[0]
        with tempfile.TemporaryDirectory() as temp_dir:
            extracted_path = Path(temp_dir) / Path(weights_path).name
            extracted_path.write_bytes(archive.read(weights_path))
            with h5py.File(extracted_path, "r") as h5_file:
                def visit(name, obj):
                    if not isinstance(obj, h5py.Dataset):
                        return
                    count = 1
                    for dim in obj.shape:
                        count *= int(dim)
                    top_level = name.split("/", 1)[0]
                    counts[top_level] = counts.get(top_level, 0) + count

                h5_file.visititems(visit)
    return counts


def nearby_metadata(model_path: Path) -> dict[str, object]:
    folder = model_path.parent
    metadata = {}
    for filename in [
        "run_config.json",
        "metadata.json",
        "best_test_metrics.json",
        "final_test_metrics.json",
        "test_metrics.json",
    ]:
        path = folder / filename
        if not path.is_file():
            continue
        try:
            metadata[filename] = json.loads(path.read_text())
        except Exception as exc:
            metadata[filename] = f"could not read: {exc}"
    return metadata


def inspect_model(model_path: Path, backend: str, include_summary: bool) -> dict:
    if not model_path.is_file():
        raise FileNotFoundError(f"model file does not exist: {model_path}")

    config, archive_metadata = read_keras_archive(model_path)
    saved_value_counts = count_saved_values(model_path)
    report = {
        "model_path": str(model_path),
        "archive_metadata": archive_metadata,
        "class_name": config.get("class_name"),
        "model_name": config.get("config", {}).get("name"),
        "layers": layer_configs(config),
        "compile_config": compile_config(config),
        "saved_value_counts": saved_value_counts,
        "nearby_metadata": nearby_metadata(model_path),
    }
    if "layers" in saved_value_counts:
        report["total_params_from_weights"] = saved_value_counts["layers"]

    if include_summary:
        try:
            model = load_keras_model(model_path, backend)
            report["keras_backend"] = backend
            report["total_params"] = int(model.count_params())
            report["trainable_params"] = int(
                sum(variable.numpy().size for variable in model.trainable_variables)
            )
            report["non_trainable_params"] = int(
                sum(variable.numpy().size for variable in model.non_trainable_variables)
            )
            report["summary"] = keras_summary_text(model)
        except Exception as exc:
            report["load_error"] = f"{type(exc).__name__}: {exc}"

    return report


def compact_layer_line(layer: dict) -> str:
    details = []
    for key in [
        "units",
        "filters",
        "kernel_size",
        "dilation_rate",
        "activation",
        "dropout_rate",
        "return_sequences",
    ]:
        value = layer.get(key)
        if value is not None:
            details.append(f"{key}={value}")
    suffix = f" ({', '.join(details)})" if details else ""
    return f"{layer['index']:>2}. {layer['name']}: {layer['class_name']}{suffix}"


def print_text_report(report: dict) -> None:
    print(f"Model: {report['model_path']}")
    if report.get("model_name"):
        print(f"Name: {report['model_name']}")
    print(f"Class: {report.get('class_name')}")

    if "total_params" in report:
        print(f"Total params: {report['total_params']:,}")
        print(f"Trainable params: {report['trainable_params']:,}")
        print(f"Non-trainable params: {report['non_trainable_params']:,}")
    elif "total_params_from_weights" in report:
        print(f"Total params from saved layer weights: {report['total_params_from_weights']:,}")

    saved_counts = report.get("saved_value_counts", {})
    if saved_counts:
        print("Saved value counts:")
        for name, count in sorted(saved_counts.items()):
            print(f"  {name}: {count:,}")

    compile_cfg = report.get("compile_config", {})
    if compile_cfg:
        print("Compile config:")
        for key, value in compile_cfg.items():
            print(f"  {key}: {value}")

    print("Layers:")
    for layer in report.get("layers", []):
        print(f"  {compact_layer_line(layer)}")

    nearby = report.get("nearby_metadata", {})
    if nearby:
        print("Nearby metadata files:")
        for filename in nearby:
            print(f"  {filename}")

    if report.get("load_error"):
        print(f"Load warning: {report['load_error']}")

    if report.get("summary"):
        print("\nKeras Summary")
        print(report["summary"])


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv)
        report = inspect_model(
            model_path=args.model_path,
            backend=args.keras_backend,
            include_summary=args.summary and not args.no_summary,
        )
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print_text_report(report)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
