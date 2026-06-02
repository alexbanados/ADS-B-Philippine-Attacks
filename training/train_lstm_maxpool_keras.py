from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


LABEL_NAMES = {
    0: "authentic",
    1: "modified_altitude",
    2: "modified_speed",
    3: "modified_position",
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a Keras LSTM classifier with global max pooling from "
            "4_input/csv_9makewindows.py arrays."
        )
    )
    parser.add_argument(
        "--data-folder",
        type=Path,
        default=Path("dataset/data_ceb_seq"),
        help="Folder containing X_train.npy/y_train.npy/etc. Default: dataset/data_ceb_seq.",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=Path("models/keras_lstm_ceb_tf_smooth_nocw_lr1e4_b128_p80_u32_d20_maxpool"),
        help=(
            "Folder where model and reports will be saved. Default: "
            "models/keras_lstm_ceb_tf_smooth_nocw_lr1e4_b128_p80_u32_d20_maxpool."
        ),
    )
    parser.add_argument(
        "--keras-backend",
        choices=("jax", "torch", "tensorflow"),
        default="tensorflow",
        help=(
            "Keras backend to use. Default: tensorflow."
        ),
    )
    parser.add_argument("--epochs", type=int, default=1000, help="Maximum epochs. Default: 1000.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size. Default: 128.")
    parser.add_argument("--lstm-units", type=int, default=32, help="LSTM units. Default: 32.")
    parser.add_argument("--dense-units", type=int, default=64, help="Dense units. Default: 64.")
    parser.add_argument("--dropout", type=float, default=0.20, help="Dropout rate. Default: 0.20.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Adam learning rate. Default: 0.0001.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=80,
        help="Early-stopping patience on validation loss. Default: 80.",
    )
    parser.add_argument(
        "--restore-best-weights",
        action="store_true",
        help=(
            "Restore best validation-loss weights at early stopping. If omitted, "
            "final_model.keras remains the actual final/stopped epoch."
        ),
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        default=True,
        help="Disable class weights. Default: disabled.",
    )
    parser.add_argument(
        "--use-class-weights",
        dest="no_class_weights",
        action="store_false",
        help="Enable class weights computed from y_train.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for TensorFlow/Keras. Default: 42.",
    )
    return parser.parse_args(argv)


def load_split_arrays(data_folder: Path) -> tuple[np.ndarray, ...]:
    arrays = []
    for split in ("train", "val", "test"):
        x_path = data_folder / f"X_{split}.npy"
        y_path = data_folder / f"y_{split}.npy"
        if not x_path.is_file() or not y_path.is_file():
            raise FileNotFoundError(f"missing {x_path} or {y_path}")
        arrays.extend([np.load(x_path), np.load(y_path)])

    x_train, y_train, x_val, y_val, x_test, y_test = arrays
    for name, x, y in [
        ("train", x_train, y_train),
        ("val", x_val, y_val),
        ("test", x_test, y_test),
    ]:
        if x.ndim != 3:
            raise ValueError(f"X_{name} must be 3D, got shape {x.shape}")
        if y.ndim != 1:
            raise ValueError(f"y_{name} must be 1D, got shape {y.shape}")
        if len(x) != len(y):
            raise ValueError(f"X_{name} and y_{name} length mismatch: {len(x)} != {len(y)}")

    return x_train, y_train, x_val, y_val, x_test, y_test


def load_feature_columns(data_folder: Path, num_features: int) -> list[str]:
    scaler_path = data_folder / "scaler.npz"
    if scaler_path.is_file():
        scaler_data = np.load(scaler_path)
        if "feature_columns" in scaler_data:
            return [str(value) for value in scaler_data["feature_columns"].tolist()]

    return [f"feature_{index}" for index in range(num_features)]


def compute_class_weights(y_train: np.ndarray, num_classes: int) -> dict[int, float]:
    labels, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    weights = {}
    for label, count in zip(labels, counts):
        weights[int(label)] = float(total / (num_classes * count))
    return weights


def import_keras(backend: str):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "matplotlib"))
    os.environ["KERAS_BACKEND"] = backend
    try:
        import keras
    except Exception as exc:
        raise RuntimeError(
            f"Keras could not be imported with backend={backend!r}."
        ) from exc
    if backend != "tensorflow":
        try:
            from keras.src.utils import module_utils

            module_utils.tensorflow._available = False
            module_utils.gfile._available = False
        except Exception:
            pass
    return keras


def build_lstm_model(
    keras,
    window_size: int,
    num_features: int,
    num_classes: int,
    lstm_units: int,
    dense_units: int,
    dropout: float,
    learning_rate: float,
):
    inputs = keras.Input(shape=(window_size, num_features), name="flight_window")
    x = keras.layers.LSTM(lstm_units, return_sequences=True, name="lstm")(inputs)
    x = keras.layers.GlobalMaxPooling1D(name="global_max_pooling")(x)
    x = keras.layers.Dropout(dropout, name="dropout_after_max_pooling")(x)
    x = keras.layers.Dense(dense_units, activation="relu", name="dense")(x)
    x = keras.layers.Dropout(dropout, name="dropout_after_dense")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="class")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="flight_attack_lstm_maxpool")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_history(output_folder: Path, history) -> None:
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(output_folder / "history.csv", index=False)

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(history_df["epoch"], history_df["loss"], label="train")
        axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].grid(True)
        axes[0].legend()

        axes[1].plot(history_df["epoch"], history_df["accuracy"], label="train")
        axes[1].plot(history_df["epoch"], history_df["val_accuracy"], label="val")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].grid(True)
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(output_folder / "history.png", dpi=160)
        plt.close(fig)
    except Exception as exc:
        print(f"WARNING: could not save history plot: {exc}", file=sys.stderr)


def save_evaluation(
    output_folder: Path,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    test_metrics: dict[str, object],
    name: str | None = None,
) -> None:
    prefix = f"{name}_" if name else ""
    labels = sorted(set(LABEL_NAMES) | set(np.unique(y_test).astype(int)) | set(np.unique(y_pred).astype(int)))
    target_names = [LABEL_NAMES.get(label, str(label)) for label in labels]

    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    confusion_df = pd.DataFrame(matrix, index=target_names, columns=target_names)
    confusion_df.to_csv(output_folder / f"{prefix}confusion_matrix.csv")

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report_dict).transpose().to_csv(output_folder / f"{prefix}classification_report.csv")

    with (output_folder / f"{prefix}test_metrics.json").open("w") as metrics_file:
        json.dump(test_metrics, metrics_file, indent=2)


def evaluate_model(model, x_test: np.ndarray, y_test: np.ndarray, batch_size: int) -> tuple[float, float, np.ndarray]:
    test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    y_proba = model.predict(x_test, batch_size=batch_size, verbose=0)
    y_pred = y_proba.argmax(axis=1)
    return float(test_loss), float(test_accuracy), y_pred


def best_epoch_from_history(history) -> int | None:
    val_losses = history.history.get("val_loss")
    if not val_losses:
        return None
    return int(np.argmin(val_losses) + 1)


def save_run_config(
    output_folder: Path,
    args: argparse.Namespace,
    feature_columns: list[str],
    class_weights: dict[int, float] | None,
    shapes: dict[str, list[int]],
) -> None:
    config = {
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "feature_columns": feature_columns,
        "label_names": LABEL_NAMES,
        "class_weights": class_weights,
        "shapes": shapes,
    }
    with (output_folder / "run_config.json").open("w") as config_file:
        json.dump(config, config_file, indent=2)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv)
        if args.epochs <= 0:
            raise ValueError("--epochs must be greater than 0")
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be greater than 0")
        if not 0 <= args.dropout < 1:
            raise ValueError("--dropout must be >= 0 and < 1")

        x_train, y_train, x_val, y_val, x_test, y_test = load_split_arrays(args.data_folder)
        window_size = int(x_train.shape[1])
        num_features = int(x_train.shape[2])
        num_classes = int(max(y_train.max(), y_val.max(), y_test.max()) + 1)
        feature_columns = load_feature_columns(args.data_folder, num_features)

        args.output_folder.mkdir(parents=True, exist_ok=True)

        random.seed(args.seed)
        np.random.seed(args.seed)
        keras = import_keras(args.keras_backend)

        class_weights = None if args.no_class_weights else compute_class_weights(y_train, num_classes)
        model = build_lstm_model(
            keras=keras,
            window_size=window_size,
            num_features=num_features,
            num_classes=num_classes,
            lstm_units=args.lstm_units,
            dense_units=args.dense_units,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
        )
        model.summary()

        checkpoint_path = args.output_folder / "best_model.keras"
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.patience,
                restore_best_weights=args.restore_best_weights,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                patience=max(2, args.patience // 2),
                factor=0.5,
                min_lr=1e-6,
            ),
        ]

        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )

        final_model_path = args.output_folder / "final_model.keras"
        model.save(final_model_path)
        save_history(args.output_folder, history)

        best_epoch = best_epoch_from_history(history)

        final_test_loss, final_test_accuracy, final_y_pred = evaluate_model(
            model,
            x_test,
            y_test,
            args.batch_size,
        )
        save_evaluation(
            output_folder=args.output_folder,
            y_test=y_test,
            y_pred=final_y_pred,
            test_metrics={
                "model": str(final_model_path),
                "epoch": len(history.history.get("loss", [])),
                "test_loss": final_test_loss,
                "test_accuracy": final_test_accuracy,
            },
            name="final",
        )

        best_model = keras.models.load_model(checkpoint_path)
        best_test_loss, best_test_accuracy, best_y_pred = evaluate_model(
            best_model,
            x_test,
            y_test,
            args.batch_size,
        )
        save_evaluation(
            output_folder=args.output_folder,
            y_test=y_test,
            y_pred=best_y_pred,
            test_metrics={
                "model": str(checkpoint_path),
                "epoch": best_epoch,
                "selection": "lowest_val_loss",
                "test_loss": best_test_loss,
                "test_accuracy": best_test_accuracy,
            },
            name="best",
        )
        save_evaluation(
            output_folder=args.output_folder,
            y_test=y_test,
            y_pred=best_y_pred,
            test_metrics={
                "model": str(checkpoint_path),
                "epoch": best_epoch,
                "selection": "lowest_val_loss",
                "test_loss": best_test_loss,
                "test_accuracy": best_test_accuracy,
            },
            name=None,
        )
        save_run_config(
            output_folder=args.output_folder,
            args=args,
            feature_columns=feature_columns,
            class_weights=class_weights,
            shapes={
                "X_train": list(x_train.shape),
                "X_val": list(x_val.shape),
                "X_test": list(x_test.shape),
            },
        )

        print(f"Saved model and reports to: {args.output_folder}")
        print(f"Best model test loss: {best_test_loss:.4f}")
        print(f"Best model test accuracy: {best_test_accuracy:.4f}")
        print(f"Final model test loss: {final_test_loss:.4f}")
        print(f"Final model test accuracy: {final_test_accuracy:.4f}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
