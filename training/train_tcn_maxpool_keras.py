from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

from train_lstm_maxpool_keras import best_epoch_from_history
from train_lstm_maxpool_keras import compute_class_weights
from train_lstm_maxpool_keras import evaluate_model
from train_lstm_maxpool_keras import import_keras
from train_lstm_maxpool_keras import load_feature_columns
from train_lstm_maxpool_keras import load_split_arrays
from train_lstm_maxpool_keras import save_evaluation
from train_lstm_maxpool_keras import save_history
from train_lstm_maxpool_keras import save_run_config


def parse_dilations(value: str) -> list[int]:
    try:
        dilations = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--dilations must be comma-separated integers") from exc
    if not dilations or any(dilation <= 0 for dilation in dilations):
        raise argparse.ArgumentTypeError("--dilations must contain positive integers")
    return dilations


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a Keras TCN classifier with global max pooling from "
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
        default=Path("models/keras_tcn_ceb_tf_smooth_nocw_lr1e4_b128_p80_f32_d20_maxpool"),
        help=(
            "Folder where model and reports will be saved. Default: "
            "models/keras_tcn_ceb_tf_smooth_nocw_lr1e4_b128_p80_f32_d20_maxpool."
        ),
    )
    parser.add_argument(
        "--keras-backend",
        choices=("jax", "torch", "tensorflow"),
        default="tensorflow",
        help="Keras backend to use. Default: tensorflow.",
    )
    parser.add_argument("--epochs", type=int, default=1000, help="Maximum epochs. Default: 1000.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size. Default: 128.")
    parser.add_argument("--filters", type=int, default=32, help="TCN convolution filters. Default: 32.")
    parser.add_argument("--kernel-size", type=int, default=3, help="TCN kernel size. Default: 3.")
    parser.add_argument(
        "--block-depth",
        type=int,
        choices=(1, 2),
        default=2,
        help=(
            "Convolution layers per residual TCN block. Use 1 for a simpler "
            "TCN. Default: 2."
        ),
    )
    parser.add_argument(
        "--dilations",
        type=parse_dilations,
        default=parse_dilations("1,2,4,8,16"),
        help="Comma-separated dilation rates. Default: 1,2,4,8,16.",
    )
    parser.add_argument("--stacks", type=int, default=1, help="Number of dilation stacks. Default: 1.")
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
        help="Random seed. Default: 42.",
    )
    return parser.parse_args(argv)


def residual_tcn_block(
    keras,
    x,
    filters: int,
    kernel_size: int,
    dilation_rate: int,
    block_depth: int,
    dropout: float,
    name: str,
):
    shortcut = x

    y = keras.layers.Conv1D(
        filters,
        kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        kernel_initializer="he_normal",
        name=f"{name}_conv1",
    )(x)
    y = keras.layers.LayerNormalization(name=f"{name}_norm1")(y)
    y = keras.layers.Activation("relu", name=f"{name}_relu1")(y)
    y = keras.layers.Dropout(dropout, name=f"{name}_dropout1")(y)

    if block_depth == 2:
        y = keras.layers.Conv1D(
            filters,
            kernel_size,
            padding="causal",
            dilation_rate=dilation_rate,
            kernel_initializer="he_normal",
            name=f"{name}_conv2",
        )(y)
        y = keras.layers.LayerNormalization(name=f"{name}_norm2")(y)
        y = keras.layers.Activation("relu", name=f"{name}_relu2")(y)
        y = keras.layers.Dropout(dropout, name=f"{name}_dropout2")(y)

    if shortcut.shape[-1] != filters:
        shortcut = keras.layers.Conv1D(
            filters,
            1,
            padding="same",
            kernel_initializer="he_normal",
            name=f"{name}_residual_projection",
        )(shortcut)

    x = keras.layers.Add(name=f"{name}_add")([shortcut, y])
    return keras.layers.Activation("relu", name=f"{name}_out")(x)


def build_tcn_model(
    keras,
    window_size: int,
    num_features: int,
    num_classes: int,
    filters: int,
    kernel_size: int,
    dilations: list[int],
    stacks: int,
    block_depth: int,
    dense_units: int,
    dropout: float,
    learning_rate: float,
):
    inputs = keras.Input(shape=(window_size, num_features), name="flight_window")
    x = inputs
    for stack_index in range(stacks):
        for dilation in dilations:
            block_name = f"tcn_s{stack_index + 1}_d{dilation}"
            x = residual_tcn_block(
                keras=keras,
                x=x,
                filters=filters,
                kernel_size=kernel_size,
                dilation_rate=dilation,
                block_depth=block_depth,
                dropout=dropout,
                name=block_name,
            )

    x = keras.layers.GlobalMaxPooling1D(name="global_max_pooling")(x)
    x = keras.layers.Dense(dense_units, activation="relu", name="dense")(x)
    x = keras.layers.Dropout(dropout, name="dropout_after_dense")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="class")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="flight_attack_tcn_maxpool")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv)
        if args.epochs <= 0:
            raise ValueError("--epochs must be greater than 0")
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be greater than 0")
        if args.filters <= 0:
            raise ValueError("--filters must be greater than 0")
        if args.kernel_size <= 0:
            raise ValueError("--kernel-size must be greater than 0")
        if args.block_depth not in (1, 2):
            raise ValueError("--block-depth must be 1 or 2")
        if args.stacks <= 0:
            raise ValueError("--stacks must be greater than 0")
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
        model = build_tcn_model(
            keras=keras,
            window_size=window_size,
            num_features=num_features,
            num_classes=num_classes,
            filters=args.filters,
            kernel_size=args.kernel_size,
            dilations=args.dilations,
            stacks=args.stacks,
            block_depth=args.block_depth,
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
        best_metrics = {
            "model": str(checkpoint_path),
            "epoch": best_epoch,
            "selection": "lowest_val_loss",
            "test_loss": best_test_loss,
            "test_accuracy": best_test_accuracy,
        }
        save_evaluation(
            output_folder=args.output_folder,
            y_test=y_test,
            y_pred=best_y_pred,
            test_metrics=best_metrics,
            name="best",
        )
        save_evaluation(
            output_folder=args.output_folder,
            y_test=y_test,
            y_pred=best_y_pred,
            test_metrics=best_metrics,
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
