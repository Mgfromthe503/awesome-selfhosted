"""Grid-search training script for a CNN crime-classification model.

Usage:
  python crime_model_grid_search.py --data crime_data.npy --labels crime_labels.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam

USE_SCIKERAS = True
try:
    from scikeras.wrappers import KerasClassifier
except ImportError:  # fallback for older TensorFlow environments
    USE_SCIKERAS = False
    from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CNN hyperparameter search for crime data")
    parser.add_argument("--data", type=Path, default=Path("crime_data.npy"), help="Path to input feature array")
    parser.add_argument("--labels", type=Path, default=Path("crime_labels.npy"), help="Path to label array")
    parser.add_argument("--train-size", type=float, default=0.8, help="Fraction of samples for training")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per GridSearchCV candidate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--cv", type=int, default=3, help="Number of CV folds")
    return parser.parse_args()


def _prepare_images(data: np.ndarray) -> np.ndarray:
    """Ensure input arrays are 4D float32 tensors for Conv2D."""
    data = data.astype("float32")

    if data.ndim == 3:  # (N, H, W) -> (N, H, W, 1)
        data = np.expand_dims(data, axis=-1)

    if data.ndim != 4:
        raise ValueError(f"Expected 3D or 4D image tensor, got shape {data.shape}")

    if data.shape[-1] != 1:
        raise ValueError(f"Expected grayscale images with channel size 1, got {data.shape[-1]}")

    return data


def create_model(learning_rate: float = 0.001, num_filters: int = 32, num_neurons: int = 64) -> Sequential:
    model = Sequential(
        [
            Conv2D(num_filters, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(num_filters * 2, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(num_neurons, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


def main() -> None:
    args = parse_args()

    crime_data = np.load(args.data)
    labels = np.load(args.labels)

    crime_data = _prepare_images(crime_data)
    labels = labels.astype("float32")

    train_data, _, train_labels, _ = train_test_split(
        crime_data,
        labels,
        train_size=args.train_size,
        random_state=42,
        stratify=labels,
    )

    param_grid = {
        "learning_rate": [0.001, 0.01, 0.1],
        "num_filters": [32, 64, 128],
        "num_neurons": [64, 128, 256],
    }

    if USE_SCIKERAS:
        model = KerasClassifier(
            model=create_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=0,
        )
    else:
        model = KerasClassifier(
            build_fn=create_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=0,
        )

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=args.cv, n_jobs=1)
    grid_search.fit(train_data, train_labels)

    print("Best hyperparameters:", grid_search.best_params_)
    print("Best CV score:", grid_search.best_score_)


if __name__ == "__main__":
    main()
