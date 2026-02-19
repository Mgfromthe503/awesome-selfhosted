"""Train baseline and optional CNN models for crime prediction.

This script is a productionized version of the provided notebook-style code:
- validates input files
- makes feature/target explicit
- performs a tabular RandomForest baseline
- optionally performs CNN hyperparameter tuning with GridSearchCV when TensorFlow is available
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split

try:
    from scikeras.wrappers import KerasClassifier
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
    from tensorflow.keras.optimizers import Adam

    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crime model training pipeline")
    parser.add_argument("--dataset", default="dataset.csv", help="Path to CSV dataset")
    parser.add_argument("--crime-data", default="crime_data.npy", help="Path to image-like features .npy")
    parser.add_argument("--crime-labels", default="crime_labels.npy", help="Path to labels .npy")
    parser.add_argument("--target", required=True, help="Target column in dataset.csv")
    parser.add_argument(
        "--features",
        nargs="+",
        required=True,
        help="Feature columns from dataset.csv for RandomForest",
    )
    parser.add_argument(
        "--skip-cnn",
        action="store_true",
        help="Skip CNN hyperparameter search even if TensorFlow is installed",
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame, features: Sequence[str], target: str) -> None:
    missing = [col for col in [*features, target] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")


def create_model(learning_rate: float = 0.001, num_filters: int = 32, num_neurons: int = 64):
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


def run_cnn_grid_search(crime_data: np.ndarray, labels: np.ndarray) -> None:
    train_size = min(800, len(crime_data))
    train_data = crime_data[:train_size]
    train_labels = labels[:train_size]

    if train_data.ndim == 3:
        train_data = np.expand_dims(train_data, axis=-1)

    param_grid = {
        "model__learning_rate": [0.001, 0.01],
        "model__num_filters": [32, 64],
        "model__num_neurons": [64, 128],
    }

    model = KerasClassifier(model=create_model, epochs=10, batch_size=32, verbose=0)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    grid_search.fit(train_data, train_labels)

    print("Best CNN params:", grid_search.best_params_)
    print("Best CNN CV score:", grid_search.best_score_)


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset)
    crime_data_path = Path(args.crime_data)
    crime_labels_path = Path(args.crime_labels)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    validate_columns(df, args.features, args.target)

    x = df[args.features]
    y = df[args.target]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(x_train, y_train)

    predictions = rf.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"RandomForest accuracy: {accuracy:.4f}")

    if args.skip_cnn:
        return

    if not TENSORFLOW_AVAILABLE:
        print("Skipping CNN tuning: TensorFlow/scikeras not available.")
        return

    if not crime_data_path.exists() or not crime_labels_path.exists():
        print("Skipping CNN tuning: crime_data.npy and/or crime_labels.npy missing.")
        return

    crime_data = np.load(crime_data_path)
    labels = np.load(crime_labels_path)
    run_cnn_grid_search(crime_data, labels)


if __name__ == "__main__":
    main()
