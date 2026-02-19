#!/usr/bin/env python3
"""Sherlock AI prototype with optional sklearn integration and pure-python fallback."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from sklearn import datasets
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    SKLEARN_AVAILABLE = False

try:
    from transformers import pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover
    TRANSFORMERS_AVAILABLE = False

logging.basicConfig(filename="sherlock_log.txt", level=logging.INFO)


@dataclass
class Vector12D:
    coords: list[float]

    def __post_init__(self) -> None:
        if len(self.coords) != 12:
            raise ValueError("Vector12D requires exactly 12 coordinates")

    def embed_data(self, data: list[float]) -> None:
        if len(data) != 12:
            raise ValueError("embed_data requires exactly 12 values")
        self.coords = [x + y for x, y in zip(self.coords, data)]


class SherlockAI:
    def __init__(self) -> None:
        self.state = Vector12D([0.0] * 12)
        self.history: list[str] = []
        self.action_history: list[dict[str, Any]] = []
        self.datasets: dict[str, dict[str, list[Any]]] = {}
        self.sound_library: dict[str, list[float]] = {}
        self.model: Any = None
        self.scaler: Any = None
        self.load_iris_dataset()

    def load_iris_dataset(self) -> None:
        if SKLEARN_AVAILABLE:
            iris = datasets.load_iris()
            self.load_dataset("iris", iris.data.tolist(), iris.target.tolist())
            return

        self.load_dataset(
            "iris",
            [
                [5.1, 3.5, 1.4, 0.2],
                [4.9, 3.0, 1.4, 0.2],
                [7.0, 3.2, 4.7, 1.4],
                [6.4, 3.2, 4.5, 1.5],
                [6.3, 3.3, 6.0, 2.5],
                [5.8, 2.7, 5.1, 1.9],
            ],
            [0, 0, 1, 1, 2, 2],
        )

    def load_dataset(self, name: str, data: list[list[float]], target: list[int]) -> None:
        self.datasets[name] = {
            "data": [[float(value) for value in row] for row in data],
            "target": [int(value) for value in target],
        }

    def preprocess_data(self, name: str) -> None:
        dataset = self.datasets[name]
        features = dataset["data"]

        if SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()
            transformed = self.scaler.fit_transform(features)
            dataset["data"] = transformed.tolist()
            return

        cols = len(features[0])
        means = [sum(row[i] for row in features) / len(features) for i in range(cols)]
        stds = []
        for i in range(cols):
            variance = sum((row[i] - means[i]) ** 2 for row in features) / len(features)
            stds.append(math.sqrt(variance) or 1.0)

        dataset["data"] = [
            [(row[i] - means[i]) / stds[i] for i in range(cols)]
            for row in features
        ]

    def train_model(self, name: str, random_state: int = 42) -> dict[str, Any]:
        dataset = self.datasets[name]
        x = dataset["data"]
        y = dataset["target"]

        if SKLEARN_AVAILABLE:
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=0.3,
                random_state=random_state,
                stratify=y,
            )
            self.model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=random_state)
            self.model.fit(x_train, y_train)
            y_pred = self.model.predict(x_test)

            accuracy = sum(int(p == t) for p, t in zip(y_pred, y_test)) / len(y_test)
            return {
                "accuracy": accuracy,
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "classification_report": classification_report(y_test, y_pred, output_dict=True),
            }

        classes = sorted(set(y))
        centroids: dict[int, list[float]] = {}
        for label in classes:
            rows = [row for row, target in zip(x, y) if target == label]
            cols = len(rows[0])
            centroids[label] = [sum(r[i] for r in rows) / len(rows) for i in range(cols)]

        predictions: list[int] = []
        for row in x:
            predictions.append(min(classes, key=lambda c: self._distance(row, centroids[c])))

        matrix = [[0 for _ in classes] for _ in classes]
        class_to_idx = {label: idx for idx, label in enumerate(classes)}
        for truth, pred in zip(y, predictions):
            matrix[class_to_idx[truth]][class_to_idx[pred]] += 1

        accuracy = sum(int(p == t) for p, t in zip(predictions, y)) / len(y)
        return {
            "accuracy": accuracy,
            "confusion_matrix": matrix,
            "classification_report": {"note": "fallback mode without sklearn"},
        }

    def _distance(self, lhs: list[float], rhs: list[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(lhs, rhs)))

    def add_sound(self, sound_name: str, sound_data: list[float]) -> None:
        self.sound_library[sound_name] = sound_data

    def ping_sound(self, sound_name: str) -> list[float] | None:
        if sound_name not in self.sound_library:
            return None
        return self.simulate_echo(self.sound_library[sound_name])

    def simulate_echo(self, sound_data: list[float]) -> list[float]:
        return sound_data * 2

    def run_nlp_demo(self, text: str) -> str:
        if not TRANSFORMERS_AVAILABLE:
            return "transformers pipeline unavailable"
        return str(pipeline("sentiment-analysis")(text))


if __name__ == "__main__":
    ai = SherlockAI()
    ai.preprocess_data("iris")
    results = ai.train_model("iris")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"Confusion Matrix: {results['confusion_matrix']}")
    print(f"Log file: {Path('sherlock_log.txt').resolve()}")
