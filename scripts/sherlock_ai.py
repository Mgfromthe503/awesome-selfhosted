#!/usr/bin/env python3
"""Lightweight Sherlock AI prototype with 12D state and dataset utilities."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logging.basicConfig(
    filename="sherlock_log.txt",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


@dataclass
class Vector12D:
    """A fixed-length 12-dimensional vector container."""

    coords: list[float]

    def __post_init__(self) -> None:
        if len(self.coords) != 12:
            msg = "Vector12D requires exactly 12 coordinates"
            raise ValueError(msg)

    def embed_data(self, data: float | list[float]) -> None:
        """Embed scalar or 12D list into this vector."""
        if isinstance(data, (int, float)):
            self.coords = [x + float(data) for x in self.coords]
            return

        if len(data) != 12:
            msg = "List embedding requires exactly 12 values"
            raise ValueError(msg)

        self.coords = [x + float(y) for x, y in zip(self.coords, data)]


class SherlockAI:
    """Small, dependency-free AI playground with basic classification."""

    def __init__(self) -> None:
        self.state = Vector12D([0.0] * 12)
        self.history: list[str] = []
        self.action_history: list[dict[str, Any]] = []
        self.datasets: dict[str, dict[str, list[Any]]] = {}
        self.sound_library: dict[str, list[float]] = {}

        self.load_dataset("iris", self._default_iris_dataset())
        logging.info("SherlockAI initialized with datasets: %s", list(self.datasets))

    def _default_iris_dataset(self) -> dict[str, list[Any]]:
        """Return a tiny iris-like dataset for local/offline execution."""
        return {
            "data": [
                [5.1, 3.5, 1.4, 0.2],
                [4.9, 3.0, 1.4, 0.2],
                [7.0, 3.2, 4.7, 1.4],
                [6.4, 3.2, 4.5, 1.5],
                [6.3, 3.3, 6.0, 2.5],
                [5.8, 2.7, 5.1, 1.9],
            ],
            "target": [0, 0, 1, 1, 2, 2],
        }

    def load_dataset(self, name: str, data: dict[str, list[Any]]) -> None:
        """Load a dataset into memory."""
        if "data" not in data or "target" not in data:
            msg = "Dataset must include 'data' and 'target' keys"
            raise ValueError(msg)
        self.datasets[name] = data
        self.action_history.append({"action": "load_dataset", "dataset": name})

    def preprocess_data(self, name: str) -> dict[str, list[Any]]:
        """Normalize dataset features to zero mean and unit variance."""
        dataset = self.datasets[name]
        data = dataset["data"]
        cols = len(data[0])

        means = [sum(row[i] for row in data) / len(data) for i in range(cols)]
        stds = []
        for i in range(cols):
            variance = sum((row[i] - means[i]) ** 2 for row in data) / len(data)
            stds.append(variance**0.5 or 1.0)

        normalized = [
            [(row[i] - means[i]) / stds[i] for i in range(cols)]
            for row in data
        ]

        processed = {"data": normalized, "target": copy.deepcopy(dataset["target"]) }
        self.datasets[f"{name}_preprocessed"] = processed
        self.action_history.append({"action": "preprocess_data", "dataset": name})
        return processed

    def train_centroid_classifier(self, name: str) -> dict[int, list[float]]:
        """Train a nearest-centroid classifier and return class centroids."""
        dataset = self.datasets[name]
        class_rows: dict[int, list[list[float]]] = {}
        for features, label in zip(dataset["data"], dataset["target"]):
            class_rows.setdefault(int(label), []).append(features)

        centroids: dict[int, list[float]] = {}
        for label, rows in class_rows.items():
            cols = len(rows[0])
            centroids[label] = [sum(r[i] for r in rows) / len(rows) for i in range(cols)]

        self.action_history.append({"action": "train", "dataset": name})
        return centroids

    def predict(self, centroids: dict[int, list[float]], sample: list[float]) -> int:
        """Predict class by nearest centroid with euclidean distance."""
        def distance(a: list[float], b: list[float]) -> float:
            return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

        return min(centroids, key=lambda label: distance(centroids[label], sample))

    def evaluate(self, name: str) -> dict[str, Any]:
        """Evaluate centroid classifier on the same dataset for smoke checks."""
        centroids = self.train_centroid_classifier(name)
        dataset = self.datasets[name]
        labels = sorted(set(dataset["target"]))
        index = {label: idx for idx, label in enumerate(labels)}

        matrix = [[0 for _ in labels] for _ in labels]
        correct = 0
        for sample, truth in zip(dataset["data"], dataset["target"]):
            pred = self.predict(centroids, sample)
            matrix[index[int(truth)]][index[pred]] += 1
            if pred == truth:
                correct += 1

        accuracy = correct / len(dataset["target"])
        report = {"accuracy": accuracy, "labels": labels, "confusion_matrix": matrix}
        self.action_history.append({"action": "evaluate", "dataset": name, "accuracy": accuracy})
        return report

    def add_sound(self, sound_name: str, sound_data: list[float]) -> None:
        self.sound_library[sound_name] = sound_data

    def ping_sound(self, sound_name: str) -> list[float] | None:
        if sound_name not in self.sound_library:
            return None
        return self.simulate_echo(self.sound_library[sound_name])

    def simulate_echo(self, sound_data: list[float], decay: float = 0.6) -> list[float]:
        return [sample * decay for sample in sound_data]


if __name__ == "__main__":
    ai = SherlockAI()
    processed_name = "iris_preprocessed"
    ai.preprocess_data("iris")
    results = ai.evaluate(processed_name)
    print(f"Accuracy: {results['accuracy']:.2f}")
    print(f"Confusion Matrix: {results['confusion_matrix']}")
    print(f"Log file: {Path('sherlock_log.txt').resolve()}")
