"""Crime solving prediction pipeline.

A dependency-light replacement for the original prototype. It avoids unavailable
packages while keeping a clean API for automation and testing.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


Row = dict[str, float | int]


@dataclass(frozen=True)
class SimpleCrimeModel:
    """Small deterministic classifier based on class centroid distance."""

    feature_columns: list[str]
    unsolved_centroid: list[float]
    solved_centroid: list[float]

    def predict_one(self, row: Row) -> int:
        values = [float(row[name]) for name in self.feature_columns]
        d_unsolved = _euclidean(values, self.unsolved_centroid)
        d_solved = _euclidean(values, self.solved_centroid)
        return 1 if d_solved <= d_unsolved else 0

    def predict(self, rows: Sequence[Row]) -> list[int]:
        return [self.predict_one(row) for row in rows]


def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def load_dataset(path: str | Path) -> list[Row]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return [
            {k: _coerce(v) for k, v in row.items()}
            for row in csv.DictReader(f)
        ]


def _coerce(value: str):
    try:
        if "." in value:
            return float(value)
        return int(value)
    except Exception:
        return value


def preprocess_data(rows: Sequence[Row], feature_columns: Sequence[str], target_column: str | None = None) -> list[Row]:
    required = list(feature_columns)
    if target_column:
        required.append(target_column)

    clean: list[Row] = []
    for row in rows:
        if all(col in row and row[col] not in (None, "") for col in required):
            clean.append(row)

    if not clean:
        raise ValueError("No rows available after preprocessing")

    return clean


def train_model(rows: Sequence[Row], feature_columns: Sequence[str], target_column: str = "solved") -> tuple[SimpleCrimeModel, str]:
    clean = preprocess_data(rows, feature_columns, target_column)

    solved = [r for r in clean if int(r[target_column]) == 1]
    unsolved = [r for r in clean if int(r[target_column]) == 0]
    if not solved or not unsolved:
        raise ValueError("Training data must include both solved=1 and solved=0 rows")

    solved_centroid = _centroid(solved, feature_columns)
    unsolved_centroid = _centroid(unsolved, feature_columns)
    model = SimpleCrimeModel(list(feature_columns), unsolved_centroid, solved_centroid)

    correct = sum(1 for row in clean if model.predict_one(row) == int(row[target_column]))
    accuracy = correct / len(clean)
    report = f"training_accuracy={accuracy:.2f}; samples={len(clean)}"
    return model, report


def _centroid(rows: Sequence[Row], feature_columns: Sequence[str]) -> list[float]:
    return [sum(float(r[name]) for r in rows) / len(rows) for name in feature_columns]


def predict_unsolved_cases(model: SimpleCrimeModel, rows: Sequence[Row]) -> list[int]:
    clean = preprocess_data(rows, model.feature_columns)
    return model.predict(clean)


def format_predictions(predictions: Iterable[int]) -> list[str]:
    return [
        f"Case {i} is likely to be solved." if pred == 1 else f"Case {i} is unlikely to be solved."
        for i, pred in enumerate(predictions)
    ]


def run_pipeline(training_csv: str | Path, unsolved_csv: str | Path, feature_columns: Sequence[str], target_column: str = "solved") -> tuple[str, list[str]]:
    training = load_dataset(training_csv)
    model, report = train_model(training, feature_columns, target_column)
    unsolved = load_dataset(unsolved_csv)
    predictions = predict_unsolved_cases(model, unsolved)
    return report, format_predictions(predictions)
