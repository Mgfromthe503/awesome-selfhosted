"""Crime-solving prediction pipeline inspired by the Sherlock prototype.

Dependency-tolerant implementation:
- Uses pandas/scikit-learn when available.
- Falls back to stdlib CSV loading and a simple threshold model when unavailable.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    HAS_SKLEARN = True
except Exception:  # pragma: no cover
    RandomForestClassifier = None  # type: ignore
    accuracy_score = train_test_split = None  # type: ignore
    HAS_SKLEARN = False

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter

    HAS_QISKIT = True
except Exception:  # pragma: no cover
    QuantumCircuit = Parameter = None  # type: ignore
    HAS_QISKIT = False


@dataclass
class SimpleThresholdModel:
    threshold: float

    def predict(self, rows: list[dict], features: Sequence[str]) -> list[int]:
        predictions: list[int] = []
        for row in rows:
            score = sum(float(row[name]) for name in features)
            predictions.append(1 if score >= self.threshold else 0)
        return predictions


@dataclass
class TrainingResult:
    model: object
    feature_names: list[str]
    test_accuracy: float
    backend: str


def load_dataset(csv_path: str | Path):
    """Load CSV data into pandas DataFrame (if available) or list[dict]."""
    if pd is not None:
        return pd.read_csv(csv_path).dropna().reset_index(drop=True)

    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if all(value != "" for value in row.values())]
    return rows


def _validate_columns(records, features: Sequence[str], target: str):
    if pd is not None and hasattr(records, "columns"):
        existing = set(records.columns)
    else:
        existing = set(records[0].keys()) if records else set()
    missing = [column for column in [*features, target] if column not in existing]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _train_fallback(records: list[dict], features: Sequence[str], target: str) -> TrainingResult:
    positives = [row for row in records if int(float(row[target])) == 1]
    negatives = [row for row in records if int(float(row[target])) == 0]

    pos_mean = sum(sum(float(r[f]) for f in features) for r in positives) / max(len(positives), 1)
    neg_mean = sum(sum(float(r[f]) for f in features) for r in negatives) / max(len(negatives), 1)
    threshold = (pos_mean + neg_mean) / 2.0

    model = SimpleThresholdModel(threshold=threshold)
    preds = model.predict(records, features)
    gold = [int(float(row[target])) for row in records]
    accuracy = sum(int(a == b) for a, b in zip(preds, gold)) / max(len(gold), 1)
    return TrainingResult(model=model, feature_names=list(features), test_accuracy=accuracy, backend="fallback")


def train_solver_model(crime_data, *, features: Sequence[str], target: str = "solved", random_state: int = 42) -> TrainingResult:
    _validate_columns(crime_data, features, target)

    if HAS_SKLEARN and pd is not None and hasattr(crime_data, "columns"):
        x = crime_data[list(features)]
        y = crime_data[target]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=random_state, stratify=y if y.nunique() > 1 else None
        )
        model = RandomForestClassifier(n_estimators=200, random_state=random_state)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracy = float(accuracy_score(y_test, predictions))
        return TrainingResult(model=model, feature_names=list(features), test_accuracy=accuracy, backend="sklearn")

    records = crime_data if isinstance(crime_data, list) else crime_data.to_dict(orient="records")
    return _train_fallback(records, features, target)


def predict_unsolved_cases(model: object, unsolved_data, *, features: Sequence[str]) -> list[int]:
    if hasattr(model, "predict") and pd is not None and hasattr(unsolved_data, "columns") and HAS_SKLEARN:
        return model.predict(unsolved_data[list(features)]).astype(int).tolist()

    rows = unsolved_data if isinstance(unsolved_data, list) else unsolved_data.to_dict(orient="records")
    return model.predict(rows, features)  # type: ignore[arg-type]


def format_case_predictions(predictions: Iterable[int]) -> list[str]:
    return [f"Case {i} is likely to be solved." if p == 1 else f"Case {i} is unlikely to be solved." for i, p in enumerate(predictions)]


def build_quantum_scaffold(num_qubits: int = 2):
    if not HAS_QISKIT:
        return None
    theta = Parameter("θ")
    circuit = QuantumCircuit(num_qubits)
    for qubit in range(num_qubits):
        circuit.h(qubit)
        circuit.p(theta, qubit)
    circuit.measure_all()
    return circuit


def run_pipeline(crime_csv: str | Path, unsolved_csv: str | Path, *, features: Sequence[str], target: str = "solved") -> tuple[TrainingResult, list[str]]:
    crime_data = load_dataset(crime_csv)
    training = train_solver_model(crime_data, features=features, target=target)
    unsolved = load_dataset(unsolved_csv)
    raw_predictions = predict_unsolved_cases(training.model, unsolved, features=training.feature_names)
    return training, format_case_predictions(raw_predictions)
