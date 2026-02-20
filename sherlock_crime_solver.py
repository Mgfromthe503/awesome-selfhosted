"""Sherlock crime-solving pipeline with optional ML/quantum dependencies."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple


@dataclass
class Sherlock:
    username: str
    password: str
    family_members: Sequence[str]
    user_mood: int = 0
    user_charity: str = ""
    user_age: Optional[int] = None
    user_threat_level: int = 0
    user_scholarship: bool = False
    authenticated_users: list[str] = field(init=False)

    def __post_init__(self) -> None:
        self.authenticated_users = [self.username, *self.family_members]


def _to_number(value: str) -> float:
    if value.isdigit():
        return float(int(value))
    return float(value)


def load_and_preprocess_crime_data(path: str, features: Sequence[str], target: str) -> list[dict[str, float]]:
    """Load, validate, and clean rows from CSV.

    Rows with missing required values are dropped.
    """

    required = [*features, target]
    rows: list[dict[str, float]] = []

    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")

        missing_cols = [col for col in required if col not in reader.fieldnames]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        for row in reader:
            if any(row.get(col, "") in ("", None) for col in required):
                continue
            rows.append({col: _to_number(row[col]) for col in required})

    return rows


def train_crime_model(
    data: list[dict[str, float]],
    features: Sequence[str],
    target: str,
    *,
    random_state: int = 42,
    test_size: float = 0.2,
) -> Tuple[object, float]:
    """Train and evaluate classifier using sklearn if available."""

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
    except Exception as exc:
        raise RuntimeError("scikit-learn is not installed in this environment") from exc

    x = [[row[f] for f in features] for row in data]
    y = [int(row[target]) for row in data]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    model = RandomForestClassifier(random_state=random_state)
    model.fit(x_train, y_train)
    score = float(model.score(x_test, y_test))
    return model, score


def predict_unsolved_cases(model: object, unsolved_rows: list[dict[str, float]], features: Sequence[str]) -> list[int]:
    matrix = [[row[f] for f in features] for row in unsolved_rows]
    predictions = model.predict(matrix)
    return [int(p) for p in predictions]


def load_unsolved_data(path: str, features: Sequence[str]) -> list[dict[str, float]]:
    required = list(features)
    rows: list[dict[str, float]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")

        missing_cols = [col for col in required if col not in reader.fieldnames]
        if missing_cols:
            raise ValueError(f"Missing required columns in unsolved set: {missing_cols}")

        for row in reader:
            if any(row.get(col, "") in ("", None) for col in required):
                continue
            rows.append({col: _to_number(row[col]) for col in required})
    return rows


def build_quantum_circuit(n_qubits: int = 2):
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter
    except Exception as exc:
        raise RuntimeError("Qiskit is not installed in this environment") from exc

    theta = Parameter("θ")
    qc = QuantumCircuit(n_qubits)
    for idx in range(n_qubits):
        qc.h(idx)
        qc.p(theta, idx)
    qc.measure_all()
    return qc


def run_pipeline(crime_csv: str, unsolved_csv: str, *, features: Sequence[str], target: str = "solved"):
    crime_data = load_and_preprocess_crime_data(crime_csv, features, target)
    model, score = train_crime_model(crime_data, features, target)
    unsolved = load_unsolved_data(unsolved_csv, features)
    predictions = predict_unsolved_cases(model, unsolved, features)
    return score, predictions


def format_prediction_report(predictions: Iterable[int]) -> list[str]:
    return [
        f"Case {i} is likely to be solved." if int(pred) == 1 else f"Case {i} is unlikely to be solved."
        for i, pred in enumerate(predictions)
    ]
