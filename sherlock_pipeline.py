"""End-to-end Sherlock crime-model training and inference pipeline.

This module turns the ad-hoc notebook-style script into reusable functions that can
be tested and automated in CI.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
except ModuleNotFoundError:  # pragma: no cover - optional runtime deps
    pd = None
    ColumnTransformer = RandomForestClassifier = Pipeline = OneHotEncoder = None
    train_test_split = accuracy_score = precision_score = recall_score = f1_score = None

LOGGER = logging.getLogger(__name__)

FEATURES = [
    "age",
    "gender",
    "race",
    "location",
    "time_of_day",
    "weapon",
    "motive",
    "victim_age",
    "victim_gender",
    "crime_type",
]
TARGET = "solved"


def _require_ml_dependencies() -> None:
    if pd is None or Pipeline is None:
        raise ModuleNotFoundError("pandas and scikit-learn are required for the Sherlock pipeline")


CATEGORICAL_FEATURES = [
    "gender",
    "race",
    "location",
    "time_of_day",
    "weapon",
    "motive",
    "victim_gender",
    "crime_type",
]
NUMERIC_FEATURES = ["age", "victim_age"]


@dataclass(frozen=True)
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float


def _build_model(random_state: int = 42) -> Pipeline:
    _require_ml_dependencies()
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )
    classifier = RandomForestClassifier(random_state=random_state, n_estimators=200)
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def load_and_clean_data(path: str | Path) -> "pd.DataFrame":
    _require_ml_dependencies()
    df = pd.read_csv(path).dropna().copy()
    missing = [col for col in FEATURES + [TARGET] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def train_model(crime_data: "pd.DataFrame", random_state: int = 42) -> tuple[Pipeline, ModelMetrics]:
    _require_ml_dependencies()
    X_train, X_test, y_train, y_test = train_test_split(
        crime_data[FEATURES],
        crime_data[TARGET],
        test_size=0.2,
        random_state=random_state,
        stratify=crime_data[TARGET],
    )
    model = _build_model(random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = ModelMetrics(
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, zero_division=0),
        recall=recall_score(y_test, y_pred, zero_division=0),
        f1=f1_score(y_test, y_pred, zero_division=0),
    )
    LOGGER.info("Model metrics: %s", metrics)
    return model, metrics


def save_model(model: Pipeline, path: str | Path) -> None:
    _require_ml_dependencies()
    with Path(path).open("wb") as file:
        pickle.dump(model, file)


def load_model(path: str | Path) -> Pipeline:
    _require_ml_dependencies()
    with Path(path).open("rb") as file:
        return pickle.load(file)


def predict_unsolved(model: Pipeline, unsolved_data: "pd.DataFrame") -> "pd.Series":
    _require_ml_dependencies()
    missing = [col for col in FEATURES if col not in unsolved_data.columns]
    if missing:
        raise ValueError(f"Missing required columns for inference: {missing}")
    return pd.Series(model.predict(unsolved_data[FEATURES]), name="prediction")


def save_prediction_report(predictions: "pd.Series", path: str | Path) -> None:
    _require_ml_dependencies()
    pd.DataFrame({"Predictions": predictions}).to_csv(path, index=False)


def run_quantum_demo() -> dict[str, Any]:
    """Run a tiny quantum simulation when qiskit is available.

    Returns a dict with status information so automation can report outcomes.
    """

    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import Aer
    except Exception as exc:  # pragma: no cover - optional dependency
        return {"status": "skipped", "reason": f"qiskit unavailable: {exc}"}

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    backend = Aer.get_backend("qasm_simulator")
    transpiled = transpile(qc, backend)
    job = backend.run(transpiled, shots=1000)
    counts = job.result().get_counts()
    return {"status": "ok", "counts": counts}
