"""Utilities for estimating Quantum Lucidity Potential (QLP)."""

from __future__ import annotations

from typing import Iterable, Sequence

try:
    from sklearn.ensemble import RandomForestRegressor  # type: ignore
except Exception:  # pragma: no cover - fallback path for minimal environments
    RandomForestRegressor = None


def _build_training_data() -> tuple[list[list[float]], list[float]]:
    """Create a small synthetic training dataset for the baseline model."""
    x = [
        [0.10, 0.20, 0.30, 0.15, 10.0, 0.40, 1.0],
        [0.25, 0.15, 0.40, 0.30, 12.0, 0.55, 1.2],
        [0.50, 0.35, 0.45, 0.40, 15.0, 0.65, 1.8],
        [0.65, 0.40, 0.55, 0.50, 18.0, 0.75, 2.3],
        [0.80, 0.60, 0.70, 0.65, 22.0, 0.85, 2.8],
    ]
    y = [0.18, 0.31, 0.52, 0.68, 0.86]
    return x, y


def _mean(values: Iterable[float]) -> float:
    values_list = list(values)
    return sum(values_list) / len(values_list)


def _fallback_predict(features: list[float]) -> float:
    """Deterministic fallback when scikit-learn is unavailable."""
    train_x, train_y = _build_training_data()
    distances = []
    for row, target in zip(train_x, train_y):
        dist = sum((a - b) ** 2 for a, b in zip(row, features)) ** 0.5
        distances.append((dist, target))

    distances.sort(key=lambda item: item[0])
    nearest_targets = [target for _, target in distances[:3]]
    return _mean(nearest_targets)


def machine_learning_based_qlp(features: Sequence[float] | Iterable[float]) -> float:
    """Estimate QLP using a RandomForestRegressor baseline model."""
    features_list = [float(value) for value in features]
    if len(features_list) != 7:
        raise ValueError("features must contain exactly 7 numeric values")

    training_x, training_y = _build_training_data()

    if RandomForestRegressor is None:
        return _fallback_predict(features_list)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(training_x, training_y)

    prediction = model.predict([features_list])
    return float(prediction[0])


def quantum_lucidity(
    Ce: float,
    Cw: float,
    Ca: float,
    Cf: float,
    NB: float,
    AS: float,
    T_QL: float,
) -> float:
    """Convenience wrapper for computing QLP from individual feature values."""
    features = [Ce, Cw, Ca, Cf, NB, AS, T_QL]
    return machine_learning_based_qlp(features)
