"""Alpha Mind Gamma model using open-source numerical libraries.

This module provides a practical replacement for speculative placeholders by using
NumPy/Pandas data processing and a small, explainable linear model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass
class AlphaMindGammaModel:
    """Simple regression model for QLP-style scoring.

    Features are derived from numeric inputs and spatial coordinates.
    """

    coefficients: np.ndarray | None = None

    @staticmethod
    def _validate_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        required = [
            "x",
            "y",
            "z",
            "fire",
            "water",
            "air",
            "earth",
            "numerology",
            "frequency",
            "light_code",
        ]
        self._validate_columns(df, required)

        features = pd.DataFrame(index=df.index)
        features["bias"] = 1.0

        coords = df[["x", "y", "z"]].to_numpy(dtype=float)
        features["spatial_norm"] = np.linalg.norm(coords, axis=1)

        features["elemental_sum"] = df[["fire", "water", "air", "earth"]].sum(axis=1)
        features["numerology"] = df["numerology"].astype(float)
        features["frequency_mean"] = df["frequency"].astype(float)
        features["light_code"] = df["light_code"].astype(float)
        return features

    def fit(self, df: pd.DataFrame, target_col: str = "target_qlp") -> np.ndarray:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' missing")

        x = self.build_features(df).to_numpy(dtype=float)
        y = df[target_col].to_numpy(dtype=float)
        self.coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
        return self.coefficients

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("Model is not fitted yet")

        x = self.build_features(df).to_numpy(dtype=float)
        return x @ self.coefficients


def synthetic_dataset(size: int = 100, seed: int = 143) -> pd.DataFrame:
    """Generate deterministic synthetic data for local tests/demo runs."""
    rng = np.random.default_rng(seed)

    frame = pd.DataFrame(
        {
            "x": rng.normal(0, 1, size),
            "y": rng.normal(0, 1, size),
            "z": rng.normal(0, 1, size),
            "fire": rng.uniform(0, 1, size),
            "water": rng.uniform(0, 1, size),
            "air": rng.uniform(0, 1, size),
            "earth": rng.uniform(0, 1, size),
            "numerology": rng.integers(1, 40, size),
            "frequency": rng.uniform(100, 900, size),
            "light_code": rng.integers(100, 120, size),
        }
    )

    # Ground-truth relationship used to benchmark reproducibility.
    spatial = np.linalg.norm(frame[["x", "y", "z"]].to_numpy(), axis=1)
    element = frame[["fire", "water", "air", "earth"]].sum(axis=1)
    frame["target_qlp"] = (
        1.5
        + 0.8 * spatial
        + 0.4 * element
        + 0.1 * frame["numerology"]
        + 0.01 * frame["frequency"]
        + 0.05 * frame["light_code"]
        + rng.normal(0, 0.03, size)
    )
    return frame


def run_demo() -> None:
    data = synthetic_dataset()
    train = data.iloc[:80]
    test = data.iloc[80:]

    model = AlphaMindGammaModel()
    model.fit(train)
    predictions = model.predict(test)

    mae = np.mean(np.abs(predictions - test["target_qlp"].to_numpy()))
    print(f"Alpha Mind Gamma MAE: {mae:.4f}")


if __name__ == "__main__":
    run_demo()
