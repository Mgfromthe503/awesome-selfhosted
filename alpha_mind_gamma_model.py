"""Alpha Mind Gamma model using open-source numerical libraries.

This module provides a practical replacement for speculative placeholders by using
NumPy/Pandas data processing and a small, explainable linear model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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


def _record_from_row(row: pd.Series, predicted: float) -> dict:
    prompt = (
        "Predict target_qlp from features: "
        f"x={row['x']:.6f}, y={row['y']:.6f}, z={row['z']:.6f}, "
        f"fire={row['fire']:.6f}, water={row['water']:.6f}, air={row['air']:.6f}, "
        f"earth={row['earth']:.6f}, numerology={float(row['numerology']):.0f}, "
        f"frequency={row['frequency']:.6f}, light_code={float(row['light_code']):.0f}."
    )
    completion = f"target_qlp={predicted:.6f}"
    return {
        "prompt": prompt,
        "completion": completion,
        "metadata": {
            "source": "alpha_mind_gamma_model",
            "task": "regression_estimation",
        },
    }


def export_training_jsonl(
    out_path: str | Path,
    *,
    size: int = 300,
    seed: int = 143,
    train_ratio: float = 0.8,
) -> Path:
    """Export prompt/completion JSONL examples for Sherlock fine-tuning."""
    if size < 10:
        raise ValueError("size must be >= 10")
    if not (0.1 <= train_ratio <= 0.95):
        raise ValueError("train_ratio must be between 0.1 and 0.95")

    data = synthetic_dataset(size=size, seed=seed)
    cutoff = int(size * train_ratio)
    train = data.iloc[:cutoff]
    infer = data.iloc[cutoff:]

    model = AlphaMindGammaModel()
    model.fit(train)
    preds = model.predict(infer)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as fh:
        for (_, row), pred in zip(infer.iterrows(), preds):
            fh.write(json.dumps(_record_from_row(row, float(pred)), ensure_ascii=False) + "\n")

    return out


def run_demo() -> None:
    data = synthetic_dataset()
    train = data.iloc[:80]
    test = data.iloc[80:]

    model = AlphaMindGammaModel()
    model.fit(train)
    predictions = model.predict(test)

    mae = np.mean(np.abs(predictions - test["target_qlp"].to_numpy()))
    print(f"Alpha Mind Gamma MAE: {mae:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-jsonl", type=Path, default=None, help="Write prompt/completion JSONL to this path.")
    parser.add_argument("--size", type=int, default=300, help="Synthetic dataset size used for export.")
    parser.add_argument("--seed", type=int, default=143, help="RNG seed for reproducible export.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio before generating predictions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.export_jsonl:
        out = export_training_jsonl(args.export_jsonl, size=args.size, seed=args.seed, train_ratio=args.train_ratio)
        print(f"Wrote training JSONL: {out}")
    else:
        run_demo()


if __name__ == "__main__":
    main()
