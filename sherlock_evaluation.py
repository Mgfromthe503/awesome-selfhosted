"""Evaluation helpers for Sherlock training datasets and predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


def evaluate_binary_predictions(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    if not y_true:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def dataset_snapshot(jsonl_path: str | Path) -> dict:
    path = Path(jsonl_path)
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    sources: dict[str, int] = {}
    for ln in lines:
        row = json.loads(ln)
        src = str(row.get("metadata", {}).get("source", "unknown"))
        sources[src] = sources.get(src, 0) + 1

    return {
        "path": str(path),
        "records": len(lines),
        "sources": sources,
    }
