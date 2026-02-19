#!/usr/bin/env python3
"""Generate benchmark snapshot for Sherlock V1."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sherlock_evaluation import dataset_snapshot


DATASETS = [
    "data/processed/alpha_training.jsonl",
    "data/processed/sherlock_snippet_training.jsonl",
    "data/processed/sherlock_training_combined.jsonl",
]


def _safe_snapshot(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "records": 0, "sources": {}, "exists": False}
    out = dataset_snapshot(p)
    out["exists"] = True
    return out


def build_benchmark_snapshot() -> dict:
    snaps = [_safe_snapshot(p) for p in DATASETS]
    total_records = sum(s["records"] for s in snaps)

    aggregate_sources: dict[str, int] = {}
    for s in snaps:
        for k, v in s.get("sources", {}).items():
            aggregate_sources[k] = aggregate_sources.get(k, 0) + int(v)

    return {
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "datasets": snaps,
        "total_records": total_records,
        "aggregate_sources": aggregate_sources,
    }


def main() -> None:
    out = Path("data/processed/benchmark_v1_snapshot.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    snap = build_benchmark_snapshot()
    out.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()

