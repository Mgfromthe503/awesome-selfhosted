"""Training-data quality checks for Sherlock datasets.

This script validates JSONL files used for training and catches schema drift early.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

REQUIRED_KEYS = {"prompt", "completion"}
OPTIONAL_KEYS = {"metadata", "source"}


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc})") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no}: record must be a JSON object")
            yield line_no, payload


def validate_jsonl_file(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"missing file: {path}"]

    for line_no, rec in _iter_jsonl(path):
        missing = REQUIRED_KEYS - set(rec.keys())
        if missing:
            errors.append(f"{path}:{line_no}: missing keys {sorted(missing)}")
            continue

        prompt = rec.get("prompt")
        completion = rec.get("completion")
        if not isinstance(prompt, str) or not prompt.strip():
            errors.append(f"{path}:{line_no}: prompt must be non-empty string")
        if not isinstance(completion, str) or not completion.strip():
            errors.append(f"{path}:{line_no}: completion must be non-empty string")

        unknown = set(rec.keys()) - REQUIRED_KEYS - OPTIONAL_KEYS
        if unknown:
            errors.append(f"{path}:{line_no}: unknown keys {sorted(unknown)}")
    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "data" / "processed" / "sherlock_training_combined.jsonl",
        root / "data" / "processed" / "sherlock_snippet_training.jsonl",
        root / "data" / "processed" / "sherlock_vision_training.jsonl",
        root / "data" / "test_alpha_training.jsonl",
    ]

    all_errors: list[str] = []
    for dataset in candidates:
        all_errors.extend(validate_jsonl_file(dataset))

    if all_errors:
        print("Training data validation failed:")
        for err in all_errors:
            print(f"- {err}")
        return 1

    print("Training data validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
