#!/usr/bin/env python3
"""Build unified Sherlock training dataset JSONL."""

from __future__ import annotations

import argparse

from sherlock_dataset_builder import build_unified_training_dataset
from sherlock_evaluation import dataset_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/processed/sherlock_unified_training.jsonl")
    parser.add_argument("--include-vision", action="store_true")
    parser.add_argument("--images", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = build_unified_training_dataset(
        out_path=args.out,
        include_alpha=True,
        include_vision=args.include_vision,
        image_paths=args.images,
    )
    snap = dataset_snapshot(out)
    print(f"Wrote dataset: {out}")
    print(f"Records: {snap['records']}")
    print(f"Sources: {snap['sources']}")


if __name__ == "__main__":
    main()
