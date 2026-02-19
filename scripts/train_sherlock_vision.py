#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sherlock_vision_training import train_sherlock_vision


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Sherlock vision model from annotations")
    p.add_argument("--annotations", default="data/images/annotations.jsonl")
    p.add_argument("--output-dir", default="data/processed/vision_checkpoints")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    result = train_sherlock_vision(
        annotations_path=a.annotations,
        output_dir=a.output_dir,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        device=a.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
