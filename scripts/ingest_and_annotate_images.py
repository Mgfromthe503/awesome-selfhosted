#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sherlock_image_pipeline import ingest_images, create_annotation_template, validate_annotations, build_vision_training_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest images and generate annotation/template artifacts")
    p.add_argument("--sources", nargs="+", required=True)
    p.add_argument("--out-images", default="data/images/raw")
    p.add_argument("--annotations", default="data/images/annotations.jsonl")
    p.add_argument("--out-jsonl", default="data/processed/sherlock_vision_training.jsonl")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    ingest = ingest_images(a.sources, out_dir=a.out_images)
    ann = create_annotation_template(images_dir=a.out_images, out_path=a.annotations)
    check = validate_annotations(ann)
    jsonl = build_vision_training_jsonl(ann, out_path=a.out_jsonl)
    print(json.dumps({"ingest": ingest, "annotations": str(ann), "validation": check, "vision_jsonl": str(jsonl)}, indent=2))


if __name__ == "__main__":
    main()
