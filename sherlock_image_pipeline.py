"""Sherlock image ingestion and annotation pipeline."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ingest_images(
    source_dirs: list[str | Path],
    *,
    out_dir: str | Path = "data/images/raw",
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ingested = []
    skipped = []
    for src in source_dirs:
        s = Path(src)
        if not s.exists():
            skipped.append({"source": str(s), "reason": "missing"})
            continue

        for p in s.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXT:
                continue
            digest = _sha256(p)[:16]
            dest_name = f"{digest}_{p.name}"
            dest = out / dest_name
            shutil.copy2(p, dest)
            ingested.append({"source": str(p), "dest": str(dest), "sha16": digest, "bytes": dest.stat().st_size})

    manifest = out / "ingest_manifest.json"
    manifest.write_text(json.dumps({"ingested": ingested, "skipped": skipped}, indent=2), encoding="utf-8")

    return {"out_dir": str(out), "manifest": str(manifest), "ingested_count": len(ingested), "skipped_count": len(skipped)}


def create_annotation_template(
    *,
    images_dir: str | Path = "data/images/raw",
    out_path: str | Path = "data/images/annotations.jsonl",
) -> Path:
    images = Path(images_dir)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in images.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXT:
            rows.append(
                {
                    "image_path": str(p),
                    "label": "unknown",
                    "bbox": [],
                    "split": "train",
                    "notes": "",
                }
            )

    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    return out


def validate_annotations(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]

    errors: list[str] = []
    valid = 0
    for idx, ln in enumerate(lines, start=1):
        try:
            row = json.loads(ln)
        except Exception:
            errors.append(f"line {idx}: invalid json")
            continue

        if "image_path" not in row:
            errors.append(f"line {idx}: missing image_path")
            continue
        if "label" not in row:
            errors.append(f"line {idx}: missing label")
            continue
        if not Path(str(row["image_path"])).exists():
            errors.append(f"line {idx}: image not found -> {row['image_path']}")
            continue

        bbox = row.get("bbox", [])
        if bbox and (not isinstance(bbox, list) or len(bbox) != 4):
            errors.append(f"line {idx}: bbox must be [] or [x,y,w,h]")
            continue

        valid += 1

    return {"path": str(p), "records": len(lines), "valid": valid, "errors": errors}


def build_vision_training_jsonl(
    annotations_path: str | Path = "data/images/annotations.jsonl",
    out_path: str | Path = "data/processed/sherlock_vision_training.jsonl",
) -> Path:
    ann = Path(annotations_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as fh:
        for ln in ann.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            row = json.loads(ln)
            rec = {
                "prompt": f"Analyze image and classify object/person. image_path={row.get('image_path')}",
                "completion": f"label={row.get('label','unknown')}",
                "metadata": {
                    "source": "vision",
                    "image_path": row.get("image_path"),
                    "bbox": row.get("bbox", []),
                    "split": row.get("split", "train"),
                    "notes": row.get("notes", ""),
                },
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return out
