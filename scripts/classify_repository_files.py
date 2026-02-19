#!/usr/bin/env python3
"""Classify snippet files into project buckets.

Safe defaults:
- scans only `data/snippet_inbox`
- writes to `data/snippet_projects`
- dry-run by default (writes manifest only)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INBOX = REPO_ROOT / "data" / "snippet_inbox"
DEFAULT_DEST = REPO_ROOT / "data" / "snippet_projects"

CATEGORIES = {
    "mm_physics_models": [
        "schrodinger",
        "hamiltonian",
        "lagrangian",
        "entropy",
        "field",
        "tensor",
        "gravity",
        "cosmology",
        "relativity",
    ],
    "mm_quantum_models": [
        "qiskit",
        "quantum",
        "qubit",
        "entanglement",
        "wavefunction",
        "hilbert",
        "operator",
    ],
    "mm_ai_models": [
        "neural",
        "training",
        "model",
        "prediction",
        "machine learning",
        "deep learning",
        "transformer",
    ],
    "mm_bioinformatics": [
        "dna",
        "rna",
        "genome",
        "bioinformatics",
        "protein",
        "sequence",
        "biological",
    ],
    "mm_utilities": ["utils", "helper", "config", "parser", "io"],
}

SUPPORTED_EXT = {".py", ".md", ".txt", ".json", ".yaml", ".yml"}
UNCLASSIFIED = "mm_unclassified"


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return ""


def classify(content: str) -> tuple[str, int]:
    scores = {category: 0 for category in CATEGORIES}
    for category, keywords in CATEGORIES.items():
        for keyword in keywords:
            if keyword in content:
                scores[category] += 1

    best = max(scores, key=scores.get)
    return (best, scores[best]) if scores[best] > 0 else (UNCLASSIFIED, 0)


def unique_target_path(target: Path) -> Path:
    if not target.exists():
        return target

    stem = target.stem
    suffix = target.suffix
    counter = 1
    while True:
        candidate = target.with_name(f"{stem}_dup{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inbox", type=Path, default=DEFAULT_INBOX)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--mode", choices=["copy", "move"], default="copy")
    parser.add_argument("--execute", action="store_true", help="Apply file operations. Without this flag, dry-run only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inbox = args.inbox.resolve()
    dest = args.dest.resolve()

    inbox.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)

    categories = list(CATEGORIES) + [UNCLASSIFIED]
    for category in categories:
        (dest / category).mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    for path in inbox.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXT:
            continue

        content = read_text(path)
        category, score = classify(content)
        target = unique_target_path(dest / category / path.name)

        if args.execute:
            if args.mode == "copy":
                shutil.copy2(path, target)
            else:
                shutil.move(str(path), target)

        manifest.append(
            {
                "file": str(path),
                "destination": str(target),
                "assigned_project": category,
                "score": score,
                "mode": args.mode,
                "dry_run": not args.execute,
            }
        )

    manifest_path = dest / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Scanned snippets: {len(manifest)}")
    print(f"Mode: {'DRY RUN' if not args.execute else args.mode.upper()}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
