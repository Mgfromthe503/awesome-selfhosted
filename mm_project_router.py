#!/usr/bin/env python3
"""Classify repository text files into project buckets.

The script scans the current working directory for supported text files,
classifies each file based on keyword matches, and moves files into
`projects/<category>/`. A `projects/manifest.json` file is generated with the
original source path and assigned category.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path.cwd()
DEST = ROOT / "projects"

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
    "mm_utilities": [
        "utils",
        "helper",
        "config",
        "parser",
        "io",
    ],
}

SUPPORTED_EXT = {".py", ".md", ".txt", ".json"}
UNCLASSIFIED = "mm_unclassified"
EXCLUDED_DIRS = {"projects", ".git", "__pycache__"}


def read_text(path: Path) -> str:
    """Read file content in lowercase, returning an empty string on failure."""
    try:
        return path.read_text(errors="ignore").lower()
    except OSError:
        return ""


def classify(content: str) -> str:
    """Return the best category for content based on keyword frequency."""
    scores = {category: 0 for category in CATEGORIES}

    for category, keywords in CATEGORIES.items():
        for keyword in keywords:
            if keyword in content:
                scores[category] += 1

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else UNCLASSIFIED


def unique_target_path(target: Path) -> Path:
    """Return a non-colliding target path by appending _dupN when needed."""
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


def main() -> None:
    DEST.mkdir(exist_ok=True)
    manifest: list[dict[str, str]] = []

    categories = list(CATEGORIES) + [UNCLASSIFIED]
    for category in categories:
        (DEST / category).mkdir(exist_ok=True)

    for path in ROOT.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_EXT
            and path.name != Path(__file__).name
            and not any(part in EXCLUDED_DIRS for part in path.parts)
        ):
            content = read_text(path)
            category = classify(content)

            target = unique_target_path(DEST / category / path.name)
            source_str = str(path)
            shutil.move(str(path), target)

            manifest.append({"file": source_str, "assigned_project": category})

    with (DEST / "manifest.json").open("w", encoding="utf-8") as file_obj:
        json.dump(manifest, file_obj, indent=2)

    print("✔ Code scanned, classified, and routed into projects/")
    print("✔ manifest.json created")


if __name__ == "__main__":
    main()
