#!/usr/bin/env python3
"""Route text files into `projects/` category folders using keyword scoring.

This router supports dry-run previews, move/copy execution modes, and JSON
manifests that include source, destination, category, and score details.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path.cwd()
DEFAULT_DEST_DIRNAME = "projects"

CATEGORIES: dict[str, list[str]] = {
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

SUPPORTED_EXT = {".py", ".md", ".txt", ".json"}
UNCLASSIFIED = "mm_unclassified"
EXCLUDED_DIRS = {DEFAULT_DEST_DIRNAME, ".git", "__pycache__", ".venv", "venv"}


@dataclass
class RoutePlan:
    source: Path
    destination: Path
    assigned_project: str
    score: int


def read_text(path: Path) -> str:
    """Read file content in lowercase, returning an empty string on read error."""
    try:
        return path.read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return ""


def classify(content: str) -> tuple[str, int]:
    """Return the best category and score for content based on keyword matches."""
    scores = {category: 0 for category in CATEGORIES}

    for category, keywords in CATEGORIES.items():
        for keyword in keywords:
            if keyword in content:
                scores[category] += 1

    best = max(scores, key=scores.get)
    return (best, scores[best]) if scores[best] > 0 else (UNCLASSIFIED, 0)


def unique_target_path(target: Path) -> Path:
    """Return a non-colliding target path by appending _dupN when required."""
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


def iter_candidates(root: Path, script_name: str, dest_dirname: str) -> Iterable[Path]:
    """Yield candidate files under root that are eligible for routing."""
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXT:
            continue
        if path.name == script_name:
            continue
        if any(part in EXCLUDED_DIRS or part == dest_dirname for part in path.parts):
            continue
        yield path


def build_plan(root: Path, dest_root: Path, script_name: str) -> list[RoutePlan]:
    """Create routing plan for all eligible files."""
    plans: list[RoutePlan] = []
    for source in iter_candidates(root, script_name=script_name, dest_dirname=dest_root.name):
        content = read_text(source)
        category, score = classify(content)
        target = unique_target_path(dest_root / category / source.name)
        plans.append(RoutePlan(source=source, destination=target, assigned_project=category, score=score))
    return plans


def ensure_dest_dirs(dest_root: Path) -> None:
    """Create destination category directories."""
    dest_root.mkdir(exist_ok=True)
    for category in [*CATEGORIES.keys(), UNCLASSIFIED]:
        (dest_root / category).mkdir(exist_ok=True)


def apply_plan(plans: list[RoutePlan], mode: str) -> None:
    """Execute the routing plan in copy or move mode."""
    for plan in plans:
        plan.destination.parent.mkdir(parents=True, exist_ok=True)
        if mode == "copy":
            shutil.copy2(plan.source, plan.destination)
        else:
            shutil.move(str(plan.source), plan.destination)


def write_manifest(dest_root: Path, plans: list[RoutePlan], mode: str, dry_run: bool) -> Path:
    """Write a JSON manifest for all routed files."""
    manifest = [
        {
            "file": str(plan.source),
            "destination": str(plan.destination),
            "assigned_project": plan.assigned_project,
            "score": plan.score,
            "mode": mode,
            "dry_run": dry_run,
        }
        for plan in plans
    ]
    manifest_path = dest_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Root directory to scan.")
    parser.add_argument(
        "--dest",
        type=Path,
        default=ROOT / DEFAULT_DEST_DIRNAME,
        help="Destination root for categorized files.",
    )
    parser.add_argument(
        "--mode",
        choices=["move", "copy"],
        default="move",
        help="How to route files when --execute is set.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Apply changes. Without this flag, the script performs a dry run only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    dest_root = args.dest.resolve()

    ensure_dest_dirs(dest_root)
    plans = build_plan(root=root, dest_root=dest_root, script_name=Path(__file__).name)

    if args.execute:
        apply_plan(plans, mode=args.mode)

    manifest_path = write_manifest(dest_root, plans, mode=args.mode, dry_run=not args.execute)

    print(f"✔ Planned routes: {len(plans)}")
    print(f"✔ Mode: {'DRY RUN' if not args.execute else args.mode.upper()}")
    print(f"✔ manifest.json created at: {manifest_path}")


if __name__ == "__main__":
    main()
