"""Local access helpers for Codex to consume MM/Sherlock training data."""

from __future__ import annotations

import json
from pathlib import Path

from mm_language_framework import MMFramework, FuturisticQuantumKeyGenerator as MMFQKG
from sherlock_training_data import (
    EmojiParser,
    FuturisticQuantumKeyGenerator as SherlockFQKG,
    emoji_translator,
    get_polygon_staking_training_data,
    spiritual_meanings,
)


def build_training_snapshot() -> dict:
    mm = MMFramework(rng_seed=1337)
    parser = EmojiParser()
    return {
        "mm": {
            "hermetic_principles": mm.hermetic_principles,
            "emoji_translator": mm.emoji_translator,
            "elements": mm.elements,
        },
        "sherlock": {
            "emoji_parser_map": parser.emoji_map,
            "emoji_translator": emoji_translator,
            "spiritual_meanings": spiritual_meanings,
            "futuristic_qkg_methods": [m for m in dir(SherlockFQKG) if not m.startswith("_")],
            "polygon_staking_training_data": get_polygon_staking_training_data(),
        },
        "mm_access": {
            "futuristic_qkg_methods": [m for m in dir(MMFQKG) if not m.startswith("_")],
        },
    }


def write_snapshot(path: str = "training_data_snapshot.json") -> str:
    snapshot = build_training_snapshot()
    out = Path(path)
    out.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(out)


if __name__ == "__main__":
    target = write_snapshot()
    print(f"Wrote snapshot: {target}")
