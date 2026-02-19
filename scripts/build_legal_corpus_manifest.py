"""Build a normalized legal-corpus manifest for Sherlock training.

Reads ``configs/legal_corpora_sources.yaml`` and emits JSON manifest records.
"""

from __future__ import annotations

import json
from pathlib import Path


def _parse_simple_yaml(path: Path) -> dict:
    """Minimal parser for this repo's simple YAML structure.

    We avoid external dependencies to keep setup lightweight.
    """
    text = path.read_text(encoding="utf-8")
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]

    sources: list[dict[str, str]] = []
    compliance: dict[str, bool] = {}
    current: dict[str, str] | None = None
    in_compliance = False

    for ln in lines:
        s = ln.strip()
        if s == "sources:":
            in_compliance = False
            continue
        if s == "compliance:":
            in_compliance = True
            if current is not None:
                sources.append(current)
                current = None
            continue

        if in_compliance:
            if ":" in s:
                k, v = [p.strip() for p in s.split(":", 1)]
                compliance[k] = v.lower() == "true"
            continue

        if s.startswith("- "):
            if current is not None:
                sources.append(current)
            current = {}
            s = s[2:]
            if s and ":" in s:
                k, v = [p.strip() for p in s.split(":", 1)]
                current[k] = v.strip('"')
            continue

        if ":" in s and current is not None:
            k, v = [p.strip() for p in s.split(":", 1)]
            current[k] = v.strip('"')

    if current is not None:
        sources.append(current)

    return {"sources": sources, "compliance": compliance}


def build_manifest(config_path: Path, output_path: Path) -> dict:
    payload = _parse_simple_yaml(config_path)
    manifest = {
        "version": 1,
        "source_count": len(payload["sources"]),
        "sources": payload["sources"],
        "compliance": payload["compliance"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    config = root / "configs" / "legal_corpora_sources.yaml"
    out = root / "data" / "processed" / "legal_corpus_manifest.json"
    manifest = build_manifest(config, out)
    print(f"Wrote legal corpus manifest with {manifest['source_count']} sources to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
