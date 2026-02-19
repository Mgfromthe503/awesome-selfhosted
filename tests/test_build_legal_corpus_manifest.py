from pathlib import Path

from scripts.build_legal_corpus_manifest import build_manifest


def test_build_manifest(tmp_path: Path):
    config = Path(__file__).resolve().parents[1] / "configs" / "legal_corpora_sources.yaml"
    output = tmp_path / "manifest.json"
    manifest = build_manifest(config, output)

    assert output.exists()
    assert manifest["source_count"] >= 5
    assert manifest["compliance"]["block_unknown_license"] is True
