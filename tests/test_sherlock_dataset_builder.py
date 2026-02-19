import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sherlock_dataset_builder import build_unified_training_dataset
from sherlock_evaluation import dataset_snapshot


def test_build_unified_dataset_and_snapshot(tmp_path):
    out = tmp_path / "unified.jsonl"
    path = build_unified_training_dataset(out_path=out, include_alpha=True, include_vision=False)
    assert path.exists()

    snap = dataset_snapshot(path)
    assert snap["records"] > 0
    assert "emoji_translator" in snap["sources"] or "mm_emoji_knowledge_base" in snap["sources"]
