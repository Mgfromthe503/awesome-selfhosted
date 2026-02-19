import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

from sherlock_multimodal_training import load_records, train_multimodal_model


def test_multimodal_training_loop_runs(tmp_path):
    data_file = tmp_path / "mini.jsonl"
    rows = [
        {"prompt": "find anomaly in sequence", "completion": "anomaly at index 2", "metadata": {"source": "mm_language"}},
        {"prompt": "translate emoji", "completion": "fire", "metadata": {"source": "emoji_translator"}},
        {"prompt": "predict target qlp", "completion": "target_qlp=3.14", "metadata": {"source": "alpha_mind_gamma_model"}},
        {"prompt": "explain symbol", "completion": "symbolic meaning", "metadata": {"source": "mm_emoji_knowledge_base"}},
    ]
    with data_file.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    recs = load_records(data_file)
    assert len(recs) == 4

    out_dir = tmp_path / "ckpt"
    result = train_multimodal_model(data_file, output_dir=out_dir, epochs=1, batch_size=2)
    assert result["num_records"] == 4
    assert pathlib.Path(result["checkpoint"]).exists()
