import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sherlock_vision_training import train_sherlock_vision


def _write_test_image(path: pathlib.Path) -> None:
    try:
        from PIL import Image

        img = Image.new("RGB", (16, 16), color=(128, 64, 32))
        img.save(path)
    except Exception:
        # Fallback minimal bytes for environments without pillow decode paths.
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"demo")


def test_vision_training_loop_runs(tmp_path):
    ann = tmp_path / "ann.jsonl"
    img = tmp_path / "img.png"
    _write_test_image(img)

    rows = [
        {"image_path": str(img), "label": "target", "bbox": [], "split": "train", "notes": ""},
        {"image_path": str(img), "label": "target", "bbox": [], "split": "val", "notes": ""},
    ]
    with ann.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    result = train_sherlock_vision(ann, output_dir=tmp_path / "ckpt", epochs=1, batch_size=1)
    assert result["num_rows"] == 2
    assert pathlib.Path(result["checkpoint"]).exists()
