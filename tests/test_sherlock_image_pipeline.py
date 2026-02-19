import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sherlock_image_pipeline import build_vision_training_jsonl, create_annotation_template, ingest_images, validate_annotations


def test_image_ingest_and_annotation_pipeline(tmp_path):
    src = tmp_path / "src"
    src.mkdir(parents=True)

    # Create minimal png-like file for pipeline file handling.
    img = src / "sample.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"demo")

    out_images = tmp_path / "raw"
    ingest = ingest_images([src], out_dir=out_images)
    assert ingest["ingested_count"] == 1

    ann = create_annotation_template(images_dir=out_images, out_path=tmp_path / "annotations.jsonl")
    check = validate_annotations(ann)
    assert check["records"] == 1
    assert check["valid"] == 1

    out_jsonl = build_vision_training_jsonl(ann, out_path=tmp_path / "vision.jsonl")
    lines = out_jsonl.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["metadata"]["source"] == "vision"
