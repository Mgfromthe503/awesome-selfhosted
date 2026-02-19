import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_training_data import validate_jsonl_file


class ValidateTrainingDataTests(unittest.TestCase):
    def test_valid_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ok.jsonl"
            records = [
                {"prompt": "p1", "completion": "c1", "metadata": {"source": "x"}},
                {"prompt": "p2", "completion": "c2", "source": "synthetic"},
            ]
            p.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
            self.assertEqual(validate_jsonl_file(p), [])

    def test_invalid_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.jsonl"
            records = [
                {"prompt": "", "completion": "ok"},
                {"completion": "missing prompt"},
                {"prompt": "ok", "completion": "ok", "junk": True},
            ]
            p.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
            errors = validate_jsonl_file(p)
            self.assertGreaterEqual(len(errors), 3)


if __name__ == "__main__":
    unittest.main()
