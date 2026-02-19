import json
import unittest
from pathlib import Path

try:
    from alpha_mind_gamma_model import AlphaMindGammaModel, export_training_jsonl, synthetic_dataset

    DEPENDENCIES_AVAILABLE = True
except ModuleNotFoundError:
    DEPENDENCIES_AVAILABLE = False


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "numpy/pandas are not available in this environment")
class AlphaMindGammaModelTests(unittest.TestCase):
    def test_fit_and_predict(self):
        data = synthetic_dataset(size=120, seed=42)
        train = data.iloc[:100]
        test = data.iloc[100:]

        model = AlphaMindGammaModel()
        coeffs = model.fit(train)
        preds = model.predict(test)

        self.assertEqual(coeffs.ndim, 1)
        self.assertEqual(len(preds), len(test))

        mae = (abs(preds - test["target_qlp"])).mean()
        self.assertLess(mae, 0.15)

    def test_missing_columns_error(self):
        data = synthetic_dataset(size=10)[["x", "y", "z"]].copy()
        model = AlphaMindGammaModel()
        with self.assertRaises(ValueError):
            model.build_features(data)

    def test_export_training_jsonl(self):
        out = Path("data/test_alpha_training.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            out.unlink()

        path = export_training_jsonl(out, size=50, seed=7, train_ratio=0.8)
        self.assertTrue(path.exists())

        lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 10)

        first = json.loads(lines[0])
        self.assertIn("prompt", first)
        self.assertIn("completion", first)
        self.assertIn("metadata", first)


if __name__ == "__main__":
    unittest.main()
