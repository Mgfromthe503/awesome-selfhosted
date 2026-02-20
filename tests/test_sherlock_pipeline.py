import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    pd = None

from sherlock_pipeline import (
    FEATURES,
    load_model,
    predict_unsolved,
    save_model,
    save_prediction_report,
    train_model,
)


@unittest.skipIf(pd is None, "pandas is not installed in this environment")
class SherlockPipelineTests(unittest.TestCase):
    def _training_df(self):
        rows = []
        for i in range(30):
            rows.append(
                {
                    "age": 20 + i % 25,
                    "gender": "M" if i % 2 else "F",
                    "race": "A" if i % 3 else "B",
                    "location": "north" if i % 2 else "south",
                    "time_of_day": "night" if i % 2 else "day",
                    "weapon": "knife" if i % 3 else "none",
                    "motive": "robbery" if i % 2 else "dispute",
                    "victim_age": 18 + i % 20,
                    "victim_gender": "M" if i % 2 else "F",
                    "crime_type": "theft" if i % 2 else "assault",
                    "solved": 1 if i % 4 else 0,
                }
            )
        return pd.DataFrame(rows)

    def test_train_save_load_predict_and_report(self):
        train_df = self._training_df()
        model, metrics = train_model(train_df)
        self.assertGreaterEqual(metrics.accuracy, 0.0)
        self.assertLessEqual(metrics.accuracy, 1.0)

        with TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "model.pkl"
            report_path = Path(tmp_dir) / "report.csv"

            save_model(model, model_path)
            loaded = load_model(model_path)

            unsolved = train_df[FEATURES].head(5).copy()
            preds = predict_unsolved(loaded, unsolved)
            self.assertEqual(len(preds), 5)

            save_prediction_report(preds, report_path)
            self.assertTrue(report_path.exists())


if __name__ == "__main__":
    unittest.main()
