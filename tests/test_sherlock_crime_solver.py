import csv
import tempfile
import unittest
from pathlib import Path

from sherlock_crime_solver import (
    Sherlock,
    format_prediction_report,
    load_and_preprocess_crime_data,
    load_unsolved_data,
    predict_unsolved_cases,
)


class DummyModel:
    def predict(self, matrix):
        return [1 if row[0] > 2 else 0 for row in matrix]


class SherlockCrimeSolverTests(unittest.TestCase):
    def test_sherlock_auth_users(self):
        s = Sherlock("root", "x", ["alice", "bob"])
        self.assertEqual(s.authenticated_users, ["root", "alice", "bob"])

    def test_load_and_preprocess_drops_nulls(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "crime.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["feature1", "feature2", "feature3", "solved"])
                writer.writeheader()
                writer.writerow({"feature1": 1, "feature2": 2, "feature3": 3, "solved": 0})
                writer.writerow({"feature1": "", "feature2": 1, "feature3": 2, "solved": 1})

            cleaned = load_and_preprocess_crime_data(
                str(csv_path), ["feature1", "feature2", "feature3"], "solved"
            )
            self.assertEqual(len(cleaned), 1)

    def test_predict_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "unsolved.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["feature1", "feature2", "feature3"])
                writer.writeheader()
                writer.writerow({"feature1": 1, "feature2": 0, "feature3": 0})
                writer.writerow({"feature1": 3, "feature2": 0, "feature3": 0})

            unsolved_rows = load_unsolved_data(str(csv_path), ["feature1", "feature2", "feature3"])
            predictions = predict_unsolved_cases(DummyModel(), unsolved_rows, ["feature1", "feature2", "feature3"])
            self.assertEqual(predictions, [0, 1])
            self.assertEqual(
                format_prediction_report(predictions),
                ["Case 0 is unlikely to be solved.", "Case 1 is likely to be solved."],
            )


if __name__ == "__main__":
    unittest.main()
