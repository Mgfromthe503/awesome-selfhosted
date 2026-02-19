import unittest

from sherlock_ai_framework import (
    Book,
    Collaboration,
    ContinuousLearning,
    DNA_System,
    Ethics,
    Principle,
    Sherlock,
)


class SherlockFrameworkTests(unittest.TestCase):
    def test_integrate_principle_and_analyze_numeric_data(self):
        dna = DNA_System()
        dna.integrate_principle(Principle(code=1, name="Mentalism", description="Mind is foundational"))
        results = dna.analyze_data([1, 2, 3, 4])
        self.assertEqual(results["data_type"], "numeric_series")
        self.assertEqual(results["count"], 4)
        self.assertEqual(results["mean"], 2.5)
        self.assertEqual(results["principle_count"], 1)

    def test_duplicate_principle_code_raises(self):
        dna = DNA_System()
        dna.integrate_principle(Principle(code=1, name="Mentalism", description="d"))
        with self.assertRaises(ValueError):
            dna.integrate_principle(Principle(code=1, name="Correspondence", description="d"))

    def test_sherlock_training_and_truth(self):
        detective = Sherlock()
        detective.train_on_book(Book(title="Signals", author="A. Author", content="pattern analysis"))
        truth = detective.seek_truth([10, 20])
        self.assertIn("confidence", truth)
        self.assertGreaterEqual(truth["confidence"], 0.5)

    def test_ethics_checks(self):
        ethics = Ethics()
        bias = ethics.check_for_bias("Model used age and gender variables")
        self.assertEqual(bias["bias_risk"], "elevated")

        privacy = ethics.ensure_privacy({"name": "Alex", "email": "x@example.com"})
        self.assertEqual(privacy["privacy_status"], "needs_redaction")

        transparency = ethics.provide_transparency()
        self.assertEqual(transparency["reports_generated"], 1)

    def test_learning_and_collaboration(self):
        detective = Sherlock()
        learning = ContinuousLearning(sherlock=detective)
        collaboration = Collaboration()

        update_result = learning.apply_update({"version": "1.0.1"})
        feedback_result = learning.gather_feedback({"quality": "good"})
        partner_result = collaboration.collaborate_with_partner("OpenAI Lab")
        contribution_result = collaboration.integrate_open_source_contribution({"pr": 42})

        self.assertEqual(update_result["status"], "applied")
        self.assertEqual(feedback_result["status"], "captured")
        self.assertEqual(partner_result["status"], "active")
        self.assertEqual(contribution_result["total_contributions"], 1)


if __name__ == "__main__":
    unittest.main()
