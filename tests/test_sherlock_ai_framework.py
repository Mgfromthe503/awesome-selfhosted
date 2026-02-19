import unittest

from sherlock_ai_framework import Book, Principle, Sherlock


class SherlockFrameworkTests(unittest.TestCase):
    def setUp(self):
        self.sherlock = Sherlock()
        self.sherlock.DNA.integratePrinciple(
            Principle(Code=1, Name="Mentalism", Description="The universe is mental")
        )

    def test_investigate_runs_end_to_end(self):
        findings = self.sherlock.investigate({"signal": "ATCG"}, principleCode=1)
        self.assertEqual(findings["principle"], "Mentalism")
        self.assertTrue(findings["quantum"]["processed"])
        self.assertEqual(findings["bioinformatics"]["dimensions"], 12)

    def test_seek_truth_matches_analyze(self):
        truth = self.sherlock.seekTruth("sample", 1)
        self.assertEqual(truth["principle"], "Mentalism")

    def test_mental_health_support_returns_resources(self):
        self.sherlock.User = {"mood_score": -0.8}
        resources = self.sherlock.mentalHealthSupport()
        self.assertGreaterEqual(len(resources), 1)

    def test_train_on_book_updates_knowledge(self):
        update_result = self.sherlock.trainOnBook(Book(title="A", content="B"))
        self.assertTrue(update_result["updated"])
        self.assertEqual(update_result["knowledge_size"], 1)


if __name__ == "__main__":
    unittest.main()
