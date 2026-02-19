import unittest

from sherlock_ai import Book, build_default_sherlock


class TestSherlockAI(unittest.TestCase):
    def setUp(self):
        self.sherlock = build_default_sherlock()

    def test_investigate_vibration(self):
        findings = self.sherlock.investigate("AABB", principle_code=3)
        self.assertEqual(findings["principle"], "Vibration")
        self.assertIn("frequency", findings["analysis"])

    def test_report(self):
        findings = self.sherlock.investigate("pattern mapping", principle_code=2)
        report = self.sherlock.report(findings)
        self.assertIn("Sherlock Report", report)
        self.assertIn("Correspondence", report)

    def test_train_on_book(self):
        extracted = self.sherlock.train_on_book(Book(title="Test", content="Alpha beta beta gamma"))
        self.assertEqual(extracted["token_count"], 4)
        self.assertEqual(extracted["unique_terms"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
