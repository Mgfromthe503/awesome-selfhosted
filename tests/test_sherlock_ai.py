import unittest

from scripts.sherlock_ai import SherlockAI, Vector12D


class TestSherlockAI(unittest.TestCase):
    def test_vector_embed_length_validation(self):
        vector = Vector12D([0.0] * 12)
        with self.assertRaises(ValueError):
            vector.embed_data([1.0, 2.0])

    def test_vector_embed_updates_coords(self):
        vector = Vector12D([1.0] * 12)
        vector.embed_data([2.0] * 12)
        self.assertEqual(vector.coords, [3.0] * 12)

    def test_train_pipeline(self):
        ai = SherlockAI()
        ai.preprocess_data("iris")
        report = ai.train_model("iris")

        self.assertIn("accuracy", report)
        self.assertIn("confusion_matrix", report)
        self.assertGreaterEqual(report["accuracy"], 0.5)

    def test_sound_ping_echo(self):
        ai = SherlockAI()
        ai.add_sound("ping", [1.0, 0.25])
        self.assertEqual(ai.ping_sound("ping"), [1.0, 0.25, 1.0, 0.25])


if __name__ == "__main__":
    unittest.main()
