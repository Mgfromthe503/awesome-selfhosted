import unittest

from scripts.sherlock_ai import SherlockAI, Vector12D


class TestSherlockAI(unittest.TestCase):
    def test_vector_embed_scalar(self):
        vector = Vector12D([0.0] * 12)
        vector.embed_data(2)
        self.assertEqual(vector.coords, [2.0] * 12)

    def test_vector_embed_list_length_validation(self):
        vector = Vector12D([0.0] * 12)
        with self.assertRaises(ValueError):
            vector.embed_data([1.0, 2.0])

    def test_preprocess_and_evaluate(self):
        ai = SherlockAI()
        processed = ai.preprocess_data("iris")
        self.assertEqual(len(processed["data"]), len(processed["target"]))

        report = ai.evaluate("iris_preprocessed")
        self.assertGreaterEqual(report["accuracy"], 0.9)
        self.assertEqual(len(report["confusion_matrix"]), 3)

    def test_sound_ping(self):
        ai = SherlockAI()
        ai.add_sound("click", [1.0, 0.5, 0.25])
        echoed = ai.ping_sound("click")
        self.assertEqual(echoed, [0.6, 0.3, 0.15])


if __name__ == "__main__":
    unittest.main()
