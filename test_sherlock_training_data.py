import unittest

from sherlock_training_data import SherlockAI, Vector12D, build_sherlock_embedded_data


class TestSherlockTrainingData(unittest.TestCase):
    def test_vector12d_scalar_embed(self):
        v = Vector12D([0] * 12)
        v.embed_data(2)
        self.assertEqual(v.coords, [2.0] * 12)

    def test_preprocess_data_normalizes(self):
        s = SherlockAI()
        s.load_dataset("d", [[1, 10], [3, 30]])
        out = s.preprocess_data("d")
        self.assertEqual(out[0], [0.0, 0.0])
        self.assertEqual(out[1], [1.0, 1.0])

    def test_action_and_rollback(self):
        s = SherlockAI()
        start = list(s.state.coords)
        s.perform_action("test", data=1)
        self.assertNotEqual(s.state.coords, start)
        self.assertTrue(s.rollback())
        self.assertEqual(s.state.coords, start)

    def test_embedded_payload(self):
        payload = build_sherlock_embedded_data()
        self.assertIn("state", payload)
        self.assertIn("block_hashes", payload)
        self.assertEqual(len(payload["block_hashes"]), 3)


if __name__ == "__main__":
    unittest.main()
