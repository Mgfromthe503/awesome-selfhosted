import unittest

from quantum_lucidity import machine_learning_based_qlp, quantum_lucidity


class QuantumLucidityTests(unittest.TestCase):
    def test_prediction_returns_float(self):
        result = quantum_lucidity(0.3, 0.2, 0.4, 0.25, 14, 0.5, 1.5)
        self.assertIsInstance(result, float)

    def test_prediction_is_reproducible(self):
        features = [0.5, 0.35, 0.45, 0.4, 15, 0.65, 1.8]
        first = machine_learning_based_qlp(features)
        second = machine_learning_based_qlp(features)
        self.assertAlmostEqual(first, second, places=10)

    def test_invalid_feature_length_raises(self):
        with self.assertRaises(ValueError):
            machine_learning_based_qlp([1, 2, 3])


if __name__ == "__main__":
    unittest.main()
