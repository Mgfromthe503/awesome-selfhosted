import math
import unittest

from quantum_lucidity import (
    elemental_factors,
    feedback_adjustment,
    numerology_angel_factors,
    pi_approximation,
    quantum_lucidity,
    unsolved_equations_solver,
)


class QuantumLucidityTests(unittest.TestCase):
    def test_elemental_factors_defaults_missing_values(self):
        self.assertEqual(elemental_factors({"Earth": 1.0}), (1.0, 0.0, 0.0, 0.0))

    def test_numerology_angel_factors_normalized(self):
        nb, as_factor = numerology_angel_factors(112288, {"SomeMetric": 42})
        self.assertGreaterEqual(nb, 0.0)
        self.assertLessEqual(nb, 1.0)
        self.assertGreaterEqual(as_factor, 0.0)
        self.assertLessEqual(as_factor, 1.0)

    def test_quantum_lucidity_in_unit_interval(self):
        result = quantum_lucidity(0.4, 0.5, 0.6, 0.7, 0.22, 0.8, 0.5)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_feedback_adjustment_applies_delta(self):
        adjusted = feedback_adjustment(1.0, 0.5, [0.1, 0.2])
        self.assertEqual(adjusted, [0.6, 0.7])

    def test_unsolved_equations_solver(self):
        root = unsolved_equations_solver(lambda x: x**2 - 2, 1.0)
        self.assertTrue(math.isclose(root, math.sqrt(2), rel_tol=1e-8))

    def test_pi_approximation(self):
        estimate = pi_approximation(10000)
        self.assertTrue(abs(estimate - math.pi) < 1e-3)


if __name__ == "__main__":
    unittest.main()
