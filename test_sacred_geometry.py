import math
import unittest

from sacred_geometry import (
    CODING_SYSTEM,
    GEOMETRY_DICT,
    SYMBOL_CODES,
    Vector7D,
    map_to_symbol,
)


class TestSacredGeometryMappings(unittest.TestCase):
    def test_provided_mappings_are_present(self):
        self.assertEqual(GEOMETRY_DICT["Seed of Life"], "SOL")
        self.assertEqual(SYMBOL_CODES["circle"], "C1")
        self.assertEqual(CODING_SYSTEM["Sri Yantra"], "04")

    def test_map_to_symbol(self):
        text = "Seed of life and Sri Yantra patterns"
        self.assertEqual(map_to_symbol(text), ["01", "04"])


class TestVector7D(unittest.TestCase):
    def setUp(self):
        self.v1 = Vector7D(1, 2, 3, 4, 5, 6, 7)
        self.v2 = Vector7D(2, 4, 6, 8, 10, 12, 14)

    def test_basic_operations(self):
        self.assertEqual((self.v1 + self.v2).to_list(), [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0])
        self.assertEqual((self.v2 - self.v1).to_list(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertEqual((self.v1 * 2).to_list(), [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0])

    def test_metric_operations(self):
        self.assertEqual(self.v1.dot_product(self.v2), 280.0)
        self.assertTrue(math.isclose(self.v1.norm(), math.sqrt(140)))
        self.assertTrue(math.isclose(self.v1.distance(self.v2), math.sqrt(140)))
        self.assertTrue(math.isclose(self.v1.cosine_similarity(self.v2), 1.0))

    def test_normalize_zero_vector(self):
        self.assertEqual(Vector7D(0, 0, 0, 0, 0, 0, 0).normalize().to_list(), [0.0] * 7)


if __name__ == "__main__":
    unittest.main()
