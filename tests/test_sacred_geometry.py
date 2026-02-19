import unittest

from sacred_geometry import sacred_geometry_code, sacred_symbol_value


class SacredGeometryTests(unittest.TestCase):
    def test_generates_expected_code_order(self):
        text = "Draw a circle around the square and connect it to the triangle"
        self.assertEqual(sacred_geometry_code(text), "001003002")

    def test_ignores_case_and_punctuation(self):
        text = "Circle, TRIANGLE; and square!"
        self.assertEqual(sacred_geometry_code(text), "001002003")

    def test_symbol_lookup_is_case_insensitive(self):
        self.assertEqual(sacred_symbol_value("Flower of Life"), 1)
        self.assertEqual(sacred_symbol_value("sri yantra"), 3)

    def test_unknown_symbol_raises_keyerror(self):
        with self.assertRaises(KeyError):
            sacred_symbol_value("Unknown Symbol")


if __name__ == "__main__":
    unittest.main()
