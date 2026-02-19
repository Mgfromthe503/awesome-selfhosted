import unittest

import sacred_geometry_toolkit as sgt


class SacredGeometryToolkitTests(unittest.TestCase):
    def test_text_codes(self):
        out = sgt.text_to_sacred_geometry_codes("AB")
        self.assertEqual(len(out), 2)
        self.assertTrue(out[0]["code"].startswith("SG-"))

    def test_platonic_cube(self):
        cube = sgt.generate_platonic_solid("cube", scale=2)
        self.assertEqual(len(cube["vertices"]), 8)
        self.assertEqual(cube["solid"], "cube")

    def test_astro_translation(self):
        mapped = sgt.translate_astrological_numbers([1, 12, 13])
        self.assertEqual(mapped[0]["shape"], "circle")
        self.assertEqual(mapped[2]["shape"], "circle")

    def test_resonance(self):
        freq = sgt.calculate_resonance_frequency("square", 0.5)
        self.assertGreater(freq, 0)

    def test_model_ops(self):
        model = sgt.SacredGeometryModel.from_platonic("tetrahedron")
        initial = model.vertices[0]
        model.scale(2)
        self.assertNotEqual(initial, model.vertices[0])
        model.rotate_z(45)
        model.translate(1, 1, 1)
        self.assertEqual(len(model.vertices), 4)


if __name__ == "__main__":
    unittest.main()
