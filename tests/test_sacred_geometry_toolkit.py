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

    def test_astro_translation_extended_cycle(self):
        mapped = sgt.translate_astrological_numbers([1, 12, 13, 36, 37])
        self.assertEqual(mapped[0]["shape"], "circle")
        self.assertEqual(mapped[1]["shape"], "dodecagon")
        self.assertEqual(mapped[2]["shape"], "flower of life")
        self.assertEqual(mapped[3]["shape"], "star tetrahedron")
        self.assertEqual(mapped[4]["shape"], "circle")

    def test_shape_knowledge_is_large(self):
        shapes = sgt.list_supported_shapes()
        self.assertGreaterEqual(len(shapes), 40)
        self.assertIn("dodecahedron", shapes)

    def test_symbolic_profile(self):
        profile = sgt.get_symbolic_profile("metatron's cube")
        self.assertEqual(profile["type"], "symbol")
        self.assertEqual(profile["value"], 2)

    def test_training_record_generation(self):
        records = sgt.generate_symbolic_training_records()
        self.assertGreater(len(records), 45)
        self.assertTrue(any(r["source"] == "shape_knowledge" for r in records))
        self.assertTrue(any(r["source"] == "symbol_knowledge" for r in records))

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
