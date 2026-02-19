import math
import unittest

from sacred_geometry_ai import (
    SacredGeometryAI,
    Vector11D,
    build_feature_vector,
    generate_fractal_points,
)


class Vector11DTests(unittest.TestCase):
    def test_projection_onto_sphere(self):
        vector = Vector11D(*range(1, 12))
        projected = vector.project_onto_sphere(5)
        self.assertAlmostEqual(projected.norm(), 5.0, places=7)

    def test_projection_of_zero_vector(self):
        vector = Vector11D(*([0] * 11))
        projected = vector.project_onto_sphere(4)
        self.assertEqual(projected.to_list(), [0.0] * 11)

    def test_cosine_similarity_handles_zero_vector(self):
        a = Vector11D(*([0] * 11))
        b = Vector11D(*([1] * 11))
        self.assertEqual(a.cosine_similarity(b), 0.0)


class FractalTests(unittest.TestCase):
    def test_generate_triangle_fractal_points(self):
        points = generate_fractal_points("triangle", 20, seed=1)
        self.assertEqual(len(points), 20)
        self.assertTrue(all(math.isfinite(x) and math.isfinite(y) for x, y in points))

    def test_unsupported_shape_raises(self):
        with self.assertRaises(ValueError):
            generate_fractal_points("octagon", 10)


class SacredGeometryAITests(unittest.TestCase):
    def test_predict_shape(self):
        model = SacredGeometryAI()
        sample = Vector11D(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        label, score = model.predict_shape(sample)
        self.assertEqual(label, "sphere")
        self.assertGreaterEqual(score, 0.99)

    def test_harmony_score_is_bounded(self):
        model = SacredGeometryAI()
        sample = build_feature_vector([0.2] * 11)
        score = model.harmony_score(sample)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
