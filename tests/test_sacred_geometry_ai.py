import math
import unittest

from sacred_geometry_ai import Vector11D


class Vector11DTests(unittest.TestCase):
    def test_requires_exact_dimension(self):
        with self.assertRaises(ValueError):
            Vector11D(1, 2)

    def test_project_onto_sphere_zero_vector(self):
        zero = Vector11D(*([0] * 11))
        projected = zero.project_onto_sphere(5)
        self.assertEqual(projected.to_list(), [0.0] * 11)

    def test_project_onto_sphere_radius(self):
        vec = Vector11D(*range(1, 12))
        projected = vec.project_onto_sphere(5)
        self.assertAlmostEqual(projected.norm(), 5.0, places=7)

    def test_fractal_triangle_points_grow(self):
        vec = Vector11D(*range(1, 12))
        points = vec.fractal_nature_points("triangle", iterations=4)
        self.assertEqual(len(points), 1 + 4 * 3)

    def test_fractal_shape_validation(self):
        vec = Vector11D(*range(1, 12))
        with self.assertRaises(ValueError):
            vec.fractal_nature_points("octahedron", iterations=1)

    def test_cosine_similarity_zero_safe(self):
        zero = Vector11D(*([0] * 11))
        vec = Vector11D(*([1] * 11))
        self.assertEqual(zero.cosine_similarity(vec), 0.0)


if __name__ == "__main__":
    unittest.main()
