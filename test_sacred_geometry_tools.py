import unittest

import numpy as np

import sacred_geometry_tools as sgt


class SacredGeometryTests(unittest.TestCase):
    def test_fibonacci_sequence(self):
        self.assertEqual(sgt.fibonacci_sequence(7), [0, 1, 1, 2, 3, 5, 8])

    def test_flower_center_count(self):
        self.assertEqual(len(sgt.flower_of_life_centers(rings=2)), 19)

    def test_seed_centers(self):
        self.assertEqual(len(sgt.seed_of_life_centers()), 7)

    def test_solids_shapes(self):
        self.assertEqual(sgt.cube().vertices.shape, (8, 3))
        self.assertEqual(sgt.octahedron().vertices.shape, (6, 3))
        self.assertEqual(sgt.icosahedron().vertices.shape, (12, 3))

    def test_torus_shape(self):
        x, y, z = sgt.torus(nu=20, nv=10)
        self.assertEqual(x.shape, (10, 20))
        self.assertEqual(y.shape, (10, 20))
        self.assertEqual(z.shape, (10, 20))


if __name__ == "__main__":
    unittest.main()
