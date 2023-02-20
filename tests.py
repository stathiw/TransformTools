#!/usr/bin/python

import numpy as np
import unittest

from geometry import *

class TestVector(unittest.TestCase):
    def test_init(self):
        vec = Vector(np.array([[1, 2, 3]]))
        # Check values
        self.assertEqual(vec[0][0], 1)
        self.assertEqual(vec[0][1], 2)
        self.assertEqual(vec[0][2], 3)

    def test_transpose(self):
        vec = Vector(np.array([[1, 2, 3]]))
        vec_t = vec.transpose()

        self.assertEqual(vec_t.shape, (3, 1))

    def test_dot_product(self):
        # Vector multiplication
        vec_a = Vector(np.array([[1, 2, 3]]))
        vec_b = Vector(np.array([[2, 2, 2]]))
        vec_c = vec_a * vec_b.transpose()

        self.assertEqual(vec_c.shape, (1, 1))
        self.assertEqual(vec_c[0][0], 12)

        # Matrix multiplication
        matrix_eye = np.identity(n=3, dtype=float)
        vec_c = vec_a * matrix_eye

        self.assertEqual(vec_c.shape, (1, 3))
        self.assertEqual(vec_c[0][0], 1)
        self.assertEqual(vec_c[0][1], 2)
        self.assertEqual(vec_c[0][2], 3)

        vec_d = matrix_eye * vec_a.transpose()
        self.assertEqual(vec_d.shape, (3, 3))
        matrix_res = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        self.assertTrue(np.array_equal(vec_d, matrix_res))


class TestPoint3(unittest.TestCase):
    def test_init(self):
        # Default constructor
        p = Point3()
        # Check values
        self.assertEqual(p[0][0], 0)
        self.assertEqual(p[0][1], 0)
        self.assertEqual(p[0][2], 0)

        p = Point3(1, 2, 3)
        # Check values
        self.assertEqual(p[0][0], 1)
        self.assertEqual(p[0][1], 2)
        self.assertEqual(p[0][2], 3)

        # Get x, y, z
        self.assertEqual(p.x, 1)
        self.assertEqual(p.y, 2)
        self.assertEqual(p.z, 3)

        # Set x, y, z
        p.x = 4.5
        p.y = 5.5
        p.z = 6.5
        self.assertEqual(p.x, 4.5)
        self.assertEqual(p.y, 5.5)
        self.assertEqual(p.z, 6.5)

    def test_arithmetic(self):
        # Test addition
        p = Point3(1, 2, 3)
        p2 = Point3(4, 5, 6)
        p3 = p + p2

        self.assertEqual(p3.x, 5)
        self.assertEqual(p3.y, 7)
        self.assertEqual(p3.z, 9)

        # Test subtraction
        p3 = p - p2
        
        self.assertEqual(p3.x, -3)
        self.assertEqual(p3.y, -3)
        self.assertEqual(p3.z, -3)

        # Test multiplication
        p3 = p * 2
        self.assertEqual(p3.x, p.x * 2)
        self.assertEqual(p3.y, p.y * 2)
        self.assertEqual(p3.z, p.z * 2)

        p3 = 2 * p
        self.assertEqual(p3.x, p.x * 2)
        self.assertEqual(p3.y, p.y * 2)
        self.assertEqual(p3.z, p.z * 2)

    def test_dot_product(self):
        p = Point3(1, 2, 3)
        p2 = Point3(4, 5, 6)
        dot = p.dot(p2)

        self.assertEqual(dot, 32)

    def test_cross_product(self):
        p = Point3(1, 2, 3)
        p2 = Point3(4, 5, 6)
        cross = p.cross(p2)

        self.assertEqual(cross.x, -3)
        self.assertEqual(cross.y, 6)
        self.assertEqual(cross.z, -3)

    def test_norm(self):
        # Get random point
        p = Point3(np.random.rand(), np.random.rand(), np.random.rand())
        norm = np.sqrt(p.x**2 + p.y**2 + p.z**2)

        self.assertEqual(norm, p.norm())

    def test_distance(self):
        p1 = Point3(np.random.rand(), np.random.rand(), np.random.rand())
        p2 = Point3(np.random.rand(), np.random.rand(), np.random.rand())
        dist = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

        self.assertEqual(dist, p1.distance(p2))

if __name__ == "__main__":
    print("Test geometry!")

    unittest.main()