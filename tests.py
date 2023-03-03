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
        self.assertEqual(vec_d.shape, (3, 1))
        vec_res = np.array([[1, 2, 3]]).transpose()
        self.assertTrue(np.array_equal(vec_d, vec_res))


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

    def test_rotate(self):
        p = Point3(1, 0, 0)

        # Rotate using a quaternion
        q1 = Quaternion.RzRyRx(np.pi/2, 0, 0)
        q1.normalise()
        p1 = p.rotate(q1)
        self.assertTrue(np.isclose(p1.x, 0))
        self.assertTrue(np.isclose(p1.y, -1))
        self.assertTrue(np.isclose(p1.z, 0))
       
        # Rotate using a rotation matrix
        rot = Rot3.RPY(0, 0, -np.pi/2)
        p2 = p.rotate(rot)

        self.assertTrue(np.isclose(p2.x, 0))
        self.assertTrue(np.isclose(p2.y, 1))
        self.assertTrue(np.isclose(p2.z, 0))

        # Rotate back
        p3 = p2.rotate(rot.inverse())
        self.assertTrue(np.isclose(p3.x, 1))
        self.assertTrue(np.isclose(p3.y, 0))
        self.assertTrue(np.isclose(p3.z, 0))

class TestQuaternion(unittest.TestCase):
    def test_init(self):
        q = Quaternion()
        self.assertEqual(q.w, 1)
        self.assertEqual(q.x, 0)
        self.assertEqual(q.y, 0)
        self.assertEqual(q.z, 0)

        q = Quaternion(1, 2, 3, 4)
        self.assertEqual(q.w, 1)
        self.assertEqual(q.x, 2)
        self.assertEqual(q.y, 3)
        self.assertEqual(q.z, 4)

        q1 = Quaternion.RzRyRx(0, 0, 0)
        self.assertEqual(q1.w, 1)
        self.assertEqual(q1.x, 0)
        self.assertEqual(q1.y, 0)
        self.assertEqual(q1.z, 0)

    def test_normalise(self):
        w = np.random.rand()
        x = np.random.rand()
        y = np.random.rand()
        z = np.random.rand()
        q = Quaternion(w, x, y, z)
        q.normalise()

        norm = np.sqrt(w**2 + x**2 + y**2 + z**2)
        self.assertTrue(np.isclose(q.w, w / norm))
        self.assertTrue(np.isclose(q.x, x / norm))
        self.assertTrue(np.isclose(q.y, y / norm))
        self.assertTrue(np.isclose(q.z, z / norm))

    def test_multiply(self):
        q = Quaternion(np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand())
        q.normalise()
        q_inv = q.inverse()
        q_res = q * q_inv

        self.assertTrue(np.isclose(q_res.w, 1))
        self.assertTrue(np.isclose(q_res.x, 0))
        self.assertTrue(np.isclose(q_res.y, 0))
        self.assertTrue(np.isclose(q_res.z, 0))

        q_res = q_inv * q
        self.assertTrue(np.isclose(q_res.w, 1))
        self.assertTrue(np.isclose(q_res.x, 0))
        self.assertTrue(np.isclose(q_res.y, 0))
        self.assertTrue(np.isclose(q_res.z, 0))

        # Rotate a point
        p = Point3(1, 0, 0)
        q = Quaternion.RzRyRx(np.pi/2, 0, 0)
        q.normalise()
        p_rot = p.rotate(q)
       
        self.assertTrue(np.isclose(p_rot.x, 0))
        self.assertTrue(np.isclose(p_rot.y, -1))
        self.assertTrue(np.isclose(p_rot.z, 0))


class TestRot3(unittest.TestCase):
    def test_init(self):
        rot = Rot3()
        self.assertTrue(np.array_equal(rot, np.identity(n=3, dtype=float)))

        # Use rpy classmethod to initialise Rot3 using roll, pitch, yaw
        rot = Rot3.RPY(0, 0, 0) # Identity
        self.assertTrue(np.array_equal(rot, np.identity(n=3, dtype=float)))

        # Construct from matrix
        rot = Rot3(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        self.assertTrue(np.array_equal(rot, np.identity(n=3, dtype=float)))

        # Construct from quaternion
        q = Quaternion.RzRyRx(0, 0, 0)
        rot = Rot3.Quaternion(q)
        self.assertTrue(np.array_equal(rot, np.identity(n=3, dtype=float)))

    def test_angle_axis(self):
        roll = np.random.rand()
        pitch = np.random.rand()
        yaw = np.random.rand()

        rot = Rot3.RPY(roll, pitch, yaw)

        self.assertTrue(np.isclose(rot.roll(), roll))
        self.assertTrue(np.isclose(rot.pitch(), pitch))
        self.assertTrue(np.isclose(rot.yaw(), yaw))

    def test_inverse(self):
        roll = np.random.rand()
        pitch = np.random.rand()
        yaw = np.random.rand()

        rot = Rot3.RPY(roll, pitch, yaw)
        rot_inv = rot.inverse()

        self.assertTrue(np.array_equal(rot_inv, rot.T))

        I = rot * rot_inv

        # Check equal to 10 decimal places
        self.assertTrue(I == np.identity(n=3, dtype=float))

    def test_mul(self):
        # Rotate a point
        p = Point3(1, 0, 0)
        rot = Rot3.RPY(np.pi/2, 0, 0)
        p_rot = rot * p

    def test_inverse(self):
        rot = Rot3.RPY(np.pi/2, 0, 0)
        rot_inv = rot.inverse()

        self.assertTrue(np.array_equal(rot_inv, rot.T))
        result = rot * rot_inv
        self.assertTrue(np.array_equal(result, np.identity(n=3, dtype=float)))

    def test_rot(self):
        # Rotate point and then rotate back to original
        point = Point3(np.random.rand(), np.random.rand(), np.random.rand())
        rot = Rot3.RPY(np.random.rand(), np.random.rand(), np.random.rand())

        point_rot = rot * point
        point_orig = rot.inverse() * point_rot

        self.assertTrue(np.isclose(point.x, point_orig.x))
        self.assertTrue(np.isclose(point.y, point_orig.y))
        self.assertTrue(np.isclose(point.z, point_orig.z))


class TestPose3(unittest.TestCase):
    def test_init(self):
        pose = Pose3()
        self.assertTrue(np.array_equal(pose.R, np.identity(n=3, dtype=float)))
        self.assertTrue(np.array_equal(pose.t, np.zeros((1, 3), dtype=float)))

        rot = Rot3.RPY(0, 0, 0)
        t = Point3(0, 0, 0)
        pose = Pose3(rot, t)

        self.assertTrue(np.array_equal(pose.R, rot))
        self.assertTrue(np.array_equal(pose.t, t))

        pose2 = Pose3()
        self.assertTrue(pose == pose2)

    def test_inverse(self):
        rot = Rot3.RPY(np.pi/2, 0, 0)
        t = Point3(1, 0, 0)
        pose = Pose3(rot, t)
        pose_inv = pose.inverse()
        result = pose * pose_inv

        self.assertTrue(np.array_equal(result.R, np.identity(n=3, dtype=float)))

    def test_compose(self):
        rotA = Rot3.RPY(0, 0, 0)
        tA = Point3(0, 0, 1)
        poseA = Pose3(rotA, tA)

        rotB = Rot3.RPY(np.pi/2, 0, 0)
        tB = Point3(0, 0.5, 0)
        poseB = Pose3(rotB, tB)

        poseC = poseA.compose(poseB)

    def test_multiply(self):
        rotA = Rot3.RPY(0, 0, 0)
        tA = Point3(0, 0, 1)
        poseA = Pose3(rotA, tA)

        rotB = Rot3.RPY(np.pi/2, 0, 0)
        tB = Point3(0, 0.5, 0)
        poseB = Pose3(rotB, tB)   

        poseC = poseA * poseB
        self.assertTrue(np.array_equal(poseC.R, poseB.R))
        self.assertTrue(poseC.t.x == poseB.t.x + poseA.t.x)
        self.assertTrue(poseC.t.y == poseB.t.y + poseA.t.y)
        self.assertTrue(poseC.t.z == poseB.t.z + poseA.t.z)

    def test_transform(self):
        """ 
        Transform applies translation first then rotation
        """
        rotA = Rot3.RPY(np.pi/4, 0, 0)
        tA = Point3(0, 0, 1)
        poseA = Pose3(rotA, tA)

        rotB = Rot3.RPY(-3*np.pi/4, 0, 0)
        tB = Point3(0, 1, 0)
        poseB = Pose3(rotB, tB)

        T_b_a = poseA.transform_to(poseB)
        rot_eul = T_b_a.R.to_euler()
        trans = T_b_a.t

        self.assertTrue(np.isclose(abs(rot_eul.x), np.pi))
        self.assertTrue(np.isclose(rot_eul.y, 0))
        self.assertTrue(np.isclose(rot_eul.z, 0))

        self.assertTrue(np.isclose(trans.x, 0))
        self.assertTrue(np.isclose(trans.y, 0))
        self.assertTrue(np.isclose(trans.z, -np.sqrt(2)))

class TransformTest(unittest.TestCase):
    def test_init(self):
        transform = Transform("a", "b")
        self.assertTrue(np.array_equal(transform.R, np.identity(n=3, dtype=float)))
        self.assertTrue(np.array_equal(transform.t, np.zeros((1, 3), dtype=float)))

        rot = Rot3.RPY(0, 0, 0)
        t = Point3(0, 0, 0)
        transform2 = Transform("A", "B", t, rot)

        self.assertTrue(np.array_equal(transform2.R, rot))
        self.assertTrue(np.array_equal(transform2.t, t))

        self.assertTrue(transform == transform2)

    def test_inverse(self):
        rot = Rot3.RPY(np.pi/2, 0, 0)
        t = Point3(1, 0, 0)
        transform = Transform("a", "b", t, rot)
        transform_inv = transform.inverse()
        result = transform * transform_inv

        self.assertTrue(np.array_equal(result.matrix(), np.identity(n=4, dtype=float)))


    def test_compose(self):
        rotA = Rot3.RPY(0, 0, 0)
        tA = Point3(0, 0, 1)
        transformA = Transform("a", "b", tA, rotA)

        rotB = Rot3.RPY(np.pi/2, 0, 0)
        tB = Point3(0, 0.5, 0)
        transformB = Transform("b", "c", tB, rotB)

        transformC = transformA.compose(transformB)

        self.assertTrue(np.array_equal(transformC.R, rotA * rotB))
        self.assertTrue(np.array_equal(transformC.t, tA + rotA * tB))

    def test_multiply(self):
        rotA = Rot3.RPY(np.pi/2, 0, 0)
        tA = Point3(0, 0, 1)
        T_a_b = Transform("a", "b", tA, rotA)

        rotA = Rot3.RPY(np.pi/2, 0, 0)
        tA = Point3(0, 0.5, 0)
        PoseA = Pose3(rotA, tA)

        # Transform pose from frame A to frame B
        PoseB = T_a_b * PoseA

        # Apply translation then rotation
        rotB = rotA * rotA
        tB = Point3(0, 0, 1.5)
        self.assertTrue(np.array_equal(PoseB.R, rotB))
        self.assertTrue(np.isclose(PoseB.t.x, tB.x))
        self.assertTrue(np.isclose(PoseB.t.y, tB.y))
        self.assertTrue(np.isclose(PoseB.t.z, tB.z))

        # Transform pose from frame B to frame A
        PoseA_ = T_a_b.inverse() * PoseB

        self.assertTrue(np.array_equal(PoseA_.R, PoseA.R))
        self.assertTrue(np.isclose(PoseA_.t.x, PoseA.t.x))
        self.assertTrue(np.isclose(PoseA_.t.y, PoseA.t.y))
        self.assertTrue(np.isclose(PoseA_.t.z, PoseA.t.z))

if __name__ == "__main__":
    print("Test geometry!")

    unittest.main()
