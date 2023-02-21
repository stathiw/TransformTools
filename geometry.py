#!/usr/bin/python

"""geometry.py: Contains class definitions for geometry objects

"""

__author__ = "Stathi Weir"

__license__ = "BSD-3-Clause"
__version__ = "0.0.1"
__maintainer__ = "Stathi Weir"
__email__ = "stathi.weir@gmail.com"
__status__ = "Development"

import numpy as np


class Vector(np.ndarray):
    """
    Class Vector which inherits from numpy.ndarray and is 1xN dimensional
    """
    def __new__(cls, input_array):
        return np.asarray(input_array, dtype=float).view(cls)

    def __array_finalize__(self, obj) -> None:
        """
        This method is called when a new instance of Vector is created
        """
        if obj is None: return

    def __str__(self):
        """
        Returns vector as string
        """
        return str(self.view(np.ndarray))

    def __mul__(self, arr):
        """ Returns Vector

        Arguments:
            self (Vector): Vector to multiply (1xN dimensional)
            arr (np.ndarray): Array to multiply with (NxP) dimensional

        Returns:
            Vector: Result of dot product
        """
        if len(arr.shape) != 2:
            print("Err: multiplying object must be 2 dimensional")
            return
        
        # If vector is 1xN dimensional, then arr must be NxP dimensional
        if arr.shape[0] != self.shape[1]:
            print("multiplying object must be of shape ({}, N)".format(self.shape[1]))
            return

        # Calculate dot product
        tmp = np.zeros(shape=(1, arr.shape[1]))
        for i in range(arr.shape[1]):
            for j in range(self.shape[1]):
                tmp[0][i] += self.view(np.ndarray)[0][j] * arr[j][i]
    
        return Vector(tmp)

    def __rmul__(self, arr):
        """ Returns Vector

        Arguments:
            arr (np.ndarray): Array to multiply with

        Returns:
            Vector: Result of dot product
        """
        if len(arr.shape) != 2:
            print("Err: multiplying object must be 2 dimensional")
            return
        
        # If vector is 1xN dimensional, then arr must be NxP dimensional
        if arr.shape[1] != self.shape[0]:
            print("multiplying object must be of shape (N, {})".format(self.shape[0]))
            return

        # Calculate dot product
        tmp = np.zeros(shape=(arr.shape[1], self.shape[0]))
        for i in range(arr.shape[1]):
            for j in range(self.shape[0]):
                tmp[i][j] += self.view(np.ndarray)[j][0] * arr[j][i]

        return Vector(tmp)

    def transpose(self):
        return Vector(self.view(np.ndarray).T)



class Point3(Vector):
    """
    Class Point3 which inherits from Vector to represent a 3D point
    """
    def __new__(cls, x=0.0, y=0.0, z=0.0):
        return Vector(np.array([[x, y, z]], dtype=float)).view(cls)

    @property
    def x(self):
        return self.view(np.ndarray)[0][0]

    @property
    def y(self):
        return self.view(np.ndarray)[0][1]

    @property
    def z(self):
        return self.view(np.ndarray)[0][2]

    def __str__(self):
        return str(self.view(np.ndarray))

    @x.setter
    def x(self, value):
        self.view(np.ndarray)[0][0] = value

    @y.setter
    def y(self, value):
        self.view(np.ndarray)[0][1] = value

    @z.setter
    def z(self, value):
        self.view(np.ndarray)[0][2] = value

    def __add__(self, other):
        return Point3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Point3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        return Point3(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other):
        return Point3(self.x * other, self.y * other, self.z * other)

    def dot(self, other):
        """
        Returns the dot product of two points
        """
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        """
        Returns the cross product of two points
        """
        return Point3(self.y * other.z - self.z * other.y,
                      self.z * other.x - self.x * other.z,
                      self.x * other.y - self.y * other.x)

    def distance(self, other):
        """
        Returns the distance between two points
        """
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)

    def norm(self):
        """
        Returns the norm of the point
        """
        return np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)


class Quaternion(Vector):
    """
    Class Quaternion which inherits from Vector to represent a quaternion
    """
    def __new__(cls, w=1.0, x=0.0, y=0.0, z=0.0):
        return Vector(np.array([[w, x, y, z]], dtype=float)).view(cls)

    @classmethod
    def RPY(cls, roll, pitch, yaw):
        """
        Construct Quaternion given Euler angles
        """
        cr = np.cos(roll / 2)
        sr = np.sin(roll / 2)
        cp = np.cos(pitch / 2)
        sp = np.sin(pitch / 2)
        cy = np.cos(yaw / 2)
        sy = np.sin(yaw / 2)

        return Quaternion(w=cr * cp * cy + sr * sp * sy,
                          x=sr * cp * cy - cr * sp * sy,
                          y=cr * sp * cy + sr * cp * sy,
                          z=cr * cp * sy - sr * sp * cy)

    # Comparison operators
    def __eq__(self, other):
        return np.array_equal(self.view(np.ndarray), other.view(np.ndarray))

    def __ne__(self, other):
        return not np.array_equal(self.view(np.ndarray), other.view(np.ndarray))

    @property
    def w(self):
        return self.view(np.ndarray)[0][0]

    @property
    def x(self):
        return self.view(np.ndarray)[0][1]

    @property
    def y(self):
        return self.view(np.ndarray)[0][2]

    @property
    def z(self):
        return self.view(np.ndarray)[0][3]

    @w.setter
    def w(self, value):
        self.view(np.ndarray)[0][0] = value

    @x.setter
    def x(self, value):
        self.view(np.ndarray)[0][1] = value

    @y.setter
    def y(self, value):
        self.view(np.ndarray)[0][2] = value

    @z.setter
    def z(self, value):
        self.view(np.ndarray)[0][3] = value

    def __str__(self):
        return str(self.view(np.ndarray))

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(w=self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
                              x=self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
                              y=self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
                              z=self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w)
        elif isinstance(other, Vector):
            return self * Quaternion(0, other.x, other.y, other.z) * self.inverse()
        else:
            return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other):
        return self * other

    def inverse(self):
        """
        Returns the inverse of the quaternion
        """
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def norm(self):
        """
        Returns the norm of the quaternion
        """
        return np.sqrt(self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalise(self):
        """
        Normalises the quaternion
        """
        norm = self.norm()
        # Divide each element by the norm
        self.w /= norm
        self.x /= norm
        self.y /= norm
        self.z /= norm

    def getEulerAngles(self):
        """
        args:
            self (Rot3): Rot3 to get Euler angles from (3x3 dimensional)
        returns:
            Point3: Point3 containing Euler angles in radians

        Returns Point3 containing Euler angles in radians
        """
        roll = np.arctan2(2 * (self.w * self.x + self.y * self.z), 1 - 2 * (self.x ** 2 + self.y ** 2))
        pitch = np.arcsin(2 * (self.w * self.y - self.z * self.x))
        yaw = np.arctan2(2 * (self.w * self.z + self.x * self.y), 1 - 2 * (self.y ** 2 + self.z ** 2))

        return Point3(roll, pitch, yaw)


class Rot3(np.ndarray):
    """
    Class Rot3 which inherits from numpy.ndarray to represent a 3D rotation matrix
    """
    def __new__(cls, input_array=np.identity(3)):
        return np.asarray(input_array, dtype=float).view(cls)

    @classmethod
    def test_method(cls, i):
        print("test {}".format(i))

    @classmethod
    def RPY(cls, roll, pitch, yaw):
        """
        Constructs a rotation matrix from roll, pitch and yaw
        """
        R_roll = np.array([[1, 0, 0],
                           [0, np.cos(roll), -np.sin(roll)],
                           [0, np.sin(roll), np.cos(roll)]])
        R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                            [0, 1, 0],
                            [-np.sin(pitch), 0, np.cos(pitch)]])
        R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                          [np.sin(yaw), np.cos(yaw), 0],
                          [0, 0, 1]])

        return cls(R_yaw @ R_pitch @ R_roll)

    @classmethod
    def Matrix(cls, R):
        """
        Constructs a Rot3 from a 3x3 matrix
        """
        return cls(R)

    @classmethod
    def Quaternion(cls, q):
        """
        Constructs a Rot3 from a quaternion
        """
        return cls(np.array([[1 - 2 * q.y ** 2 - 2 * q.z ** 2, 2 * q.x * q.y - 2 * q.z * q.w, 2 * q.x * q.z + 2 * q.y * q.w],
                             [2 * q.x * q.y + 2 * q.z * q.w, 1 - 2 * q.x ** 2 - 2 * q.z ** 2, 2 * q.y * q.z - 2 * q.x * q.w],
                             [2 * q.x * q.z - 2 * q.y * q.w, 2 * q.y * q.z + 2 * q.x * q.w, 1 - 2 * q.x ** 2 - 2 * q.y ** 2]]))


    def __array_finalize__(self, obj) -> None:
        """
        This method is called when a new instance of Rot3 is created
        """
        if obj is None: return

    def __str__(self):
        """
        Returns Rot3 as string
        """
        return str(self.view(np.ndarray))

    def __mul__(self, arr):
        """ Returns Rot3

        Arguments:
            self (Rot3): Rot3 to multiply (3x3 dimensional)
            arr (np.ndarray): Array to multiply with (3x3) dimensional

        Returns:
            Rot3: Result of dot product
        """
        if len(arr.shape) != 2:
            print("Err: multiplying object must be 2 dimensional")
            return
        
        # If Rot3 is 3x3 dimensional, then arr must be 3x3 dimensional
        if arr.shape[0] != self.shape[1]:
            print("multiplying object must be of shape ({}, 3)".format(self.shape[1]))
            return

        # Calculate dot product
        tmp = np.zeros(shape=(3, 3))
        for i in range(arr.shape[1]):
            for j in range(self.shape[1]):
                tmp[i][j] += self.view(np.ndarray)[i][j] * arr[j][i]

        return Rot3(tmp)

    def __rmul__(self, arr):
        """ Returns Rot3

        Arguments:
            arr (np.ndarray): Array to multiply with

        Returns:
            Rot3: Result of dot product
        """
        if len(arr.shape) != 2:
            print("Err: multiplying object must be 2 dimensional")
            return
        
        # If Rot3 is 3x3 dimensional, then arr must be 3x3 dimensional
        if arr.shape[1] != self.shape[0]:
            print("multiplying object must be of shape (3, {})".format(self.shape[0]))
            return



class Pose3:
    def __init__(self):
        self.rotation = Rot3()
        self.translation = Point3()


