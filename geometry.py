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

 

class Pose3:
    def __init__(self):
        self.rotation = Rot3()
        self.translation = Point3()


