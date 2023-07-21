#
# First Steps in Programming a Humanoid AI Robot
#
# FaceDatabase class
#
#

import numpy as np


class FaceDatabase:
    def __init__(self, decay=0.1):
        self.__data = list()
        self.__size = 0
        self.__decay = decay

    def __get_distance(self, face):
        return np.linalg.norm(self.__data - face, axis=1) if self.__size > 0 else np.empty(0)

    def query(self, face):
        minimum = float("inf")
        index = -1
        if self.__size > 0:
            distances = self.__get_distance(face)
            minimum = min(distances)
            index = np.argmin(distances)
        return minimum, index

    def insert(self, face):
        self.__data.append(face)
        size = self.__size
        self.__size = size + 1
        return size

    def update(self, index, face):
        self.__data[index] = self.__data[index] * (1.0 - self.__decay) + face * self.__decay
