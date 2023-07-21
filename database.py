#
# First Steps in Programming a Humanoid AI Robot
#
# FaceDatabase class
#
#

import numpy as np


class FaceDatabase:
    def __init__(self, threshold=0.5, decay=0.1):
        self.__data = list()
        self.__blur = list()
        self.__size = 0
        self.__threshold = threshold
        self.__decay = decay

    def query(self, face, update=True, insert=True):
        # Initialize minimum and index
        minimum = float("inf")
        index = -1

        # Calculate minimum and index
        if self.__size > 0:
            distances = np.linalg.norm(self.__data - face, axis=1)
            minimum = min(distances)
            index = np.argmin(distances)

        # Flag indicates whether the face already exists
        flag = minimum <= self.__threshold

        # Update face in the database
        if update and flag:
            self.__data[index] * (1.0 - self.__decay) + face * self.__decay

        # Insert new face to the database
        if insert and not flag:
            self.__data.append(face)
            index = self.__size
            self.__size = index + 1

        return index

    def get_size(self):
        return self.__size
