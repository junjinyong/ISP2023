#
# First Steps in Programming a Humanoid AI Robot
#
# FaceDatabase class
#
#

import numpy as np


class FaceDatabase:
    def __init__(self, threshold=0.6, decay=0.1):
        self.__data = list()
        self.__blur = list()
        self.__threshold = threshold
        self.__decay = decay

    def compare(self, face, index):
        if 0 <= index < self.get_size():
            return np.linalg.norm(self.__data[index] - face) <= self.__threshold
        else:
            return False

    def query(self, face, update=True, insert=True):
        # Initialize minimum and index
        minimum: float = float("inf")
        index: int = -1
        blur: bool = False

        # Calculate minimum and index
        if len(self.__data) > 0:
            distances = np.linalg.norm(self.__data - face, axis=1)
            minimum = min(distances)
            index = int(np.argmin(distances))

        # Flag indicates whether the face already exists
        flag = minimum <= self.__threshold

        # Whether to blur or not
        if flag:
            blur = self.__blur[index]

        # Update face in the database
        if update and flag:
            self.__data[index] = self.__data[index] * (1.0 - self.__decay) + face * self.__decay

        # Insert new face to the database
        if insert and not flag:
            index = len(self.__data)
            self.__data.append(face)
            self.__blur.append(False)

        return index, blur

    def get_size(self):
        return len(self.__data)

    def toggle(self, index):
        if 0 <= index < self.get_size():
            self.__blur[index] = not self.__blur[index]
