#
# First Steps in Programming a Humanoid AI Robot
#
# FaceDetector class
# Detect faces using Dlib's 68-point face detector and perform pose estimation
#

import dlib
import imutils
import numpy as np


class FaceDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def detect(self, frame):
        return self.detector(frame, 1)

    def predict(self, frame, bounding_box):
        shape = self.predictor(frame, bounding_box)
        shape = np.array(imutils.face_utils.shape_to_np(shape))

        return shape
