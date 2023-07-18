#
# First Steps in Programming a Humanoid AI Robot
#
# FaceDetector class
# Detect faces using Dlib's 68-point face detector and perform pose estimation
#

import dlib


class FaceDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def detect(self, frame):
        return self.detector(frame, 1)
