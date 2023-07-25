#
# First Steps in Programming a Humanoid AI Robot
#
# FaceRecognizer class
#
#

import dlib

pose_predictor_model_location = r"shape_predictor_68_face_landmarks.dat"
face_recognition_model_location = r"dlib_face_recognition_resnet_model_v1.dat"


class FaceRecognizer:
    def __init__(self):
        # Download pretrained models here:
        # https://github.com/ageitgey/face_recognition_models/tree/master/face_recognition_models/models
        self.__detector = dlib.get_frontal_face_detector()
        self.__predictor = dlib.shape_predictor(pose_predictor_model_location)
        self.__rescaler = dlib.get_face_chip
        self.__encoder = dlib.face_recognition_model_v1(face_recognition_model_location).compute_face_descriptor

    def detect(self, frame, upsample_num_times=1):
        return self.__detector(frame, upsample_num_times=upsample_num_times)

    def predict(self, frame, box):
        return self.__predictor(frame, box)
    
    def rescale(self, frame, box):
        return self.__rescaler(frame, box)

    def encode(self, frame, num_jitters=1):
        return self.__encoder(frame, num_jitters=num_jitters)
