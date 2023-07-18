#
# First Steps in Programming a Humanoid AI Robot
#
# Blur faces
#
#
#

# Import required modules
import sys
import cv2
import imutils
import dlib
import numpy as np
from facedetector import FaceDetector

sys.path.append('..')
from lib.camera_v2 import Camera
from lib.robot import Robot
from lib.ros_environment import ROSEnvironment


def onMouse(event, u, v, flags, param):
    pass


def main():
    # Initalize ROS environment
    ROSEnvironment()

    # Initalize camera and robot
    camera = Camera()
    robot = Robot()
    camera.start()
    robot.start()

    # Initialize Face Detector
    facedetector = FaceDetector()

    # Create a window called "Frame" and install a mouse handler
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", onMouse)

    # Loop
    while True:
        # Get image from camera
        img = camera.getImage()

        # Detect faces from image
        faces = facedetector.detect(img)

        if faces:
            # Determine main person
            face = faces[0]

            # Draw box
            cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)

        # Display image
        cv2.imshow("Frame", img[..., ::-1])

        # Exit loop if key was pressed
        key = cv2.waitKey(1)
        if key > 0:
            break


#
# Program entry point when started directly
#
if __name__ == '__main__':
    main()
