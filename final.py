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
from blurer import Blurer

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

    # Initialize face detector and bluerer
    facedetector = FaceDetector()
    blurer = Blurer()

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

            # Find facial landmark
            # shape = facedetector.predictor(img, face)
            # num = 34
            # x = shape.part(num).x
            # y = shape.part(num).y

            # Blur face
            for face in faces[1:]:
                img = blurer.blur(img, face)

            # Draw box
            cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)

            # Track the main person
            # Get 2d coordinates
            u = (face.left() + face.right()) / 2
            v = (face.top() + face.bottom()) / 2

            # Convert the 2d coordinates to 3d coordinates in camera frame
            (x, y, z) = camera.convert2d_3d(u, v)
            # Convert the 3d coordinates from the camera frame into
            # Gretchen's frame using a transformatio matrix
            (x, y, z) = camera.convert3d_3d(x, y, z)
            # have Gretchen look at that point
            robot.lookatpoint(x, y, z)

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
