#
# First Steps in Programming a Humanoid AI Robot
#
#
#
#
#

# Import required modules
import sys

import cv2
import numpy as np

from blurer import Blurer
from database import FaceDatabase
from recognizer import FaceRecognizer
from utils import look, findNearest

sys.path.append('..')
from lib.camera_v2 import Camera
from lib.robot import Robot
from lib.ros_environment import ROSEnvironment

# Signal indicates the type of mouse event
# Point indicates the clicked coordinate
signal: int = 0
point: tuple = (None, None)


def onMouse(event, u, v, flags, param=None):
    # Refer to global variables
    global signal, point

    # Save coordinate of clicked point
    point = (u, v)

    # Number each action
    if event == cv2.EVENT_LBUTTONDOWN:
        signal = 1
    elif event == cv2.EVENT_RBUTTONDOWN:
        signal = 2


def main():
    # Refer to global variables
    global signal, point

    # Initalize ROS environment
    # Initalize camera and robot
    ROSEnvironment()
    camera = Camera()
    robot = Robot()
    camera.start()
    robot.start()

    # Initialize face recognizer and blurer
    recognizer: FaceRecognizer = FaceRecognizer()
    blurer: Blurer = Blurer()
    database: FaceDatabase = FaceDatabase()
    host: int = -1

    # Create a window called "Frame" and install a mouse handler
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", onMouse, param=None)

    # Loop
    while True:
        # Get image from camera
        image = camera.getImage()

        # Detect faces from image
        locations = recognizer.detect(image, upsample_num_times=1)

        if locations:
            # Calculate face encoding
            landmarks = [recognizer.predict(image, location) for location in locations]
            encodings = [np.array(recognizer.encode(image, landmark, num_jitters=1)) for landmark in landmarks]

            # Calculate the distance to all faces in the database
            # If there is a face whose distance is less than the threshold
            # It is considered to belong to the same person
            indices: list = list()
            protection: list = list()
            owner: int = -1
            for (order, face) in enumerate(encodings):
                if database.compare(face, host):
                    owner = order
                index, blur = database.query(face, update=True, insert=True)
                indices.append(index)
                protection.append(blur)

            # Process click events
            if signal:
                # Find face nearest to the clicked point
                target = findNearest(point, locations)

                # Query the found face to the database to obtain DB index
                face = encodings[target]
                index = database.query(face, update=False, insert=False)[0]

                # invert whether to blur or not if left button down
                # change main person if right button down
                if signal == 1:
                    database.toggle(index)
                elif signal == 2:
                    host = index

                # Reset action number
                signal = 0

            # Blur faces
            for (face, blur) in zip(locations, protection):
                if blur:
                    image = blurer.blur(image, face)

            # Draw boxes
            for (face, index) in zip(locations, indices):
                left, top, right, bottom = face.left(), face.top(), face.right(), face.bottom()
                name = "ID: " + str(index)
                color = (255, 0, 0) if index == owner else (0, 0, 255)
                cv2.rectangle(image, (left, top), (right, bottom), color, 2)
                cv2.putText(image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

            # Track the main person
            if owner >= 0:
                look(robot, camera, locations[owner])

        else:
            # Reset action number
            signal = 0

        # Display image
        cv2.imshow("Frame", image[..., ::-1])

        # Exit loop if key was pressed
        if cv2.waitKey(1) > 0:
            break


#
# Program entry point when started directly
#
if __name__ == '__main__':
    main()
