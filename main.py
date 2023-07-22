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

from recognizer import FaceRecognizer
from blurer import Blurer
from database import FaceDatabase
from utils import look, findNearest

sys.path.append('..')
from lib.camera_v2 import Camera
from lib.robot import Robot
from lib.ros_environment import ROSEnvironment


host: int = -1
locations: list = list()
database: FaceDatabase = FaceDatabase()


def onMouse(self, event, u, v, flags, param=None):
    # Refer to global variables
    global host, locations, database

    # Calculate the closest person to the given point
    index: int = findNearest(u, v, locations)
    if not index:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        print("Toggle\n")
        database.toggle(index)
    elif event == cv2.EVENT_RBUTTONDOWN:
        print("Set main person\n")
        host = index


def main():
    # Refer to global variables
    global host, locations, database

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

            indices: list = list()
            protection: list = list()
            for face in encodings:
                # Calculate the distance to all faces in the database
                # If there is a face whose distance is less than the threshold
                # It is considered to belong to the same person
                index, blur = database.query(face, update=True, insert=True)
                indices.append(index)
                protection.append(blur)

            # Blur faces
            for (face, blur) in zip(locations, protection):
                if blur:
                    image = blurer.blur(image, face)

            # Draw boxes
            for (face, index) in zip(locations, indices):
                left, top, right, bottom = face.left(), face.top(), face.right(), face.bottom()
                name = "ID: " + str(index)
                color = (255, 0, 0) if index == host else (0, 0, 255)
                cv2.rectangle(image, (left, top), (right, bottom), color, 2)
                cv2.putText(image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

            # Track the main person
            look(robot, camera, host)

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
