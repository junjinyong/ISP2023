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

sys.path.append('..')
from lib.camera_v2 import Camera
from lib.robot import Robot
from lib.ros_environment import ROSEnvironment


def onMouse(event, u, v, flags, param):
    pass


def look(robot, camera, host):
    # Get 2d coordinates
    u = (host.left() + host.right()) / 2
    v = (host.top() + host.bottom()) / 2

    # Convert the 2d coordinates to 3d coordinates in camera frame
    (x, y, z) = camera.convert2d_3d(u, v)
    # Convert the 3d coordinates from the camera frame into
    # Gretchen's frame using a transformatio matrix
    (x, y, z) = camera.convert3d_3d(x, y, z)
    # have Gretchen look at that point
    robot.lookatpoint(x, y, z)


def main():
    # Initalize ROS environment
    # Initalize camera and robot
    ROSEnvironment()
    camera = Camera()
    robot = Robot()
    camera.start()
    robot.start()

    # Initialize face recognizer, face database and bluerer
    recognizer = FaceRecognizer()
    blurer = Blurer()
    database = FaceDatabase()

    # Create a window called "Frame" and install a mouse handler
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", onMouse)

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

            indices = list()
            for face in encodings:
                # Calculate the distance to all faces in the database
                # If there is a face whose distance is less than the threshold
                # It is considered to belong to the same person
                index = database.query(face, update=True, insert=True)
                indices.append(index)

            # Determine main person
            host = locations[0]

            # Blur faces
            for face in locations[1:]:
                image = blurer.blur(image, face)

            # Draw boxes
            for (face, index) in zip(locations, indices):
                left, top, right, bottom = face.left(), face.top(), face.right(), face.bottom()
                name = "ID: " + str(index)
                cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

            # Track the main person
            look(robot, camera, host)

        # Display image
        cv2.imshow("Frame", image[..., ::-1])

        # Exit loop if key was pressed
        key = cv2.waitKey(1)
        if key > 0:
            break


#
# Program entry point when started directly
#
if __name__ == '__main__':
    main()
