#
# First Steps in Programming a Humanoid AI Robot
#
#
#
#

def dispose(face):
    # Get the coordinate of four vertices
    left = face.left()
    right = face.right()
    top = face.top()
    bottom = face.bottom()

    # Calculate coordinate of top left corner
    maximum = max(right - left, top - bottom)
    corner = ((left + right - maximum) // 2, (top + bottom - maximum) // 2)

    return maximum, corner


def findNearest(p, locations):
    # Initialize minimum and index
    minimum: float = 100.0
    index: int = -1
    u, v = p

    # Find the closest face whose distance is less than the tolerance
    for (order, face) in enumerate(locations):
        # Calculate center
        x = (face.left() + face.right()) / 2
        y = (face.top() + face.bottom()) / 2

        # Calculate distance between clicked point and the center of the face
        distance = abs(x - u) + abs(y - v)

        # Update minimum
        if distance < minimum:
            minimum = distance
            index = order

    return index


def look(robot, camera, face):
    # Get 2d coordinates
    u = (face.left() + face.right()) / 2
    v = (face.top() + face.bottom()) / 2

    # Rescale coordinates
    # Countervail the excessive movement of the camera due to the slow feedback
    u = (320 + 320 + u) / 3
    v = (240 + 240 + v) / 3

    # Convert the 2d coordinates to 3d coordinates in camera frame
    (x, y, z) = camera.convert2d_3d(u, v)

    # Convert the 3d coordinates from the camera frame into
    # Gretchen's frame using a transformation matrix
    (x, y, z) = camera.convert3d_3d(x, y, z)

    # have Gretchen look at that point
    robot.lookatpoint(x, y, z, velocity=0.54)
