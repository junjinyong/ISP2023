#
# First Steps in Programming a Humanoid AI Robot
#
#
#
#

def dispose(face):
    left = face.left()
    right = face.right()
    top = face.top()
    bottom = face.bottom()

    maximum = max(right - left, top - bottom)
    corner = ((left + right - maximum) // 2, (top + bottom - maximum) // 2)

    return maximum, corner


def findNearest(p, locations):
    minimum: float = 100.0
    index: int = -1
    u, v = p
    for (order, face) in enumerate(locations):
        x = (face.left() + face.right()) / 2
        y = (face.top() + face.bottom()) / 2
        distance = abs(x - u) + abs(y - v)
        print("distance:", distance)
        if distance < minimum:
            minimum = distance
            index = order
    return index


def look(robot, camera, face):
    if face < 0:
        return

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
