#
# First Steps in Programming a Humanoid AI Robot
#
#
#
#

def findNearest(u, v, locations):
    minimum: float = 100.0
    index: int = -1
    for (number, face) in enumerate(locations):
        x = (face.left() + face.right()) / 2
        y = (face.top() + face.bottom()) / 2
        distance = abs(x - u) + abs(y - v)
        if distance < minimum:
            minimum = distance
            index = number
    return index


def look(robot, camera, host):
    if host < 0:
        return

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
