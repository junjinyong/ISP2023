#
# First Steps in Programming a Humanoid AI Robot
#
# Blurer class
#
#

from PIL import Image, ImageFont, ImageDraw
import numpy as np
import cv2
import emoji


def dispose(face):
    left = face.left()
    right = face.right()
    top = face.top()
    bottom = face.bottom()

    maximum = max(right - left, top - bottom)
    corner = ((left + right - maximum) // 2, (top + bottom - maximum) // 2)

    return maximum, corner


class Blurer:
    def __init__(self):
        self.raw = Image.open(r"./smiley.png")

    def blur(self, img, face):
        # Make into PIL image
        result = Image.fromarray(img)

        # Get a drawing context
        (size, pos) = dispose(face)
        icon = self.raw.resize(size)

        # Draw emoji on face
        result.paste(icon, pos, mask=icon)

        # Convert back to OpenCV image
        result = np.array(result)

        return result

