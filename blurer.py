#
# First Steps in Programming a Humanoid AI Robot
#
# Blurer class
#
#

from PIL import Image
import numpy as np
from utils import dispose


class Blurer:
    def __init__(self):
        self.__raw = Image.open(r"./smiley.png")

    def blur(self, img, face):
        # Make into PIL image
        result = Image.fromarray(img)

        # Get a drawing context
        (size, pos) = dispose(face)
        icon = self.__raw.resize((size, size))

        # Draw emoji on face
        result.paste(icon, pos, mask=icon)

        # Convert back to OpenCV image
        result = np.array(result)

        return result
