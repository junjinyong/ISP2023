#
# First Steps in Programming a Humanoid AI Robot
#
# Blurer class
#
#

import numpy as np
from PIL import Image

from utils import dispose

# Image reference:
# https://pixabay.com/vectors/smiley-emoticon-happy-face-icon-1635449/
image_location = r"smiley.png"


class Blurer:
    def __init__(self):
        self.__raw = Image.open(image_location)

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
