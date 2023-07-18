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


class Blurer:
    def __init__(self):
        self.font = ImageFont.truetype("Arial Unicode.ttf", 64)
        self.tick = str(emoji.emojize(':grinning_face_with_big_eyes:'))

    def blur(self, img, pos):
        # Make into PIL image
        pil_img = Image.fromarray(img)

        # Get a drawing context
        draw = ImageDraw.Draw(pil_img)
        draw.text(pos, self.tick, (255, 255, 255), font=self.font)

        # Conver back to OpenCV image
        result = np.array(pil_img)

        return result
