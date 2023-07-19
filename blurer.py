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
        # self.font = "./Arial Unicode.ttf"
        self.font = "/usr/share/fonts/truetype/freefont/FreeMono.ttf"
        self.tick = str(emoji.emojize(':grinning_face_with_big_eyes:'))

    def blur(self, img, face):
        # Make into PIL image
        pil_img = Image.fromarray(img)

        # Get a drawing context
        draw = ImageDraw.Draw(pil_img)

        # Draw emoji on face
        pos1 = (face.left(), face.top())
        size1 = face.right() - face.left()
        font1 = ImageFont.truetype(self.font, size1)
        draw.text(pos1, self.tick, (255, 255, 255), font=font1)

        # Conver back to OpenCV image
        result = np.array(pil_img)

        return result
