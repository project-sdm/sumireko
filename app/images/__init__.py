from cv2.typing import MatLike
import cv2

import shared.image


def image_search(sift: cv2.SIFT, img: MatLike):
    img = shared.image.downscale(img)
    _, descriptors = sift.detectAndCompute(img, None)
