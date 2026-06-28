import cv2
from cv2.typing import MatLike


def downscale(img: MatLike, max_side=512) -> MatLike:
    h, w = img.shape[:2]

    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    return img
