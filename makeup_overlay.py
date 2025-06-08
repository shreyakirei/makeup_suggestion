import cv2
import numpy as np

def apply_virtual_makeup(image, landmarks, tone, shape):
    h, w, _ = image.shape
    img = image.copy()

    def get_px(x, y): return int(x * w), int(y * h)

    # Lips (upper + lower midpoint)
    lips = [landmarks[61], landmarks[291]]
    for x, y in lips:
        px, py = get_px(x, y)
        cv2.circle(img, (px, py), 5, (150, 0, 150), -1)

    # Blush near cheeks
    cheeks = [landmarks[234], landmarks[454]]
    for x, y in cheeks:
        px, py = get_px(x, y)
        cv2.circle(img, (px, py), 15, (255, 182, 193), -1)

    # Eyeshadow (simple color)
    eyes = [landmarks[33], landmarks[263]]
    for x, y in eyes:
        px, py = get_px(x, y)
        cv2.circle(img, (px, py), 10, (173, 216, 230), -1)

    return img
