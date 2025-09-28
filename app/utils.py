# utils.py

import cv2 as cv
import numpy as np
from typing import Tuple

# --- საბაზისო ფუნქციები ---
def rotate_image(image, angle_deg):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv.warpAffine(image, M, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)

# --- INPAINT ---
def inpaint_grid(image: np.ndarray, grid_mask: np.ndarray) -> np.ndarray:
    """
    Removes the grid from a COLOR or GRAYSCALE image using a mask.
    """
    mask = (grid_mask > 0).astype(np.uint8) * 255
    return cv.inpaint(image, mask, 3, cv.INPAINT_TELEA)

# --- Helper ---
def to_png_bytes(arr: np.ndarray) -> bytes:
    """
    Encodes a numpy array to PNG bytes.
    """
    if arr.dtype != np.uint8:
        arr = arr.clip(0, 255).astype(np.uint8)
    is_ok, buf = cv.imencode(".png", arr)
    return buf.tobytes() if is_ok else b""
