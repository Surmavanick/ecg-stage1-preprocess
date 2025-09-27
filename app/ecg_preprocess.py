import os
import base64
from typing import Dict, Any

import cv2 as cv
import numpy as np

from .utils import (
    variance_of_laplacian,
    rotate_image,
    detect_skew_angle_via_hough,
    grid_mask_from_hsv,
    inpaint_grid,
    estimate_grid_period,
    to_png_bytes,
    trace_mask_from_gray,
)

OUTPUT_DIR = "/app/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_URL = os.getenv("BASE_URL", "https://your-app-name.onrender.com")


def _bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv.imdecode(arr, cv.IMREAD_COLOR)
    return bgr


def run_pipeline(image_bytes: bytes, grid_color: str = 'red', **kwargs) -> Dict[str, Any]:
    bgr = _bytes_to_bgr(image_bytes)
    if bgr is None:
        empty_png = base64.b64encode(to_png_bytes(np.zeros((32, 32, 3), np.uint8))).decode("utf-8")
        return {"debug": {}, "images": {}, "masks": {}, "download_urls": {}}

    # 1. Deskew
    gray0 = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    angle = detect_skew_angle_via_hough(gray0)
    rotated_bgr = rotate_image(bgr, angle)
    rotated_gray = cv.cvtColor(rotated_bgr, cv.COLOR_BGR2GRAY)

    # 2. Grid mask
    grid_mask = grid_mask_from_hsv(rotated_bgr, color=grid_color)
    kernel = np.ones((2,2), np.uint8)
    grid_mask = cv.morphologyEx(grid_mask, cv.MORPH_OPEN, kernel, iterations=1)
    grid_mask = cv.morphologyEx(grid_mask, cv.MORPH_CLOSE, kernel, iterations=1)

    # 3. Inpaint COLOR image (rectified ECG without grid)
    rectified_color = inpaint_grid(rotated_bgr, grid_mask)

    # 4. Trace mask (from rectified grayscale)
    rectified_gray = cv.cvtColor(rectified_color, cv.COLOR_BGR2GRAY)
    trace_mask = trace_mask_from_gray(rectified_gray)

    # QC
    period_x, period_y = estimate_grid_period(grid_mask)
    blur_var = variance_of_laplacian(rotated_gray)

    # Save
    rectified_file = os.path.join(OUTPUT_DIR, "rectified.png")
    grid_file = os.path.join(OUTPUT_DIR, "grid.png")
    trace_file = os.path.join(OUTPUT_DIR, "trace.png")
    try:
        cv.imwrite(rectified_file, rectified_color)
        cv.imwrite(grid_file, grid_mask)
        cv.imwrite(trace_file, trace_mask)
    except Exception:
        pass

    # Encode
    rectified_b64 = base64.b64encode(to_png_bytes(rectified_color)).decode("utf-8")
    grid_b64 = base64.b64encode(to_png_bytes(grid_mask)).decode("utf-8")
    trace_b64 = base64.b64encode(to_png_bytes(trace_mask)).decode("utf-8")

    return {
        "debug": {
            "rotation_deg": float(angle),
            "grid_period_px": {"x": float(period_x), "y": float(period_y)},
            "blur_var": float(blur_var),
        },
        "images": {"rectified_png_b64": rectified_b64},
        "masks": {"trace_png_b64": trace_b64, "grid_png_b64": grid_b64},
        "download_urls": {
            "rectified": f"{BASE_URL}/download/rectified.png",
            "grid": f"{BASE_URL}/download/grid.png",
            "trace": f"{BASE_URL}/download/trace.png",
        },
    }
