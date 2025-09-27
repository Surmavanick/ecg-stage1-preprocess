# ecg_preprocess.py

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
    to_png_bytes,  # Removed trace_mask_from_gray as we use a better method now
)

OUTPUT_DIR = "/app/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_URL = os.getenv("BASE_URL", "https://your-app-name.onrender.com")


def _bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv.imdecode(arr, cv.IMREAD_COLOR)
    return bgr


def run_pipeline(image_bytes: bytes, grid_color: str = "red") -> Dict[str, Any]:
    bgr = _bytes_to_bgr(image_bytes)
    if bgr is None:
        empty_png = base64.b64encode(to_png_bytes(np.zeros((32, 32), np.uint8))).decode("utf-8")
        return {
            "debug": {"rotation_deg": 0.0, "px_per_mm": 0.0, "grid_period_px": {"x": 0.0, "y": 0.0}, "blur_var": 0.0},
            "images": {"rectified_png_b64": empty_png},
            "masks": {"trace_png_b64": empty_png, "grid_png_b64": empty_png},
            "download_urls": {},
        }

    # --- NEW, MORE ROBUST LOGIC ---

    # 1. Deskew the image first
    gray0 = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    angle = detect_skew_angle_via_hough(gray0)
    rotated_bgr = rotate_image(bgr, angle)
    rotated_gray = cv.cvtColor(rotated_bgr, cv.COLOR_BGR2GRAY)

    # 2. Isolate the dark trace reliably using Otsu's thresholding
    # This gives us a clean mask of the trace itself.
    _, trace_mask = cv.threshold(rotated_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    
    # 3. Get the initial grid mask using the color method (as before)
    initial_grid_mask = grid_mask_from_hsv(rotated_bgr, color=grid_color)

    # 4. Refine the grid mask by REMOVING the trace from it
    # This solves the problem of the trace being detected as part of the grid.
    refined_grid_mask = cv.subtract(initial_grid_mask, trace_mask)
    # Optional: clean up small noise from the refined mask
    kernel = np.ones((2,2), np.uint8)
    refined_grid_mask = cv.morphologyEx(refined_grid_mask, cv.MORPH_OPEN, kernel)

    # 5. Inpaint the grayscale image using the REFINED grid mask
    no_grid_image = inpaint_grid(rotated_gray, refined_grid_mask)

    # The final `trace_mask` is the one we generated in step 2, which is already clean.
    # The final `grid_mask` is the `refined_grid_mask` from step 4.

    # --- END OF NEW LOGIC ---

    # QC metrics
    period_x, period_y = estimate_grid_period(refined_grid_mask)
    valid_periods = [p for p in (period_x, period_y) if p and p > 0]
    px_per_mm = float(np.mean(valid_periods) if valid_periods else 20.0)
    blur_var = variance_of_laplacian(rotated_gray)

    # Save files
    rectified_file = os.path.join(OUTPUT_DIR, "rectified.png")
    grid_file = os.path.join(OUTPUT_DIR, "grid.png")
    trace_file = os.path.join(OUTPUT_DIR, "trace.png")
    try:
        # Save the inpainted image, which should now look much better
        cv.imwrite(rectified_file, no_grid_image) 
        cv.imwrite(grid_file, refined_grid_mask)
        cv.imwrite(trace_file, trace_mask)
    except Exception:
        pass

    # Encode base64
    rectified_b64 = base64.b64encode(to_png_bytes(no_grid_image)).decode("utf-8")
    grid_b64 = base64.b64encode(to_png_bytes(refined_grid_mask)).decode("utf-8")
    trace_b64 = base64.b64encode(to_png_bytes(trace_mask)).decode("utf-8")

    return {
        "debug": {
            "rotation_deg": float(angle),
            "px_per_mm": float(px_per_mm),
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
