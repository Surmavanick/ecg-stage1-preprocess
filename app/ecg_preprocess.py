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
    grid_mask_from_gray_lines,
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


def run_pipeline(image_bytes: bytes, **kwargs) -> Dict[str, Any]:
    bgr = _bytes_to_bgr(image_bytes)
    if bgr is None:
        # Return empty response if image loading fails
        empty_png = base64.b64encode(to_png_bytes(np.zeros((32, 32), np.uint8))).decode("utf-8")
        return {"debug": {}, "images": {"rectified_png_b64": empty_png}, "masks": {}, "download_urls": {}}

    # --- Step 1: Deskew ---
    gray0 = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    angle = detect_skew_angle_via_hough(gray0)
    rotated_bgr = rotate_image(bgr, angle)

    # --- Step 2: Create a high-contrast image for grid detection ---
    # THE MAIN FIX: We split the BGR image into 3 channels (Blue, Green, Red)
    # The blue channel makes the blue/dark trace almost white and the red grid very dark.
    # This creates a perfect high-contrast image for our grid detection logic.
    blue_channel, _, _ = cv.split(rotated_bgr)
    
    # We invert the blue channel so the grid lines become white (foreground) for the mask functions.
    high_contrast_gray = cv.bitwise_not(blue_channel)


    # --- Step 3: Find the Grid using the morphological function on the high-contrast image ---
    # Now, grid_mask_from_gray_lines will work reliably.
    grid_mask = grid_mask_from_gray_lines(high_contrast_gray)


    # --- Step 4: Find the initial trace mask from the original grayscale image ---
    rotated_gray = cv.cvtColor(rotated_bgr, cv.COLOR_BGR2GRAY)
    trace_mask_initial = trace_mask_from_gray(rotated_gray)
    
    # Refine grid mask by removing any parts of the trace
    grid_mask = cv.subtract(grid_mask, trace_mask_initial)


    # --- Step 5: Inpaint the ORIGINAL COLOR image ---
    rectified_color = inpaint_grid(rotated_bgr, grid_mask)

    # --- Step 6: Find the final, clean trace mask from the rectified image ---
    rectified_gray = cv.cvtColor(rectified_color, cv.COLOR_BGR2GRAY)
    trace_mask_final = trace_mask_from_gray(rectified_gray)

    # --- QC & Save ---
    period_x, period_y = estimate_grid_period(grid_mask)
    blur_var = variance_of_laplacian(gray0)
    coverage = cv.countNonZero(grid_mask) / (grid_mask.size + 1e-9)

    rectified_file = os.path.join(OUTPUT_DIR, "rectified.png")
    grid_file = os.path.join(OUTPUT_DIR, "grid.png")
    trace_file = os.path.join(OUTPUT_DIR, "trace.png")
    try:
        cv.imwrite(rectified_file, rectified_color)
        cv.imwrite(grid_file, grid_mask)
        cv.imwrite(trace_file, trace_mask_final)
    except Exception as e:
        print(f"Error saving files: {e}")

    # --- Encode & Return ---
    rectified_b64 = base64.b64encode(to_png_bytes(rectified_color)).decode("utf-8")
    grid_b64 = base64.b64encode(to_png_bytes(grid_mask)).decode("utf-8")
    trace_b64 = base64.b64encode(to_png_bytes(trace_mask_final)).decode("utf-8")

    return {
        "debug": {
            "rotation_deg": float(angle),
            "px_per_mm": float(np.mean([p for p in (period_x, period_y) if p and p > 0]) if any(p and p > 0 for p in (period_x, period_y)) else 20.0),
            "grid_period_px": {"x": float(period_x), "y": float(period_y)},
            "blur_var": float(blur_var),
            "grid_coverage_pct": float(coverage),
        },
        "images": {"rectified_png_b64": rectified_b64},
        "masks": {"trace_png_b64": trace_b64, "grid_png_b64": grid_b64},
        "download_urls": {
            "rectified": f"{BASE_URL}/download/rectified.png",
            "grid": f"{BASE_URL}/download/grid.png",
            "trace": f"{BASE_URL}/download/trace.png",
        },
    }
