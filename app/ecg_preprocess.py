# ecg_preprocess.py

import os
import base64
from typing import Dict, Any

import cv2 as cv
import numpy as np

# We need all these helper functions from utils.py
from .utils import (
    variance_of_laplacian,
    rotate_image,
    detect_skew_angle_via_hough,
    inpaint_grid,
    estimate_grid_period,
    to_png_bytes,
    trace_mask_from_gray,
    grid_mask_from_hsv, # Re-adding the color function
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
        empty_png = base64.b64encode(to_png_bytes(np.zeros((32, 32), np.uint8))).decode("utf-8")
        return {"debug": {}, "images": {}, "masks": {}, "download_urls": {}}

    # --- UNIVERSAL LOGIC FOR BOTH COLOR AND GRAYSCALE ---

    # 1. Deskew image (works on the color image)
    gray0 = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    angle = detect_skew_angle_via_hough(gray0)
    rotated_bgr = rotate_image(bgr, angle) # Keep the color version
    rotated_gray = cv.cvtColor(rotated_bgr, cv.COLOR_BGR2GRAY) # Grayscale for calculations

    # 2. Find the TRACE MASK from the grayscale image
    # This is reliable for finding the dark trace regardless of color
    trace_mask = trace_mask_from_gray(rotated_gray)

    # 3. Find the GRID MASK
    # We will try the color method first. If it fails (finds nothing), we'll try the grayscale method.
    grid_mask = grid_mask_from_hsv(rotated_bgr, color=grid_color)
    
    # Check if the color method found a reasonable grid. If not, fallback to grayscale method.
    if cv.countNonZero(grid_mask) < (grid_mask.shape[0] * grid_mask.shape[1] * 0.01): # If grid is less than 1% of image
        _, grid_and_trace_mask = cv.threshold(rotated_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        grid_mask = cv.subtract(grid_and_trace_mask, trace_mask)

    # 4. Refine the grid mask by removing any trace parts that might have been caught
    grid_mask = cv.subtract(grid_mask, trace_mask)
    kernel = np.ones((2,2), np.uint8)
    grid_mask = cv.morphologyEx(grid_mask, cv.MORPH_OPEN, kernel)

    # 5. --- THE MAIN FIX IS HERE ---
    # Inpaint the original COLOR image (`rotated_bgr`) instead of the grayscale one.
    final_image = cv.inpaint(rotated_bgr, grid_mask, 3, cv.INPAINT_TELEA)

    # --- END OF FIX ---

    # QC metrics
    period_x, period_y = estimate_grid_period(grid_mask)
    valid_periods = [p for p in (period_x, period_y) if p and p > 0]
    px_per_mm = float(np.mean(valid_periods) if valid_periods else 20.0)
    blur_var = variance_of_laplacian(rotated_gray)

    # Save the final, clean files
    rectified_file = os.path.join(OUTPUT_DIR, "rectified.png")
    grid_file = os.path.join(OUTPUT_DIR, "grid.png")
    trace_file = os.path.join(OUTPUT_DIR, "trace.png")
    try:
        cv.imwrite(rectified_file, final_image) # Save the final COLOR image
        cv.imwrite(grid_file, grid_mask)
        cv.imwrite(trace_file, trace_mask)
    except Exception:
        pass

    # Encode for JSON response
    rectified_b64 = base64.b64encode(to_png_bytes(final_image)).decode("utf-8")
    grid_b64 = base64.b64encode(to_png_bytes(grid_mask)).decode("utf-8")
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
