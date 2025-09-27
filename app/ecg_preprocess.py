import os
import base64
from typing import Dict, Any

import cv2 as cv
import numpy as np

from .utils import (
    variance_of_laplacian,
    rotate_image,
    detect_skew_angle_via_hough,
    grid_mask_from_gray_lines,   # მხოლოდ morphology grid
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
        empty_png = base64.b64encode(
            to_png_bytes(np.zeros((32, 32, 3), np.uint8))
        ).decode("utf-8")
        return {
            "debug": {"rotation_deg": 0.0, "px_per_mm": 0.0,
                      "grid_period_px": {"x": 0.0, "y": 0.0},
                      "blur_var": 0.0, "grid_coverage_pct": 0.0},
            "images": {"rectified_png_b64": empty_png},
            "masks": {"trace_png_b64": empty_png, "grid_png_b64": empty_png},
            "download_urls": {},
        }

    # --- Step 1: Deskew (COLOR ინარჩუნებს) ---
    gray0 = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    angle = detect_skew_angle_via_hough(gray0)
    rotated_bgr  = rotate_image(bgr, angle)
    rotated_gray = cv.cvtColor(rotated_bgr, cv.COLOR_BGR2GRAY)

    # --- Step 2: Trace (საწყისი) ---
    trace_mask_initial = trace_mask_from_gray(rotated_gray)

    # --- Step 3: GRID (მხოლოდ morphology) ---
    grid_mask = grid_mask_from_gray_lines(rotated_gray)

    # trace გამოვაკლოთ grid-ს
    grid_mask = cv.subtract(grid_mask, trace_mask_initial)
    grid_mask = cv.morphologyEx(grid_mask, cv.MORPH_OPEN,
                                np.ones((3,3), np.uint8), iterations=1)

    coverage = cv.countNonZero(grid_mask) / (grid_mask.size + 1e-9)

    # --- Step 4: Inpaint (ფერად rotated_bgr-ზე) ---
    rectified_color = inpaint_grid(rotated_bgr, grid_mask)

    # --- Step 5: Trace ხელახლა (უკვე grid-ის გარეშე) ---
    rectified_gray = cv.cvtColor(rectified_color, cv.COLOR_BGR2GRAY)
    trace_mask = trace_mask_from_gray(rectified_gray)

    # --- Step 6: QC ---
    period_x, period_y = estimate_grid_period(grid_mask)
    valid_periods = [p for p in (period_x, period_y) if p and p > 0]
    px_per_mm = float(np.mean(valid_periods) if valid_periods else 20.0)
    blur_var = variance_of_laplacian(rotated_gray)

    # --- Step 7: Save ---
    rectified_file = os.path.join(OUTPUT_DIR, "rectified.png")
    grid_file = os.path.join(OUTPUT_DIR, "grid.png")
    trace_file = os.path.join(OUTPUT_DIR, "trace.png")
    try:
        cv.imwrite(rectified_file, rectified_color)
        cv.imwrite(grid_file, grid_mask)
        cv.imwrite(trace_file, trace_mask)
    except Exception:
        pass

    # --- Step 8: Encode ---
    rectified_b64 = base64.b64encode(to_png_bytes(rectified_color)).decode("utf-8")
    grid_b64 = base64.b64encode(to_png_bytes(grid_mask)).decode("utf-8")
    trace_b64 = base64.b64encode(to_png_bytes(trace_mask)).decode("utf-8")

    return {
        "debug": {
            "rotation_deg": float(angle),
            "px_per_mm": float(px_per_mm),
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
