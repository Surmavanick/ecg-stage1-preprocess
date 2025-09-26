# app/ecg_preprocess.py
import os
import base64
from typing import Dict, Any

import cv2 as cv
import numpy as np

from .utils import (
    variance_of_laplacian,
    rotate_image,
    detect_skew_angle_via_hough,
    grid_mask_from_red_hsv,
    inpaint_grid,
    estimate_grid_period,
    trace_mask_from_gray,
    to_png_bytes,
)

OUTPUT_DIR = "/app/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    """Decode uploaded bytes into BGR image (OpenCV)."""
    arr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv.imdecode(arr, cv.IMREAD_COLOR)
    return bgr


def run_pipeline(image_bytes: bytes, speed_hint: int | None = None, gain_hint: int | None = None) -> Dict[str, Any]:
    """
    Stage 1 pipeline:
      - deskew/dewarp (rotation only here)
      - grid removal (inpaint red grid)
      - trace mask (binary)
      - px/mm estimate
      - debug images + base64 outputs
    Returns dict with keys: debug, images.rectified_png_b64, masks.trace_png_b64, masks.grid_png_b64
    """
    # --- 0) Read/validate ---
    bgr = _bytes_to_bgr(image_bytes)
    if bgr is None:
        # უსაფრთხოდ დავაბრუნოთ ცარიელი მაგრამ ვალიდური სტრუქტურა
        empty_png = base64.b64encode(to_png_bytes(np.zeros((32, 32), np.uint8))).decode("utf-8")
        return {
            "debug": {"rotation_deg": 0.0, "px_per_mm": 0.0, "grid_period_px": {"x": 0.0, "y": 0.0}, "blur_var": 0.0},
            "images": {"rectified_png_b64": empty_png},
            "masks": {"trace_png_b64": empty_png, "grid_png_b64": empty_png},
        }

    # --- 1) Deskew (rotation by Hough) ---
    gray0 = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    angle = detect_skew_angle_via_hough(gray0)
    rotated = rotate_image(bgr, angle)

    # --- 2) Grid mask + removal ---
    grid_mask = grid_mask_from_red_hsv(rotated)            # binary mask of red grid (may be empty on gray grids)
    rotated_gray = cv.cvtColor(rotated, cv.COLOR_BGR2GRAY)
    no_grid = inpaint_grid(rotated_gray, grid_mask)        # inpaint only where grid mask>0

    # --- 3) Trace mask (binary) ---
    trace_mask = trace_mask_from_gray(no_grid)

    # --- 4) Grid period → px/mm estimate ---
    period_x, period_y = estimate_grid_period(grid_mask)   # 0.0 if not detectable
    valid_periods = [p for p in (period_x, period_y) if p and p > 0]
    px_per_mm = float(np.mean(valid_periods) if valid_periods else 20.0)  # sane fallback

    # --- 5) QC metrics ---
    blur_var = variance_of_laplacian(rotated_gray)

    # --- 6) Write debug images to disk ---
    try:
        cv.imwrite(os.path.join(OUTPUT_DIR, "1_rotated.png"), rotated)
        cv.imwrite(os.path.join(OUTPUT_DIR, "2_grid_mask.png"), grid_mask)
        cv.imwrite(os.path.join(OUTPUT_DIR, "3_no_grid.png"), no_grid)
        cv.imwrite(os.path.join(OUTPUT_DIR, "4_trace_mask.png"), trace_mask)
    except Exception:
        # თუ რაიმე დისკზე ჩაწერა ვერ გამოვიდა, API მაინც იმუშავებს
        pass

    # --- 7) Encode to base64 for API ---
    rectified_b64 = base64.b64encode(to_png_bytes(rotated)).decode("utf-8")
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
    }
