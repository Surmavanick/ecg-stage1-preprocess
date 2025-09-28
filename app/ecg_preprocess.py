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
    trace_mask_from_gray,
    to_png_bytes,
)

# optional ECG-Image-Kit import
try:
    import ecg_image_kit as eik
    HAS_ECG_IMAGE_KIT = True
except ImportError:
    HAS_ECG_IMAGE_KIT = False

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

    # --- if ECG-Image-Kit is available, prefer it ---
    if HAS_ECG_IMAGE_KIT:
        try:
            trace_mask = eik.extract_trace(bgr)
            rectified_b64 = base64.b64encode(to_png_bytes(bgr)).decode("utf-8")
            trace_b64 = base64.b64encode(to_png_bytes(trace_mask)).decode("utf-8")
            return {
                "debug": {"rotation_deg": 0.0, "px_per_mm": 0.0, "grid_period_px": {"x": 0.0, "y": 0.0}, "blur_var": 0.0},
                "images": {"rectified_png_b64": rectified_b64},
                "masks": {"trace_png_b64": trace_b64, "grid_png_b64": ""},
                "download_urls": {},
            }
        except Exception as e:
            print(f"⚠️ ECG-Image-Kit failed, falling back to classical pipeline: {e}")

    # --- classical OpenCV pipeline ---
    gray0 = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    angle = detect_skew_angle_via_hough(gray0)
    rotated = rotate_image(bgr, angle)

    grid_mask = grid_mask_from_hsv(rotated, color=grid_color)
    rotated_gray = cv.cvtColor(rotated, cv.COLOR_BGR2GRAY)
    no_grid = inpaint_grid(rotated_gray, grid_mask)

    trace_mask = trace_mask_from_gray(no_grid)

    period_x, period_y = estimate_grid_period(grid_mask)
    valid_periods = [p for p in (period_x, period_y) if p and p > 0]
    px_per_mm = float(np.mean(valid_periods) if valid_periods else 20.0)
    blur_var = variance_of_laplacian(rotated_gray)

    rectified_file = os.path.join(OUTPUT_DIR, "rectified.png")
    grid_file = os.path.join(OUTPUT_DIR, "grid.png")
    trace_file = os.path.join(OUTPUT_DIR, "trace.png")
    try:
        cv.imwrite(rectified_file, rotated)
        cv.imwrite(grid_file, grid_mask)
        cv.imwrite(trace_file, trace_mask)
    except Exception:
        pass

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
        "download_urls": {
            "rectified": f"{BASE_URL}/download/rectified.png",
            "grid": f"{BASE_URL}/download/grid.png",
            "trace": f"{BASE_URL}/download/trace.png",
        },
    }
