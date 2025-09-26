
import base64, io
import cv2 as cv
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple
from .utils import (
    variance_of_laplacian, rotate_image, detect_skew_angle_via_hough,
    grid_mask_from_red_hsv, inpaint_grid, estimate_grid_period,
    trace_mask_from_gray, to_png_bytes
)

def preprocess_ecg_photo(bgr: np.ndarray, speed_hint: int | None = None, gain_hint: int | None = None) -> Dict[str, Any]:
    # 1) Basic sizes
    h, w = bgr.shape[:2]

    # 2) Deskew (rotation)
    gray0 = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    angle = detect_skew_angle_via_hough(gray0)
    rotated = rotate_image(bgr, angle)

    # 3) Grid mask + inpaint
    grid_mask = grid_mask_from_red_hsv(rotated)
    rotated_gray = cv.cvtColor(rotated, cv.COLOR_BGR2GRAY)
    no_grid = inpaint_grid(rotated_gray, grid_mask)

    # 4) Trace mask
    trace_mask = trace_mask_from_gray(no_grid)

    # 5) Grid period estimate (pixels)
    period_x, period_y = estimate_grid_period(grid_mask)
    # crude px_per_mm; if period==0 fallback to heuristic 20 px
    px_per_mm = float(np.mean([p for p in [period_x, period_y] if p and p > 0]) or 20.0)

    # 6) Quality metrics
    blur_var = variance_of_laplacian(rotated_gray)

    # Write debug images to ./output
    cv.imwrite("/app/output/1_rotated.png", rotated)
    cv.imwrite("/app/output/2_grid_mask.png", grid_mask)
    cv.imwrite("/app/output/3_no_grid.png", no_grid)
    cv.imwrite("/app/output/4_trace_mask.png", trace_mask)

    # Encode to base64 for API response
    rectified_b64 = base64.b64encode(to_png_bytes(rotated)).decode("utf-8")
    grid_b64 = base64.b64encode(to_png_bytes(grid_mask)).decode("utf-8")
    trace_b64 = base64.b64encode(to_png_bytes(trace_mask)).decode("utf-8")

    return {
        "ok": True,
        "debug": {
            "rotation_deg": float(angle),
            "px_per_mm": float(px_per_mm),
            "grid_period_px": {"x": float(period_x), "y": float(period_y)},
            "blur_var": float(blur_var),
        },
        "masks": {
            "trace_png_b64": trace_b64,
            "grid_png_b64": grid_b64
        },
        "images": {
            "rectified_png_b64": rectified_b64
        }
    }
