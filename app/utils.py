
import cv2 as cv
import numpy as np
from typing import Tuple, Dict

def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv.Laplacian(gray, cv.CV_64F).var())

def rotate_image(image, angle_deg):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv.warpAffine(image, M, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    return rotated

def detect_skew_angle_via_hough(gray: np.ndarray) -> float:
    # Edge detection
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    lines = cv.HoughLines(edges, 1, np.pi/180, 200)
    if lines is None:
        return 0.0
    angles = []
    for i in range(min(len(lines), 500)):
        rho, theta = lines[i][0]
        angles.append(theta)
    if not angles:
        return 0.0
    angles = np.rad2deg(np.array(angles))
    # Map near-vertical lines to deviation from 90deg
    # We take the mode-ish central tendency
    angles = np.where(angles > 90, angles - 180, angles)
    median = float(np.median(angles))
    # Rotate opposite to bring to ~0
    return -median

def grid_mask_from_red_hsv(bgr: np.ndarray) -> np.ndarray:
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    m1 = cv.inRange(hsv, (0, 70, 70), (10, 255, 255))
    m2 = cv.inRange(hsv, (170, 70, 70), (180, 255, 255))
    mask = cv.bitwise_or(m1, m2)
    kernel = np.ones((3,3), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=1)
    return mask

def inpaint_grid(gray: np.ndarray, grid_mask: np.ndarray) -> np.ndarray:
    # Inpaint expects 8-bit 1-channel mask
    mask = (grid_mask > 0).astype(np.uint8) * 255
    return cv.inpaint(gray, mask, 3, cv.INPAINT_TELEA)

def estimate_grid_period(mask: np.ndarray) -> Tuple[float, float]:
    # Return (period_x, period_y) in pixels using autocorrelation peaks.
    if mask.ndim == 3:
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    mask = (mask > 0).astype(np.uint8)*255
    proj_x = mask.mean(axis=0)
    proj_y = mask.mean(axis=1)

    def peak_period(sig):
        sig = sig - sig.mean()
        if np.all(sig == 0):
            return 0.0
        corr = np.correlate(sig, sig, mode='full')
        corr = corr[corr.size//2:]
        # Find first peak after lag=0 by simple argrelmax heuristic
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(corr, distance=5)
        if len(peaks) < 2:
            return 0.0
        # Take smallest non-zero lag peak prominence
        return float(peaks[0])
    try:
        from scipy.signal import find_peaks  # ensure import
    except Exception:
        pass
    # Fallback without scipy
    try:
        period_x = peak_period(proj_x)
        period_y = peak_period(proj_y)
    except Exception:
        period_x = 0.0
        period_y = 0.0
    return period_x, period_y

def trace_mask_from_gray(no_grid_gray: np.ndarray) -> np.ndarray:
    # Baseline: adaptive threshold + thin clean-up
    th = cv.adaptiveThreshold(no_grid_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv.THRESH_BINARY_INV, 35, 5)
    kernel = np.ones((2,2), np.uint8)
    th = cv.morphologyEx(th, cv.MORPH_OPEN, kernel, iterations=1)
    return th

def ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def to_png_bytes(arr: np.ndarray) -> bytes:
    is_ok, buf = cv.imencode(".png", ensure_uint8(arr))
    return buf.tobytes() if is_ok else b""
