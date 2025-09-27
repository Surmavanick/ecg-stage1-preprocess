# utils.py

import cv2 as cv
import numpy as np
from typing import Tuple

# --- საბაზისო ფუნქციები (უცვლელია) ---
def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv.Laplacian(gray, cv.CV_64F).var())

def rotate_image(image, angle_deg):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv.warpAffine(image, M, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    return rotated

def detect_skew_angle_via_hough(gray: np.ndarray) -> float:
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    lines = cv.HoughLines(edges, 1, np.pi/180, 200)
    if lines is None: return 0.0
    angles = [theta for rho, theta in lines[:, 0]]
    if not angles: return 0.0
    angles = np.rad2deg(np.array(angles))
    angles = np.where(angles > 90, angles - 180, angles)
    return -float(np.median(angles))

# --- ფერის ნიღბის ფუნქცია (ძველი, მაგრამ ისევ გვჭირდება) ---
def grid_mask_from_hsv(bgr: np.ndarray, color: str = "red") -> np.ndarray:
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    color_ranges = {
        'red': {'lower1': (0, 70, 70), 'upper1': (10, 255, 255), 'lower2': (170, 70, 70), 'upper2': (180, 255, 255)},
        'blue': {'lower1': (100, 150, 0), 'upper1': (140, 255, 255), 'lower2': None, 'upper2': None},
        'green': {'lower1': (36, 50, 70), 'upper1': (89, 255, 255), 'lower2': None, 'upper2': None}
    }
    selected_color = color_ranges.get(color.lower(), color_ranges['red'])
    mask1 = cv.inRange(hsv, selected_color['lower1'], selected_color['upper1'])
    if selected_color['lower2'] is not None:
        mask2 = cv.inRange(hsv, selected_color['lower2'], selected_color['upper2'])
        return cv.bitwise_or(mask1, mask2)
    return mask1

# --- !!! ახალი ფუნქცია ნიღბის გასაწმენდად !!! ---
def refine_grid_mask_morphologically(mask: np.ndarray) -> np.ndarray:
    """
    Takes a "dirty" mask and keeps only the long straight lines (the grid).
    """
    h, w = mask.shape[:2]
    # ვერტიკალური ხაზების პოვნა
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, h // 30))
    vertical_lines = cv.morphologyEx(mask, cv.MORPH_OPEN, vertical_kernel, iterations=2)
    
    # ჰორიზონტალური ხაზების პოვნა
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (w // 30, 1))
    horizontal_lines = cv.morphologyEx(mask, cv.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # შედეგების გაერთიანება
    refined_mask = cv.add(vertical_lines, horizontal_lines)
    
    # საბოლოო გაწმენდა
    refined_mask = cv.morphologyEx(refined_mask, cv.MORPH_CLOSE, np.ones((5,5), np.uint8))
    return refined_mask

# --- დანარჩენი ფუნქციები (უცვლელია) ---
def inpaint_grid(image: np.ndarray, grid_mask: np.ndarray) -> np.ndarray:
    mask = (grid_mask > 0).astype(np.uint8) * 255
    return cv.inpaint(image, mask, 3, cv.INPAINT_TELEA)

def estimate_grid_period(mask: np.ndarray) -> Tuple[float, float]:
    proj_x = mask.mean(axis=0)
    proj_y = mask.mean(axis=1)
    def peak_period(sig):
        sig = sig - sig.mean()
        if np.all(sig == 0): return 0.0
        corr = np.correlate(sig, sig, mode='full')[len(sig)-1:]
        try:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(corr, distance=5)
            return float(peaks[0]) if len(peaks) > 0 else 0.0
        except Exception:
            return 0.0
    return peak_period(proj_x), peak_period(proj_y)

def trace_mask_from_gray(no_grid_gray: np.ndarray) -> np.ndarray:
    th = cv.adaptiveThreshold(no_grid_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 35, 5)
    th = cv.morphologyEx(th, cv.MORPH_OPEN, np.ones((2,2), np.uint8), iterations=1)
    return th

def to_png_bytes(arr: np.ndarray) -> bytes:
    is_ok, buf = cv.imencode(".png", arr.clip(0, 255).astype(np.uint8))
    return buf.tobytes() if is_ok else b""
