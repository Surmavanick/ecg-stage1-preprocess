import cv2 as cv
import numpy as np
from typing import Tuple

# --- საბაზისო ---
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
    if lines is None:
        return 0.0
    angles = []
    for i in range(min(len(lines), 500)):
        rho, theta = lines[i][0]
        angles.append(theta)
    if not angles:
        return 0.0
    angles = np.rad2deg(np.array(angles))
    angles = np.where(angles > 90, angles - 180, angles)
    median = float(np.median(angles))
    return -median


# --- GRID (მორფოლოგიური სტრუქტურებზე) ---
def grid_mask_from_gray_lines(gray: np.ndarray) -> np.ndarray:
    """
    ECG-ის ბადის ამოცნობა მხოლოდ სტრუქტურაზე დაყრდნობით.
    """
    gray = cv.medianBlur(gray, 3)

    # Contrast enhancement (CLAHE)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)

    # ხაზების გაშავება: black-hat
    bh_h = cv.morphologyEx(g, cv.MORPH_BLACKHAT, cv.getStructuringElement(cv.MORPH_RECT, (41, 1)))
    bh_v = cv.morphologyEx(g, cv.MORPH_BLACKHAT, cv.getStructuringElement(cv.MORPH_RECT, (1, 41)))
    bh = cv.addWeighted(bh_h, 0.5, bh_v, 0.5, 0)

    # ბინარიზაცია Otsu
    _, th = cv.threshold(bh, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    h, w = th.shape[:2]

    # მსხვილი ხაზები
    k_h_major = cv.getStructuringElement(cv.MORPH_RECT, (max(25, w // 45), 1))
    k_v_major = cv.getStructuringElement(cv.MORPH_RECT, (1, max(25, h // 45)))
    horiz_major = cv.morphologyEx(th, cv.MORPH_OPEN, k_h_major)
    vert_major  = cv.morphologyEx(th, cv.MORPH_OPEN, k_v_major)

    # წვრილი ხაზები
    k_h_minor = cv.getStructuringElement(cv.MORPH_RECT, (max(9, w // 140), 1))
    k_v_minor = cv.getStructuringElement(cv.MORPH_RECT, (1, max(9, h // 140)))
    horiz_minor = cv.morphologyEx(th, cv.MORPH_OPEN, k_h_minor)
    vert_minor  = cv.morphologyEx(th, cv.MORPH_OPEN, k_v_minor)

    grid = cv.bitwise_or(cv.bitwise_or(horiz_major, vert_major),
                         cv.bitwise_or(horiz_minor, vert_minor))

    # წმენდა
    grid = cv.morphologyEx(grid, cv.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    grid = cv.morphologyEx(grid, cv.MORPH_OPEN,  np.ones((2,2), np.uint8), iterations=1)

    return grid


# --- INPAINT ---
def inpaint_grid(image: np.ndarray, grid_mask: np.ndarray) -> np.ndarray:
    """
    Removes the grid from COLOR or GRAYSCALE image using Telea inpainting.
    """
    mask = (grid_mask > 0).astype(np.uint8) * 255
    return cv.inpaint(image, mask, 3, cv.INPAINT_TELEA)


# --- QC: Grid period ---
def estimate_grid_period(mask: np.ndarray) -> Tuple[float, float]:
    if mask.ndim == 3:
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    mask = (mask > 0).astype(np.uint8) * 255
    proj_x = mask.mean(axis=0)
    proj_y = mask.mean(axis=1)

    def peak_period(sig):
        sig = sig - sig.mean()
        if np.all(sig == 0):
            return 0.0
        corr = np.correlate(sig, sig, mode='full')
        corr = corr[corr.size // 2:]
        try:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(corr, distance=5)
            if len(peaks) < 1:
                return 0.0
            return float(peaks[0])
        except Exception:
            return 0.0

    return peak_period(proj_x), peak_period(proj_y)


# --- TRACE ---
def trace_mask_from_gray(no_grid_gray: np.ndarray) -> np.ndarray:
    th = cv.adaptiveThreshold(no_grid_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv.THRESH_BINARY_INV, 35, 5)
    th = cv.morphologyEx(th, cv.MORPH_OPEN, np.ones((2,2), np.uint8), iterations=1)
    return th


# --- Helper ---
def ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def to_png_bytes(arr: np.ndarray) -> bytes:
    is_ok, buf = cv.imencode(".png", ensure_uint8(arr))
    return buf.tobytes() if is_ok else b""
