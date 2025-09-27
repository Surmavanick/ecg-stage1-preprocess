import cv2 as cv
import numpy as np
from typing import Tuple

# --- No changes in these functions ---
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

# --- CHANGES START HERE ---

def grid_mask_from_hsv(bgr: np.ndarray, color: str = "red") -> np.ndarray:
    """
    Creates a grid mask based on the specified color ('red', 'blue', 'green').
    """
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    
    # Color ranges dictionary to make the function flexible
    color_ranges = {
        'red': {
            'lower1': (0, 70, 70), 'upper1': (10, 255, 255),
            'lower2': (170, 70, 70), 'upper2': (180, 255, 255)
        },
        'blue': {
            'lower1': (100, 150, 0), 'upper1': (140, 255, 255),
            'lower2': None, 'upper2': None # Blue has one continuous range
        },
        'green': {
            'lower1': (36, 50, 70), 'upper1': (89, 255, 255),
            'lower2': None, 'upper2': None # Green also has one range
        }
    }

    selected_color = color_ranges.get(color.lower())
    if not selected_color:
        # Default to red if an invalid color is provided
        selected_color = color_ranges['red']

    # Create mask based on the selected color ranges
    mask1 = cv.inRange(hsv, selected_color['lower1'], selected_color['upper1'])
    if selected_color['lower2'] is not None:
        mask2 = cv.inRange(hsv, selected_color['lower2'], selected_color['upper2'])
        mask = cv.bitwise_or(mask1, mask2)
    else:
        mask = mask1
        
    kernel = np.ones((3,3), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=1)
    return mask

# --- CHANGES END HERE ---


# --- No changes in the remaining functions ---
def inpaint_grid(gray: np.ndarray, grid_mask: np.ndarray) -> np.ndarray:
    mask = (grid_mask > 0).astype(np.uint8) * 255
    return cv.inpaint(gray, mask, 3, cv.INPAINT_TELEA)

def estimate_grid_period(mask: np.ndarray) -> Tuple[float, float]:
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
        try:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(corr, distance=5)
            if len(peaks) < 1:
                return 0.0
            return float(peaks[0])
        except (ImportError, IndexError):
            return 0.0

    period_x = peak_period(proj_x)
    period_y = peak_period(proj_y)
    return period_x, period_y

def trace_mask_from_gray(no_grid_gray: np.ndarray) -> np.ndarray:
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
