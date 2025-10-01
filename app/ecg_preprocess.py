import cv2
import numpy as np
from scipy.signal import savgol_filter
from scipy import ndimage

def remove_text_and_artifacts(gray_img, min_area=50, max_area=None):
    """
    Remove text and small artifacts from ECG image while preserving the main trace.
    """
    # Create binary image
    _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invert so ECG trace is white
    binary = cv2.bitwise_not(binary)
    
    # Remove small components (text, noise)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Create mask to keep only large components (ECG trace)
    mask = np.zeros_like(binary)
    h, w = binary.shape
    
    if max_area is None:
        max_area = h * w * 0.1  # Max 10% of image area
    
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            # Check if component is likely ECG trace (elongated, not compact like text)
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            aspect_ratio = max(width, height) / min(width, height)
            
            # Keep components that are elongated (likely ECG trace parts)
            if aspect_ratio > 3 or area > min_area * 5:
                mask[labels == i] = 255
    
    return mask

def extract_main_trace_contour(binary_mask):
    """
    Extract the main ECG trace using contour detection.
    """
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find the longest contour (main ECG trace)
    main_contour = max(contours, key=cv2.contourArea)
    
    return main_contour

def contour_to_trace(contour, img_width, img_height):
    """
    Convert contour points to a trace array.
    """
    if contour is None:
        return np.zeros(img_width)
    
    # Create trace array
    trace = np.full(img_width, img_height // 2, dtype=float)  # Default to middle
    
    # Extract points from contour
    points = contour.reshape(-1, 2)
    
    # Sort points by x-coordinate
    points = points[points[:, 0].argsort()]
    
    # For each x-coordinate, find the corresponding y-coordinate
    for x, y in points:
        if 0 <= x < img_width:
            trace[x] = y
    
    # Interpolate missing values
    valid_indices = np.where(trace != img_height // 2)[0]
    if len(valid_indices) > 1:
        # Linear interpolation for missing values
        all_indices = np.arange(img_width)
        trace = np.interp(all_indices, valid_indices, trace[valid_indices])
    
    return trace

def morphological_trace_extraction(gray_img):
    """
    Extract ECG trace using morphological operations to remove text.
    """
    # Apply morphological opening to remove thin structures (text)
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    
    # Remove horizontal lines (grid) but keep ECG trace
    temp = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel_horizontal)
    
    # Remove small vertical artifacts
    temp = cv2.morphologyEx(temp, cv2.MORPH_OPEN, kernel_vertical)
    
    # Apply Gaussian blur to smooth
    temp = cv2.GaussianBlur(temp, (3, 3), 0)
    
    return temp

def image_to_sequence(img_path, mode="dark-foreground", method="contour", window=5, remove_artifacts=True):
    """
    Extract ECG signal from an image with improved text removal.
    
    Parameters:
    - img_path: path to ECG image
    - mode: "dark-foreground" or "bright-foreground"
    - method: "contour", "morphological", or "simple" (original method)
    - window: smoothing window size
    - remove_artifacts: whether to remove text and artifacts
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Invert if needed
    if mode == "bright-foreground":
        gray = cv2.bitwise_not(gray)

    if method == "simple" or not remove_artifacts:
        # Original simple method
        trace = []
        for col in range(w):
            column = gray[:, col]
            row = np.argmin(column)  # pick darkest pixel
            trace.append(row)
        trace = np.array(trace, dtype=float)
        
    elif method == "contour":
        # Improved contour-based method
        if remove_artifacts:
            # Remove text and artifacts
            clean_mask = remove_text_and_artifacts(gray, min_area=100)
            main_contour = extract_main_trace_contour(clean_mask)
            trace = contour_to_trace(main_contour, w, h)
        else:
            # Simple threshold and contour
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary = cv2.bitwise_not(binary)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                trace = contour_to_trace(main_contour, w, h)
            else:
                trace = np.zeros(w)
                
    elif method == "morphological":
        # Morphological approach
        processed = morphological_trace_extraction(gray)
        trace = []
        for col in range(w):
            column = processed[:, col]
            # Find the most prominent dark pixel (but avoid isolated pixels)
            if np.min(column) < np.mean(column) - np.std(column):
                row = np.argmin(column)
            else:
                row = h // 2  # Default to middle if no clear signal
            trace.append(row)
        trace = np.array(trace, dtype=float)
    
    else:
        raise ValueError(f"Unknown method: {method}")

    # Apply smoothing
    if method != "simple" and len(trace) > window:
        if window > 1:
            # Apply Savitzky-Golay filter for better smoothing
            if len(trace) > window * 2:
                trace = savgol_filter(trace, min(window * 2 + 1, len(trace) // 4), 2)
            else:
                # Fallback to moving average
                kernel = np.ones(window) / window
                trace = np.convolve(trace, kernel, mode="same")

    return trace

def preprocess_ecg_image(img_path, output_debug=False):
    """
    Complete ECG preprocessing pipeline with debug output.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Remove artifacts
    clean_mask = remove_text_and_artifacts(gray)
    
    # Step 2: Extract trace
    trace = image_to_sequence(img_path, method="contour", remove_artifacts=True)
    
    debug_info = {}
    if output_debug:
        debug_info = {
            "original_shape": gray.shape,
            "trace_length": len(trace),
            "trace_range": (float(np.min(trace)), float(np.max(trace))),
            "clean_mask_nonzero": int(np.count_nonzero(clean_mask))
        }
    
    return trace, debug_info
