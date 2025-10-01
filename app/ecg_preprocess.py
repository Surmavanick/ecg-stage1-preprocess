import cv2
import numpy as np


def image_to_sequence(
    img_path,
    mode="dark-foreground",
    method="moving_average",
    window=5,
    search_radius=25,
    roi_margin=0.05,
):
    """Extract an ECG signal trace from a raster image.

    The routine denoises the image, restricts the search to the central band to
    avoid textual annotations and follows the waveform column by column with a
    continuity constraint so that isolated dark pixels (e.g. text) are ignored.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Invert if needed
    if mode == "bright-foreground":
        gray = cv2.bitwise_not(gray)

    # Light denoising helps suppress grid lines/text
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Focus on the central band of the ECG to avoid textual annotations
    margin = int(h * roi_margin)
    top = max(margin, 0)
    bottom = h - max(margin, 0)
    if bottom <= top:
        top = 0
        bottom = h

    roi = gray[top:bottom, :]

    trace = []
    if roi.size == 0:
        return np.zeros(w, dtype=float)

    if search_radius is not None:
        search_radius = int(max(search_radius, 0))

    prev_row = int(np.argmin(np.mean(roi, axis=1)))

    for col in range(w):
        column = roi[:, col]
        if col == 0 or search_radius is None:
            rel_row = int(np.argmin(column))
        else:
            start = max(prev_row - search_radius, 0)
            end = min(prev_row + search_radius + 1, roi.shape[0])
            segment = column[start:end]
            if segment.size == 0:
                rel_row = int(np.argmin(column))
            else:
                rel_row = int(start + np.argmin(segment))
        prev_row = rel_row
        trace.append(rel_row + top)

    trace = np.array(trace, dtype=float)

    # Moving average filter
    if method == "moving_average":
        kernel = np.ones(window) / window
        trace = np.convolve(trace, kernel, mode="same")

    return trace
