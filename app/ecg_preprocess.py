import cv2
import numpy as np

def image_to_sequence(img_path, mode="dark-foreground", method="moving_average", window=5):
    """
    Extract ECG signal from an image.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Invert if needed
    if mode == "bright-foreground":
        gray = cv2.bitwise_not(gray)

    trace = []
    for col in range(w):
        column = gray[:, col]
        row = np.argmin(column)  # pick darkest pixel
        trace.append(row)

    trace = np.array(trace, dtype=float)

    # Moving average filter
    if method == "moving_average":
        kernel = np.ones(window) / window
        trace = np.convolve(trace, kernel, mode="same")

    return trace
