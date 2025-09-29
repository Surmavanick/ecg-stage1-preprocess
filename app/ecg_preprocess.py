import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def image_to_sequence(img_path, mode="dark-foreground", method="moving_average", windowlen=5, plot_result=False):
    """
    Extract ECG signal from an image (Python version).
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # If bright foreground → invert
    if mode == "bright-foreground":
        gray = cv2.bitwise_not(gray)

    trace = []
    for col in range(w):
        column = gray[:, col]
        row = np.argmin(column)  # pick darkest pixel
        trace.append(row)

    trace = np.array(trace, dtype=float)

    # Normalize
    trace = (trace - np.min(trace)) / (np.max(trace) - np.min(trace))

    # Apply smoothing
    if method == "moving_average":
        kernel = np.ones(windowlen) / windowlen
        trace = np.convolve(trace, kernel, mode="same")
    elif method == "max_finder":
        pass  # already implemented by argmin
    # Other methods can be extended here

    if plot_result:
        plt.figure(figsize=(10, 3))
        plt.plot(trace, color="g")
        plt.title("Extracted ECG Signal")
        plt.xlabel("Samples")
        plt.ylabel("Normalized Amplitude")
        plt.show()

    return trace
