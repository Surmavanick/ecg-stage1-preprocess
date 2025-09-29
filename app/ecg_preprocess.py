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

    # Moving average smoothing
    if method == "moving_average":
        kernel = np.ones(windowlen) / windowlen
        trace = np.convolve(trace, kernel, mode="same")

    # Normalize
    trace_norm = (trace - trace.min()) / (trace.max() - trace.min())

    if plot_result:
        plt.figure(figsize=(12, 4))
        plt.plot(trace_norm, 'g')
        plt.title("Extracted ECG Trace (Python)")
        plt.show()

    return trace_norm

def run_pipeline(image_path: str, save_csv: str = None, plot: bool = False):
    """
    Main ECG preprocessing pipeline using classical image-based digitization.
    """
    signal = image_to_sequence(image_path, mode="dark-foreground",
                               method="moving_average", windowlen=5, plot_result=plot)

    # detect peaks (QRS)
    peaks, _ = find_peaks(signal, prominence=0.1)

    if save_csv:
        np.savetxt(save_csv, signal, delimiter=",")

    return {
        "signal": signal.tolist(),
        "num_samples": len(signal),
        "num_peaks": int(len(peaks)),
        "peaks": peaks.tolist()
    }
