import numpy as np
from scipy.signal import savgol_filter

def smooth_signal(trace, method="savgol"):
    if method == "savgol":
        return savgol_filter(trace, 11, 3)
    elif method == "moving_average":
        kernel = np.ones(5) / 5
        return np.convolve(trace, kernel, mode="same")
    return trace
