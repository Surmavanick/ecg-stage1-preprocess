import numpy as np
from scipy.signal import find_peaks


def detect_qrs(signal, prominence=0.2):
    """
    Detect QRS complexes in ECG signal.
    """
    peaks, _ = find_peaks(signal, prominence=prominence)
    return peaks


def normalize_signal(signal):
    """
    Normalize ECG to [0, 1].
    """
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
