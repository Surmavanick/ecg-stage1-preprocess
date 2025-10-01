import numpy as np
from scipy.signal import savgol_filter, medfilt, butter, filtfilt
from scipy import ndimage

def smooth_signal(trace, method="savgol", **kwargs):
    """
    Apply various smoothing methods to the ECG trace.
    
    Parameters:
    - trace: input signal array
    - method: smoothing method ("savgol", "moving_average", "median", "gaussian", "butterworth")
    - kwargs: additional parameters for specific methods
    """
    if len(trace) < 3:
        return trace
        
    if method == "savgol":
        window_length = kwargs.get("window_length", min(11, len(trace) // 4))
        if window_length % 2 == 0:
            window_length += 1  # Must be odd
        window_length = max(3, window_length)  # Minimum window size
        polyorder = kwargs.get("polyorder", min(3, window_length - 1))
        
        if len(trace) > window_length:
            return savgol_filter(trace, window_length, polyorder)
        else:
            return trace
            
    elif method == "moving_average":
        window = kwargs.get("window", 5)
        kernel = np.ones(window) / window
        return np.convolve(trace, kernel, mode="same")
        
    elif method == "median":
        kernel_size = kwargs.get("kernel_size", 5)
        return medfilt(trace, kernel_size)
        
    elif method == "gaussian":
        sigma = kwargs.get("sigma", 1.0)
        return ndimage.gaussian_filter1d(trace, sigma)
        
    elif method == "butterworth":
        # Low-pass Butterworth filter
        cutoff = kwargs.get("cutoff", 0.1)  # Normalized frequency
        order = kwargs.get("order", 2)
        
        if len(trace) > order * 3:  # Minimum length for filter
            b, a = butter(order, cutoff, btype='low')
            return filtfilt(b, a, trace)
        else:
            return trace
            
    return trace

def detect_peaks(trace, height=None, distance=None, prominence=None):
    """
    Detect peaks in ECG trace with various criteria.
    """
    from scipy.signal import find_peaks
    
    if height is None:
        height = np.mean(trace) + 0.5 * np.std(trace)
    
    if distance is None:
        distance = len(trace) // 20  # Minimum distance between peaks
    
    peaks, properties = find_peaks(trace, height=height, distance=distance, prominence=prominence)
    
    return peaks, properties

def calculate_heart_rate(peaks, sampling_rate=None, time_duration=None):
    """
    Calculate heart rate from detected peaks.
    
    Parameters:
    - peaks: array of peak indices
    - sampling_rate: samples per second (if known)
    - time_duration: total time duration in seconds (if known)
    """
    if len(peaks) < 2:
        return None
    
    if sampling_rate is not None:
        # Calculate using sampling rate
        intervals = np.diff(peaks) / sampling_rate  # in seconds
        heart_rate = 60 / np.mean(intervals)  # beats per minute
    elif time_duration is not None:
        # Calculate using total duration
        heart_rate = (len(peaks) - 1) * 60 / time_duration  # beats per minute
    else:
        # Estimate based on typical ECG characteristics
        # Assume typical ECG strip represents 10 seconds
        estimated_duration = 10.0
        heart_rate = (len(peaks) - 1) * 60 / estimated_duration
    
    return heart_rate

def remove_baseline_wander(trace, method="polynomial", **kwargs):
    """
    Remove baseline wander from ECG signal.
    """
    if method == "polynomial":
        degree = kwargs.get("degree", 3)
        x = np.arange(len(trace))
        coeffs = np.polyfit(x, trace, degree)
        baseline = np.polyval(coeffs, x)
        return trace - baseline
        
    elif method == "moving_median":
        window = kwargs.get("window", len(trace) // 10)
        baseline = ndimage.median_filter(trace, size=window)
        return trace - baseline
        
    elif method == "highpass":
        cutoff = kwargs.get("cutoff", 0.05)
        order = kwargs.get("order", 2)
        
        if len(trace) > order * 3:
            b, a = butter(order, cutoff, btype='high')
            return filtfilt(b, a, trace)
        else:
            return trace
    
    return trace

def normalize_signal(trace, method="minmax"):
    """
    Normalize ECG signal.
    """
    if method == "minmax":
        min_val, max_val = np.min(trace), np.max(trace)
        if max_val > min_val:
            return (trace - min_val) / (max_val - min_val)
        else:
            return trace
            
    elif method == "zscore":
        mean_val, std_val = np.mean(trace), np.std(trace)
        if std_val > 0:
            return (trace - mean_val) / std_val
        else:
            return trace - mean_val
            
    elif method == "robust":
        median_val = np.median(trace)
        mad = np.median(np.abs(trace - median_val))
        if mad > 0:
            return (trace - median_val) / mad
        else:
            return trace - median_val
    
    return trace
