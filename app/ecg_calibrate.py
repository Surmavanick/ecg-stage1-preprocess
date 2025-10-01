import cv2
import numpy as np

def estimate_px_per_mm(img_bgr) -> float:
    """Estimate px_per_mm from ECG grid using vertical & horizontal profiles."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    roi = gray[int(0.1*h):int(0.9*h), int(0.1*w):int(0.9*w)]
    roi = cv2.GaussianBlur(roi, (3,3), 0)

    prof_x = 255 - np.mean(roi, axis=0)
    prof_y = 255 - np.mean(roi, axis=1)

    def median_period(profile):
        p = profile - np.mean(profile)
        p[p < 0] = 0
        peaks = np.where((p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:]) & (p[1:-1] > np.percentile(p, 85)))[0] + 1
        if len(peaks) < 4:
            return None
        d = np.diff(peaks)
        d = d[(d > 3) & (d < 100)]
        if len(d) == 0:
            return None
        return float(np.median(d))

    dx = median_period(prof_x)
    dy = median_period(prof_y)
    vals = [v for v in [dx, dy] if v is not None]

    if not vals:
        return 12.0  # fallback ≈300 dpi
    if len(vals) == 2 and abs(dx - dy) / max(dx, dy) < 0.08:
        return float((dx + dy) / 2.0)
    return float(max(vals))
