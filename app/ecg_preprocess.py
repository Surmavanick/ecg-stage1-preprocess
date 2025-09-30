import cv2
import numpy as np

def remove_ecg_labels(gray):
    """
    Remove ECG text labels (I, II, III, aVR, aVL, aVF, V1–V6) from ECG image.
    Uses contour filtering + morphology to eliminate black letters too.
    """
    # Binary inverse: ტექსტი გამოკვეთილი გახდეს
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Morphology → ასოების შტრიხები ერთიანდება
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ვიპოვოთ ყველა ობიექტი
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(gray)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # ECG ხაზი გრძელია, ტექსტი შედარებით პატარაა
        if 10 < w < 120 and 10 < h < 80:
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    # ამოვშალოთ ტექსტი
    cleaned = cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)
    return cleaned


def image_to_sequence(img_path, mode="dark-foreground", method="moving_average", window=5):
    """
    Extract ECG signal from an image.
    Includes:
    - text removal (lead labels)
    - neighbor consistency filter
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert if needed
    if mode == "bright-foreground":
        gray = cv2.bitwise_not(gray)

    # Step 1: მოვაშოროთ ტექსტები (I, II, V1–V6, aVR etc.)
    gray = remove_ecg_labels(gray)

    h, w = gray.shape
    trace = []
    last_row = None

    # Step 2: სვეტების დამუშავება + neighbor consistency
    for col in range(w):
        column = gray[:, col]
        row = np.argmin(column)

        if last_row is not None and abs(row - last_row) > 30:
            # ეჭვი გვაქვს რომ ტექსტია, მოვძებნოთ სხვა კანდიდატი
            candidates = np.where(column < np.percentile(column, 5))[0]
            if len(candidates) > 0:
                row = candidates[np.argmin(np.abs(candidates - last_row))]
            else:
                row = last_row  # fallback to previous value

        trace.append(row)
        last_row = row

    trace = np.array(trace, dtype=float)

    # Moving average filter
    if method == "moving_average":
        kernel = np.ones(window) / window
        trace = np.convolve(trace, kernel, mode="same")

    return trace
