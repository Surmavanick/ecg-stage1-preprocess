from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import numpy as np
import matplotlib.pyplot as plt
import os
import uuid
import cv2
import zipfile

from app.ecg_preprocess import image_to_sequence, remove_ecg_labels
from app.utils import smooth_signal

app = FastAPI(title="ECG Preprocess API")

UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


@app.post("/process")
async def process_ecg(file: UploadFile = File(...)):
    contents = await file.read()
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(contents)

    trace = image_to_sequence(file_path)

    peaks = []
    for i in range(1, len(trace) - 1):
        if trace[i] < trace[i-1] and trace[i] < trace[i+1]:
            peaks.append(i)

    return JSONResponse(content={
        "length": len(trace),
        "peaks": peaks,
        "num_peaks": len(peaks)
    })


@app.post("/plot")
async def plot_ecg(file: UploadFile = File(...)):
    contents = await file.read()
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(contents)

    # Debug image (ტექსტის გარეშე)
    img = cv2.imread(file_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cleaned = remove_ecg_labels(gray)

    debug_name = f"debug_{uuid.uuid4().hex}.png"
    debug_path = os.path.join(RESULT_DIR, debug_name)
    cv2.imwrite(debug_path, cleaned)

    # ECG trace
    trace = image_to_sequence(file_path)
    trace = np.max(trace) - trace
    trace_smooth = smooth_signal(trace, "savgol")

    plt.figure(figsize=(10, 3))
    plt.plot(trace_smooth, color="green")
    plt.title("Extracted ECG Trace")
    plt.tight_layout()

    ecg_name = f"ecg_{uuid.uuid4().hex}.png"
    ecg_path = os.path.join(RESULT_DIR, ecg_name)
    plt.savefig(ecg_path)
    plt.close()

    # ZIP შევქმნათ
    zip_name = f"ecg_results_{uuid.uuid4().hex}.zip"
    zip_path = os.path.join(RESULT_DIR, zip_name)
    with zipfile.ZipFile(zip_path, "w") as zipf:
        zipf.write(ecg_path, arcname=ecg_name)
        zipf.write(debug_path, arcname=debug_name)

    return FileResponse(zip_path, media_type="application/zip", filename=zip_name)
