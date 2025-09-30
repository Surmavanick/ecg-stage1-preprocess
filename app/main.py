from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
import numpy as np
import matplotlib.pyplot as plt
import os
import uuid

from app.ecg_preprocess import image_to_sequence
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

    trace = image_to_sequence(file_path)

    # Invert + smooth
    trace = np.max(trace) - trace
    trace_smooth = smooth_signal(trace, "savgol")

    # Plot
    plt.figure(figsize=(10, 3))
    plt.plot(trace_smooth, color="green")
    plt.title("Extracted ECG Trace")
    plt.tight_layout()

    out_name = f"ecg_{uuid.uuid4().hex}.png"
    out_path = os.path.join(RESULT_DIR, out_name)
    plt.savefig(out_path)
    plt.close()

    return FileResponse(out_path, media_type="image/png", filename=out_name)
