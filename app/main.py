from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import shutil
import os
import io
import httpx
import matplotlib.pyplot as plt

from app.ecg_preprocess import image_to_sequence
from app.utils import detect_qrs, normalize_signal

app = FastAPI(title="ECG Digitizer API")


@app.get("/")
def root():
    return {"message": "ECG Digitizer API running!"}


@app.post("/process")
async def process_ecg(file: UploadFile = File(...)):
    """
    Upload ECG image and return digitized signal + QRS peaks as JSON.
    """
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract ECG trace
    signal = image_to_sequence(temp_path, mode="dark-foreground", method="moving_average", windowlen=5)
    signal = normalize_signal(signal)
    peaks = detect_qrs(signal)

    os.remove(temp_path)

    return {
        "length": len(signal),
        "peaks": peaks.tolist(),
        "num_peaks": len(peaks)
    }


@app.post("/plot")
async def plot_ecg(file: UploadFile = File(...)):
    """
    Upload ECG image and return plotted ECG trace as PNG.
    """
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    signal = image_to_sequence(temp_path, mode="dark-foreground", method="moving_average", windowlen=5)
    signal = normalize_signal(signal)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(signal, color="g")
    ax.set_title("Extracted ECG Signal")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Normalized Amplitude")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    os.remove(temp_path)

    return StreamingResponse(buf, media_type="image/png")


@app.get("/ping-external")
async def ping_external():
    async with httpx.AsyncClient() as client:
        r = await client.get("https://httpbin.org/get")
    return {"status": r.status_code, "data": r.json()}
