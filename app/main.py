from fastapi import FastAPI, UploadFile, File
import shutil
import os
import httpx

from app.ecg_preprocess import image_to_sequence
from app.utils import detect_qrs, normalize_signal

app = FastAPI(title="ECG Digitizer API")


@app.get("/")
def root():
    return {"message": "ECG Digitizer API running!"}


@app.post("/process")
async def process_ecg(file: UploadFile = File(...)):
    """
    Upload ECG image and return digitized signal + QRS peaks.
    """
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract ECG trace
    signal = image_to_sequence(temp_path, mode="dark-foreground", method="moving_average", windowlen=5)

    # Normalize
    signal = normalize_signal(signal)

    # Detect peaks
    peaks = detect_qrs(signal)

    # Clean up
    os.remove(temp_path)

    return {
        "length": len(signal),
        "peaks": peaks.tolist(),
        "num_peaks": len(peaks)
    }


@app.get("/ping-external")
async def ping_external():
    """
    Example external API call using httpx.
    """
    async with httpx.AsyncClient() as client:
        r = await client.get("https://httpbin.org/get")
    return {"status": r.status_code, "data": r.json()}
