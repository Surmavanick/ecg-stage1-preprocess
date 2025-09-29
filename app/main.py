from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import matplotlib.pyplot as plt
import numpy as np
import cv2
import uuid
import os

app = FastAPI()

@app.post("/process", response_class=FileResponse)
async def process_ecg(file: UploadFile = File(...)):
    # დროებითი ფაილის ჩაწერა
    raw_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
    with open(raw_path, "wb") as f:
        f.write(await file.read())

    # ECG სურათის ჩატვირთვა
    img = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Invalid image file")

    # აქ შეგიძლია ჩასვა შენი რეალური image_to_sequence
    # ეს მაგალითი უბრალოდ ხაზს აგენერირებს
    trace = np.mean(img, axis=0)

    # შედეგის გრაფიკი PNG-ში
    out_path = f"/tmp/ecg_result_{uuid.uuid4()}.png"
    plt.figure(figsize=(8,3))
    plt.plot(trace, color="green")
    plt.title("Extracted ECG Trace")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    return FileResponse(out_path, media_type="image/png", filename="ecg_result.png")
