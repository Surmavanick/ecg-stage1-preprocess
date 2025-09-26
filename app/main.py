
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import numpy as np
import cv2 as cv
from .ecg_preprocess import preprocess_ecg_photo

app = FastAPI(title="ECG Photo Preprocess – Stage 1")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ecg-photo/preprocess")
async def ecg_photo_preprocess(
    image: UploadFile = File(...),
    layout_hint: str | None = Form(default=None),
    speed_hint: int | None = Form(default=None),
    gain_hint: int | None = Form(default=None),
):
    try:
        data = await image.read()
        file_bytes = np.frombuffer(data, np.uint8)
        bgr = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
        if bgr is None:
            return JSONResponse({"ok": False, "error": "Cannot read image"}, status_code=400)

        result = preprocess_ecg_photo(bgr, speed_hint=speed_hint, gain_hint=gain_hint)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
