from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import os
import uuid
import base64
from . import ecg_preprocess

app = FastAPI(title="ECG Photo Preprocess – Stage 1")

OUTPUT_DIR = "/app/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ecg-photo/preprocess")
async def preprocess_ecg_photo(
    image: UploadFile = File(...),
    layout_hint: str = Form("string"),
    speed_hint: int = Form(0),
    gain_hint: int = Form(0),
):
    # run preprocessing
    result = ecg_preprocess.run_pipeline(await image.read())

    # base64 images
    rectified_b64 = result["images"]["rectified_png_b64"]
    trace_b64 = result["masks"]["trace_png_b64"]
    grid_b64 = result["masks"]["grid_png_b64"]

    # save files to /app/output
    def save_and_link(b64_data: str, prefix: str):
        fname = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
        path = os.path.join(OUTPUT_DIR, fname)
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64_data))
        return f"/download/{fname}"

    download_urls = {
        "rectified": save_and_link(rectified_b64, "rect"),
        "trace": save_and_link(trace_b64, "trace"),
        "grid": save_and_link(grid_b64, "grid"),
    }

    return {
        "ok": True,
        "debug": result["debug"],
        "download_urls": download_urls,
    }

@app.get("/download/{filename}")
def download_file(filename: str):
    filepath = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="image/png", filename=filename)
    return {"error": "file not found"}
