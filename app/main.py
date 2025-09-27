# main.py

import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
import httpx

from app import ecg_preprocess

app = FastAPI()

WEBHOOK_URL = os.getenv("WEBHOOK_URL", "https://your-webhook-url.n8n.cloud/webhook/ecg-ready")
OUTPUT_DIR = "/app/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/health")
async def health():
    return {"status": "ok"}


# --- CHANGES HERE ---
@app.post("/ecg-photo/preprocess")
async def preprocess_ecg_photo(
    image: UploadFile = File(...),
    grid_color: str = Form("red"),  # New form field for grid color
    layout_hint: str = Form("string"),
    speed_hint: int = Form(0),
    gain_hint: int = Form(0)
):
    """
    Receives an ECG photo, processes it, and returns a JSON.
    The grid color can now be specified ('red', 'blue', 'green').
    """
    try:
        file_bytes = await image.read()

        # --- AND HERE ---
        # Pass the grid_color from the form to the pipeline
        result = ecg_preprocess.run_pipeline(file_bytes, grid_color=grid_color)

        response_data = {
            "ok": True,
            "debug": result.get("debug", {}),
            "images": result.get("images", {}),
            "masks": result.get("masks", {}),
            "download_urls": result.get("download_urls", {})
        }

        if WEBHOOK_URL:
             async with httpx.AsyncClient() as client:
                try:
                    print(f"📡 Sending to n8n webhook: {WEBHOOK_URL}")
                    resp = await client.post(WEBHOOK_URL, json=response_data, timeout=30)
                    print(f"✅ n8n response: {resp.status_code}")
                except Exception as webhook_error:
                    print(f"⚠️ Failed to send to n8n webhook: {webhook_error}")

        return JSONResponse(content=response_data)

    except Exception as e:
        print(f"❌ Error in /ecg-photo/preprocess endpoint: {e}")
        return JSONResponse(content={"ok": False, "error": str(e)}, status_code=500)


@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        return JSONResponse(content={"ok": False, "error": "File not found"}, status_code=404)
    return FileResponse(file_path, filename=filename)
