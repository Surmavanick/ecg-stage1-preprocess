import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
import httpx

from app import ecg_preprocess

app = FastAPI()

WEBHOOK_URL = "https://foodmart.app.n8n.cloud/webhook/ecg-ready"
OUTPUT_DIR = "/app/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ecg-photo/preprocess")
async def preprocess_ecg_photo(
    image: UploadFile = File(...),
    layout_hint: str = Form("string"),
    speed_hint: int = Form(0),
    gain_hint: int = Form(0)
):
    try:
        file_bytes = await image.read()
        result = ecg_preprocess.run_pipeline(file_bytes, speed_hint=speed_hint, gain_hint=gain_hint)

        response_data = {
            "ok": True,
            "debug": result.get("debug", {}),
            "images": result.get("images", {}),
            "masks": result.get("masks", {}),
            "download_urls": result.get("download_urls", {})
        }

        # Forward to n8n webhook
        async with httpx.AsyncClient() as client:
            try:
                await client.post(WEBHOOK_URL, json=response_data, timeout=30)
            except Exception as webhook_error:
                print(f"⚠️ Webhook error: {webhook_error}")

        return JSONResponse(content=response_data)

    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)}, status_code=500)


@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        return JSONResponse(content={"ok": False, "error": "File not found"}, status_code=404)
    return FileResponse(file_path, filename=filename)
