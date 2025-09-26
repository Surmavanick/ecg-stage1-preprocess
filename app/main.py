import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
import httpx

from app import ecg_preprocess

app = FastAPI()

# n8n webhook URL
WEBHOOK_URL = "https://foodmart.app.n8n.cloud/webhook/ecg-ready"

# Output დირექტორია
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
    """
    იღებს ECG ფოტოს, ამუშავებს და აბრუნებს JSON-ს.
    პარალელურად აგზავნის შედეგს n8n webhook-ზე.
    """
    try:
        # 📥 ფაილის წამოღება
        file_bytes = await image.read()

        # 🧠 ECG preprocessing pipeline
        result = ecg_preprocess.run_pipeline(file_bytes)

        response_data = {
            "ok": True,
            "debug": result.get("debug", {}),
            "images": result.get("images", {}),
            "masks": result.get("masks", {}),
            "download_urls": result.get("download_urls", {})
        }

        # 🚀 გაგზავნა n8n-ში
        async with httpx.AsyncClient() as client:
            try:
                print(f"📡 ვაგზავნი n8n-ზე: {WEBHOOK_URL}")
                resp = await client.post(WEBHOOK_URL, json=response_data, timeout=30)
                print(f"✅ n8n პასუხი: {resp.status_code} - {resp.text}")
            except Exception as webhook_error:
                print(f"⚠️ ვერ გავაგზავნე n8n-ზე: {webhook_error}")

        # 📤 პასუხი API-ს მომხმარებელს
        return JSONResponse(content=response_data)

    except Exception as e:
        print(f"❌ შეცდომა /ecg-photo/preprocess endpoint-ზე: {e}")
        return JSONResponse(content={"ok": False, "error": str(e)}, status_code=500)


@app.get("/download/{filename}")
async def download_file(filename: str):
    """
    აბრუნებს ფაილს /app/output დირექტორიიდან როგორც download.
    """
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        return JSONResponse(content={"ok": False, "error": "File not found"}, status_code=404)
    return FileResponse(file_path, filename=filename)
