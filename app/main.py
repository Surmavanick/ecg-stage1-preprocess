from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import httpx

# ECG preprocessing მოდული
from app import ecg_preprocess

app = FastAPI()

# შენი n8n webhook URL (production)
WEBHOOK_URL = "https://foodmart.app.n8n.cloud/webhook/ecg-ready"


@app.get("/health")
async def health():
    """Health-check endpoint."""
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
    პარალელურად იგზავნება შედეგი n8n webhook-ზე.
    """
    try:
        # 📥 ფაილის წამოღება
        file_bytes = await image.read()

        # 🧠 ECG preprocessing pipeline
        result = ecg_preprocess.run_pipeline(file_bytes)

        response_data = {
            "ok": True,
            "debug": result.get("debug", {}),
            "download_urls": result.get("download_urls", {})
        }

        # 🚀 შედეგის გადაგზავნა n8n-ში
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
        return JSONResponse(
            content={"ok": False, "error": str(e)},
            status_code=500
        )
