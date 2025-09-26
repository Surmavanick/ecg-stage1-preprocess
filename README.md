# ECG Photo Preprocess – Stage 1 (Starter Kit)

This is a minimal FastAPI microservice that does **Stage 1** of your pipeline:
- Deskew/dewarp an ECG photo
- Remove the red grid (if present)
- Extract a binary mask of the ECG trace
- Estimate pixels-per-mm from the grid
- Return base64-encoded debug images and metrics

## Run (Option A: Docker)

1) Install Docker Desktop.
2) In terminal:
```
docker build -t ecg-preprocess .
docker run --rm -p 8000:8000 -v ${PWD}/output:/app/output ecg-preprocess
```
3) Open http://localhost:8000/docs in your browser. Use the `/ecg-photo/preprocess` endpoint.

## Run (Option B: Python locally)

1) Install Python 3.10+
2) In terminal:
```
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
3) Open http://localhost:8000/docs

## Notes
- Debug images are written to `./output` by default.
- This is a classical CV baseline. You can add a learned model later if needed.
- Not a medical device. For triage/research only.
