from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse, FileResponse
import numpy as np
import matplotlib.pyplot as plt
import os
import uuid
import cv2
from typing import Optional

# Local imports
from .ecg_preprocess import image_to_sequence, preprocess_ecg_image
from .utils import smooth_signal, detect_peaks
from .ecg_calibrate import estimate_px_per_mm   # <--- ახალი

app = FastAPI(title="ECG Preprocess API")

UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


@app.post("/process")
async def process_ecg(
    file: UploadFile = File(...),
    method: str = Query("contour", description="Extraction method: simple, contour, or morphological"),
    remove_artifacts: bool = Query(True, description="Remove text and artifacts")
):
    """
    Process ECG image and return basic metrics + calibration info.
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        contents = await file.read()
        f.write(contents)

    try:
        # Extract trace
        trace = image_to_sequence(file_path, method=method, remove_artifacts=remove_artifacts)

        # Detect peaks
        peaks, _ = detect_peaks(trace, height=np.mean(trace), distance=len(trace) // 20)
        peaks_list = peaks.tolist()

        # Calibration
        img = cv2.imread(file_path)
        px_per_mm = estimate_px_per_mm(img)
        fs = px_per_mm * 25.0  # Hz

        return JSONResponse(content={
            "length": len(trace),
            "peaks": peaks_list,
            "num_peaks": len(peaks_list),
            "method_used": method,
            "artifacts_removed": remove_artifacts,
            "px_per_mm": px_per_mm,
            "fs_hz": fs,
            "trace_stats": {
                "min": float(np.min(trace)),
                "max": float(np.max(trace)),
                "mean": float(np.mean(trace)),
                "std": float(np.std(trace))
            }
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/plot")
async def plot_ecg(
    file: UploadFile = File(...),
    method: str = Query("contour", description="Extraction method: simple, contour, or morphological"),
    remove_artifacts: bool = Query(True, description="Remove text and artifacts"),
    smooth_method: str = Query("savgol", description="Smoothing method: savgol, moving_average, etc.")
):
    """
    Process ECG image and return a calibrated plot (mV vs seconds).
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        contents = await file.read()
        f.write(contents)

    try:
        # Calibration
        img = cv2.imread(file_path)
        px_per_mm = estimate_px_per_mm(img)
        fs = px_per_mm * 25.0

        # Extract trace
        trace = image_to_sequence(file_path, method=method, remove_artifacts=remove_artifacts)
        trace = np.max(trace) - trace

        # Convert to mV
        trace_mV = (trace / px_per_mm) * 0.1
        t = np.arange(len(trace_mV)) / fs

        # Plot calibrated ECG
        plt.figure(figsize=(15, 6))
        plt.plot(t, smooth_signal(trace_mV, smooth_method), color="green", label="ECG (mV)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (mV)")
        plt.title(f"Calibrated ECG Trace (Method={method}, fs={fs:.1f} Hz)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out_name = f"ecg_{method}_{uuid.uuid4().hex[:8]}.png"
        out_path = os.path.join(RESULT_DIR, out_name)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()

        return FileResponse(out_path, media_type="image/png", filename=out_name)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/preprocess")
async def preprocess_ecg(
    file: UploadFile = File(...),
    output_debug: bool = Query(False, description="Include debug information")
):
    """
    Complete ECG preprocessing with debug output.
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        contents = await file.read()
        f.write(contents)

    try:
        trace, debug_info = preprocess_ecg_image(file_path, output_debug=output_debug)

        response_data = {
            "trace": trace.tolist(),
            "length": len(trace),
            "success": True
        }

        if output_debug:
            response_data["debug_info"] = debug_info

        return JSONResponse(content=response_data)
    except Exception as e:
        return JSONResponse(content={"error": str(e), "success": False}, status_code=500)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.get("/")
async def root():
    return {"message": "ECG Preprocessing API", "docs": "/docs"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
