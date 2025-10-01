from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse, FileResponse
import numpy as np
import matplotlib.pyplot as plt
import os
import uuid
import cv2
import csv

# Local imports
from .ecg_preprocess import image_to_sequence, preprocess_ecg_image
from .utils import smooth_signal, detect_peaks
from .ecg_calibrate import estimate_px_per_mm

app = FastAPI(title="ECG Preprocess API")

UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# --- Helpers ---
def trim_edges(trace, trim_ratio=0.05):
    """
    Trim start and end of ECG trace (default 5% each side).
    """
    n = len(trace)
    cut = int(n * trim_ratio)
    if n > 2 * cut:
        return trace[cut:-cut]
    return trace


def save_trace_to_csv(trace_mV, fs, out_path):
    """
    Save cleaned ECG trace to CSV with columns: time (s), amplitude (mV).
    """
    t = np.arange(len(trace_mV)) / fs
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "amplitude_mV"])
        for ti, ai in zip(t, trace_mV):
            writer.writerow([ti, ai])
    return out_path


@app.post("/process")
async def process_ecg(
    file: UploadFile = File(...),
    method: str = Query("contour", description="Extraction method: simple, contour, or morphological"),
    remove_artifacts: bool = Query(True, description="Remove text and artifacts")
):
    """
    Process ECG image and return metrics + calibration + CSV export.
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

        # Extract & calibrate
        trace = image_to_sequence(file_path, method=method, remove_artifacts=remove_artifacts)
        trace = np.max(trace) - trace
        trace_mV = (trace / px_per_mm) * 0.1

        # Trim edges
        trace_mV = trim_edges(trace_mV, trim_ratio=0.05)

        # Save CSV
        csv_name = f"ecg_{uuid.uuid4().hex[:8]}.csv"
        csv_path = os.path.join(RESULT_DIR, csv_name)
        save_trace_to_csv(trace_mV, fs, csv_path)

        # Detect peaks
        peaks, _ = detect_peaks(trace_mV, height=np.mean(trace_mV), distance=len(trace_mV)//20)

        return JSONResponse(content={
            "length": len(trace_mV),
            "num_peaks": len(peaks.tolist()),
            "peaks": peaks.tolist(),
            "px_per_mm": px_per_mm,
            "fs_hz": fs,
            "csv_file": csv_name
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
    Process ECG image and return calibrated plot (PNG) + CSV file.
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

        # Extract & calibrate
        trace = image_to_sequence(file_path, method=method, remove_artifacts=remove_artifacts)
        trace = np.max(trace) - trace
        trace_mV = (trace / px_per_mm) * 0.1

        # Trim edges
        trace_mV = trim_edges(trace_mV, trim_ratio=0.05)

        # Save CSV
        csv_name = f"ecg_{uuid.uuid4().hex[:8]}.csv"
        csv_path = os.path.join(RESULT_DIR, csv_name)
        save_trace_to_csv(trace_mV, fs, csv_path)

        # Plot
        t = np.arange(len(trace_mV)) / fs
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

        return {
            "plot": FileResponse(out_path, media_type="image/png", filename=out_name),
            "csv": FileResponse(csv_path, media_type="text/csv", filename=csv_name)
        }
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
