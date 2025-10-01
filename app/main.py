from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse, FileResponse
import numpy as np
import matplotlib.pyplot as plt
import os
import uuid
from typing import Optional

# Direct imports from same directory (Corrected with relative imports)
from .ecg_preprocess import image_to_sequence, preprocess_ecg_image # <--- შეიცვალა
from .utils import smooth_signal, detect_peaks # <--- შეიცვალა

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
    Process ECG image and return basic metrics.
    """
    # Save the uploaded file temporarily
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        contents = await file.read()
        f.write(contents)

    try:
        trace = image_to_sequence(file_path, method=method, remove_artifacts=remove_artifacts)
        
        # --- გაუმჯობესებული პიკების დეტექცია ---
        # გამოვიყენოთ უფრო სანდო ფუნქცია utils.py-დან მარტივი ციკლის ნაცვლად
        peaks, _ = detect_peaks(trace, height=np.mean(trace), distance=len(trace) // 20)
        peaks_list = peaks.tolist() # Convert numpy array to list for JSON serialization

        return JSONResponse(content={
            "length": len(trace),
            "peaks": peaks_list,
            "num_peaks": len(peaks_list),
            "method_used": method,
            "artifacts_removed": remove_artifacts,
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
        # Clean up the uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/plot")
async def plot_ecg(
    file: UploadFile = File(...),
    method: str = Query("contour", description="Extraction method: simple, contour, or morphological"),
    remove_artifacts: bool = Query(True, description="Remove text and artifacts"),
    smooth_method: str = Query("savgol", description="Smoothing method: savgol or moving_average")
):
    """
    Process ECG image and return a plot.
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        contents = await file.read()
        f.write(contents)

    try:
        trace = image_to_sequence(file_path, method=method, remove_artifacts=remove_artifacts)

        # Invert + smooth
        trace = np.max(trace) - trace
        trace_smooth = smooth_signal(trace, smooth_method)

        # Create comparison plot
        plt.figure(figsize=(15, 8))
        
        # Original trace
        plt.subplot(2, 1, 1)
        plt.plot(trace, color="blue", alpha=0.7, label="Raw extracted")
        plt.title(f"Raw ECG Trace (Method: {method}, Artifacts removed: {remove_artifacts})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Smoothed trace
        plt.subplot(2, 1, 2)
        plt.plot(trace_smooth, color="green", linewidth=2, label=f"Smoothed ({smooth_method})")
        plt.title("Processed ECG Trace")
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
        # Clean up the uploaded file
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
        # Clean up the uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/")
async def root():
    return {"message": "ECG Preprocessing API", "docs": "/docs"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
