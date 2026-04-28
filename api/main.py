from datetime import datetime
from pathlib import Path
import io
import os
import random
import sqlite3
import time
import json
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import PlainTextResponse
from PIL import Image, UnidentifiedImageError
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

DEFAULT_DATA_DIR = "/data"
DATA_DIR = Path(os.environ.get("API_DATA_DIR", DEFAULT_DATA_DIR))
LOG_DIR = Path(os.environ.get("API_LOG_DIR", str(Path(__file__).resolve().parents[2] / "logs")))
DB_PATH = DATA_DIR / "app_detections.db"
UPLOAD_DIR = DATA_DIR / "uploads"
LOG_FILE = LOG_DIR / "predictions.jsonl"

ml_predictions_total = Counter("ml_predictions_total", "Total number of predictions")
ml_inference_latency_seconds = Histogram(
    "ml_inference_latency_seconds",
    "Inference latency in seconds",
)
ml_predictions_by_model_total = Counter(
    "ml_predictions_by_model_total",
    "Number of detections per model",
    ["model"],
)
ml_validation_errors_total = Counter("ml_validation_errors_total", "Validation errors")

MODELS = [
    {"name": "yolov8", "version": "1", "registered_at": "2026-04-25T00:00:00Z"},
    {"name": "yolo26", "version": "1", "registered_at": "2026-04-25T00:00:00Z"},
    {"name": "rtdetr", "version": "1", "registered_at": "2026-04-25T00:00:00Z"},
    {"name": "rtdetrv2", "version": "1", "registered_at": "2026-04-25T00:00:00Z"},
    {"name": "rfdetr", "version": "1", "registered_at": "2026-04-25T00:00:00Z"},
    {"name": "dfine", "version": "1", "registered_at": "2026-04-25T00:00:00Z"},
    {"name": "deim-dfine", "version": "1", "registered_at": "2026-04-25T00:00:00Z"},
    {"name": "fusion-model", "version": "1", "registered_at": "2026-04-25T00:00:00Z"},
]

app = FastAPI(title="Waste Detection API")

@app.on_event("startup")
def startup_event() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS app_detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            source TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            model_name TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confiance REAL NOT NULL,
            filename TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.get("/models")
def list_models() -> List[dict]:
    return MODELS

@app.get("/history")
def history() -> List[dict]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT timestamp, source, latitude, longitude, model_name, prediction, confiance, filename FROM app_detections ORDER BY id DESC LIMIT 50"
    ).fetchall()
    conn.close()
    return [
        {
            "timestamp": row[0],
            "source": row[1],
            "latitude": row[2],
            "longitude": row[3],
            "model_name": row[4],
            "prediction": row[5],
            "confiance": row[6],
            "filename": row[7],
        }
        for row in rows
    ]

@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    model_name: str = Form(...),
) -> dict:
    start_time = time.monotonic()

    if model_name not in [m["name"] for m in MODELS]:
        ml_validation_errors_total.inc()
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Unknown model_name")
    if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
        ml_validation_errors_total.inc()
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="latitude and longitude must be valid coordinates")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()
    except (UnidentifiedImageError, OSError):
        ml_validation_errors_total.inc()
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Uploaded file must be a valid image")

    filename = f"upload_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}_{file.filename}"
    filepath = UPLOAD_DIR / filename
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(contents)

    prediction = "rubbish"
    confiance = round(random.uniform(0.65, 0.98), 2)
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO app_detections (timestamp, source, latitude, longitude, model_name, prediction, confiance, filename) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (timestamp, "manual", latitude, longitude, model_name, prediction, confiance, filename),
    )
    conn.commit()
    conn.close()

    ml_predictions_total.inc()
    ml_predictions_by_model_total.labels(model=model_name).inc()
    prediction_latency = time.monotonic() - start_time
    ml_inference_latency_seconds.observe(prediction_latency)

    log_entry = {
        "timestamp": timestamp,
        "source": "manual",
        "latitude": latitude,
        "longitude": longitude,
        "confiance": confiance,
        "model_name": model_name,
        "latence_ms": round(prediction_latency * 1000, 2),
    }
    with open(LOG_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return {
        "rubbish": prediction,
        "confiance": confiance,
        "model_used": model_name,
        "timestamp": timestamp,
    }
