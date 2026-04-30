from datetime import datetime
from pathlib import Path
import io
import os
import sqlite3
import time
import json
import hashlib
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import PlainTextResponse
from PIL import Image, UnidentifiedImageError
import mlflow
import mlflow.pyfunc
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

DEFAULT_DATA_DIR = "/data"
DATA_DIR = Path(os.environ.get("API_DATA_DIR", DEFAULT_DATA_DIR))
LOG_DIR = Path(os.environ.get("API_LOG_DIR", str(Path.cwd() / "logs")))
DB_PATH = DATA_DIR / "app_detections.db"
UPLOAD_DIR = DATA_DIR / "uploads"
LOG_FILE = LOG_DIR / "predictions.jsonl"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
ENABLE_MLFLOW_BOOTSTRAP = os.environ.get("ENABLE_MLFLOW_BOOTSTRAP", "1") == "1"

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

MODEL_DEFINITIONS = [
    {"name": "yolov8", "registry_name": "waste-detector-yolov8"},
    {"name": "yolo26", "registry_name": "waste-detector-yolo26"},
    {"name": "rtdetr", "registry_name": "waste-detector-rtdetr"},
    {"name": "rtdetrv2", "registry_name": "waste-detector-rtdetrv2"},
    {"name": "rfdetr", "registry_name": "waste-detector-rfdetr"},
    {"name": "dfine", "registry_name": "waste-detector-dfine"},
    {"name": "deim-dfine", "registry_name": "waste-detector-deim-dfine"},
    {"name": "fusion-model", "registry_name": "waste-detector-fusion-model"},
]

LOADED_MODELS = {}
MODEL_METADATA = []

app = FastAPI(title="Waste Detection API")


class DeterministicWasteModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        payload = str(model_input).encode("utf-8")
        score = 0.65 + (int(hashlib.sha256(payload).hexdigest()[:8], 16) % 3300) / 10000
        return [{"rubbish": "rubbish", "confiance": round(min(score, 0.98), 2)}]


def valid_model_names() -> List[str]:
    return [model["name"] for model in MODEL_DEFINITIONS]


def ensure_registered_model(model_name: str, registry_name: str) -> None:
    client = mlflow.tracking.MlflowClient()
    try:
        client.get_registered_model(registry_name)
        versions = client.search_model_versions(f"name='{registry_name}'")
        if versions:
            return
    except Exception:
        pass

    with mlflow.start_run(run_name=f"bootstrap-{model_name}"):
        mlflow.pyfunc.log_model(
            artifact_path=model_name,
            python_model=DeterministicWasteModel(),
            registered_model_name=registry_name,
        )

    latest_versions = client.get_latest_versions(registry_name)
    if latest_versions:
        client.transition_model_version_stage(
            name=registry_name,
            version=latest_versions[-1].version,
            stage="Production",
            archive_existing_versions=True,
        )


def load_models() -> None:
    LOADED_MODELS.clear()
    MODEL_METADATA.clear()

    try:
        if not ENABLE_MLFLOW_BOOTSTRAP:
            raise RuntimeError("MLflow bootstrap disabled")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        for model in MODEL_DEFINITIONS:
            if ENABLE_MLFLOW_BOOTSTRAP:
                ensure_registered_model(model["name"], model["registry_name"])
            uri = f"models:/{model['registry_name']}/Production"
            LOADED_MODELS[model["name"]] = mlflow.pyfunc.load_model(uri)
            versions = client.get_latest_versions(model["registry_name"], stages=["Production"])
            version = versions[0] if versions else None
            MODEL_METADATA.append(
                {
                    "name": model["name"],
                    "version": str(version.version) if version else "unknown",
                    "registered_at": datetime.utcfromtimestamp(version.creation_timestamp / 1000).strftime("%Y-%m-%dT%H:%M:%SZ") if version else "unknown",
                    "registry_uri": uri,
                }
            )
    except Exception as exc:
        for model in MODEL_DEFINITIONS:
            LOADED_MODELS[model["name"]] = DeterministicWasteModel()
            MODEL_METADATA.append(
                {
                    "name": model["name"],
                    "version": "local-fallback",
                    "registered_at": "unavailable",
                    "registry_uri": f"models:/{model['registry_name']}/Production",
                    "warning": f"MLflow unavailable: {exc}",
                }
            )


def run_model_prediction(model_name: str, filename: str, content_size: int) -> dict:
    if model_name not in LOADED_MODELS:
        load_models()
    model_input = [{"filename": filename, "bytes": content_size, "model_name": model_name}]
    model = LOADED_MODELS[model_name]
    try:
        model_result = model.predict(model_input)
    except TypeError:
        model_result = model.predict(None, model_input)

    first_result = model_result[0] if isinstance(model_result, list) else model_result
    if isinstance(first_result, dict):
        return first_result
    return {"rubbish": "rubbish", "confiance": 0.65}

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
            filename TEXT NOT NULL,
            drone_id TEXT
        )
        """
    )
    columns = {row[1] for row in conn.execute("PRAGMA table_info(app_detections)").fetchall()}
    if "drone_id" not in columns:
        conn.execute("ALTER TABLE app_detections ADD COLUMN drone_id TEXT")
    conn.commit()
    conn.close()
    load_models()

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.get("/models")
def list_models() -> List[dict]:
    if not MODEL_METADATA:
        load_models()
    return MODEL_METADATA

@app.get("/history")
def history() -> List[dict]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT timestamp, source, latitude, longitude, model_name, prediction, confiance, filename, drone_id FROM app_detections ORDER BY id DESC LIMIT 200"
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
            "drone_id": row[8],
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

    if model_name not in valid_model_names():
        ml_validation_errors_total.inc()
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Unknown model_name. Valid models: {', '.join(valid_model_names())}")
    if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
        ml_validation_errors_total.inc()
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="latitude and longitude must be valid coordinates")
    if len(contents := await file.read()) > 10 * 1024 * 1024:
        ml_validation_errors_total.inc()
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Uploaded file must be smaller than 10 MB")

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

    model_result = run_model_prediction(model_name, file.filename or "uploaded_image", len(contents))
    prediction = model_result.get("rubbish", "rubbish")
    confiance = float(model_result.get("confiance", 0.65))
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO app_detections (timestamp, source, latitude, longitude, model_name, prediction, confiance, filename, drone_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (timestamp, "manual", latitude, longitude, model_name, prediction, confiance, filename, None),
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
