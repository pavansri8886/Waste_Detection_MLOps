import json
import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("API_DATA_DIR", str(ROOT_DIR / "data"))
os.environ.setdefault("API_LOG_DIR", str(ROOT_DIR / "logs"))

from api.main import app, DB_PATH, LOG_FILE
import sqlite3

client = TestClient(app)


def setup_module() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
    if LOG_FILE.exists():
        LOG_FILE.unlink()
    os.makedirs(DB_PATH.parent, exist_ok=True)
    os.makedirs(LOG_FILE.parent, exist_ok=True)
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


def test_model_loads() -> None:
    response = client.get("/models")
    assert response.status_code == 200
    payload = response.json()
    assert any(model["name"] == "yolov8" for model in payload)


def test_predict_valid_image() -> None:
    test_image_path = Path(__file__).resolve().parents[2] / "test_image.jpg"
    with test_image_path.open("rb") as img_file:
        response = client.post(
            "/predict",
            files={"file": ("test_image.jpg", img_file, "image/jpeg")},
            data={"latitude": "48.8566", "longitude": "2.3522", "model_name": "yolov8"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_used"] == "yolov8"
    assert "confiance" in payload


def test_predict_invalid_file() -> None:
    response = client.post(
        "/predict",
        files={"file": ("requirements.txt", b"not an image", "text/plain")},
        data={"latitude": "48.8566", "longitude": "2.3522", "model_name": "yolov8"},
    )
    assert response.status_code == 422
    assert "Uploaded file must be a valid image" in response.json()["detail"]


def test_metrics_and_structured_logging() -> None:
    test_image_path = Path(__file__).resolve().parents[2] / "test_image.jpg"
    with test_image_path.open("rb") as img_file:
        response = client.post(
            "/predict",
            files={"file": ("test_image.jpg", img_file, "image/jpeg")},
            data={"latitude": "48.8566", "longitude": "2.3522", "model_name": "yolov8"},
        )
    assert response.status_code == 200

    metrics_response = client.get("/metrics")
    assert metrics_response.status_code == 200
    metrics_text = metrics_response.text
    assert "ml_predictions_total" in metrics_text
    assert "ml_inference_latency_seconds" in metrics_text
    assert "ml_predictions_by_model_total" in metrics_text
    assert "ml_validation_errors_total" in metrics_text
    assert "model=\"yolov8\"" in metrics_text

    assert LOG_FILE.exists()
    with LOG_FILE.open("r", encoding="utf-8") as log_file:
        lines = [line.strip() for line in log_file.readlines() if line.strip()]
    assert lines, "Expected at least one log line in predictions.jsonl"
    entry = json.loads(lines[-1])
    assert entry["source"] == "manual"
    assert entry["model_name"] == "yolov8"
    assert "latence_ms" in entry
