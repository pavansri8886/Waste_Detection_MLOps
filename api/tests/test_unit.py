import os
from pathlib import Path
from fastapi.testclient import TestClient

os.environ.setdefault("API_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data"))

from api.main import app, DB_PATH
import sqlite3

client = TestClient(app)


def setup_module() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
    os.makedirs(DB_PATH.parent, exist_ok=True)
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
