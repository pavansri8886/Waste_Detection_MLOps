import sys
from pathlib import Path
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from api.main import app

client = TestClient(app)


def test_api_end_to_end() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    response = client.get("/models")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
