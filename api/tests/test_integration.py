import shutil
import subprocess
import time
from pathlib import Path

import httpx
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
IMAGE_NAME = "waste-api-integration:test"
CONTAINER_NAME = "waste-api-integration-test"
API_URL = "http://127.0.0.1:18000"


def run_docker_command(*args: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    output_args = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT} if capture else {}
    return subprocess.run(
        ["docker", *args],
        cwd=ROOT_DIR,
        text=True,
        check=check,
        **output_args,
    )


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker is required for integration tests")
def test_api_end_to_end_via_docker() -> None:
    run_docker_command("rm", "-f", CONTAINER_NAME, check=False)
    run_docker_command("build", "-t", IMAGE_NAME, "api")
    run_docker_command(
        "run",
        "-d",
        "--name",
        CONTAINER_NAME,
        "-p",
        "18000:8000",
        "-e",
        "API_DATA_DIR=/tmp/data",
        "-e",
        "API_LOG_DIR=/tmp/logs",
        "-e",
        "ENABLE_MLFLOW_BOOTSTRAP=0",
        IMAGE_NAME,
    )

    try:
        deadline = time.time() + 45
        while time.time() < deadline:
            try:
                response = httpx.get(f"{API_URL}/health", timeout=2)
                if response.status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(1)
        else:
            logs = run_docker_command("logs", CONTAINER_NAME, check=False, capture=True).stdout
            pytest.fail(f"API container did not become healthy. Logs:\n{logs}")

        assert httpx.get(f"{API_URL}/health").json() == {"status": "ok"}

        models_response = httpx.get(f"{API_URL}/models")
        assert models_response.status_code == 200
        assert len(models_response.json()) == 8

        with (ROOT_DIR / "test_image.jpg").open("rb") as image_file:
            predict_response = httpx.post(
                f"{API_URL}/predict",
                files={"file": ("test_image.jpg", image_file, "image/jpeg")},
                data={"latitude": "48.8566", "longitude": "2.3522", "model_name": "yolov8"},
                timeout=10,
            )
        assert predict_response.status_code == 200
        payload = predict_response.json()
        assert payload["rubbish"] == "rubbish"
        assert payload["model_used"] == "yolov8"
        assert 0 <= payload["confiance"] <= 1

        history_response = httpx.get(f"{API_URL}/history")
        assert history_response.status_code == 200
        assert history_response.json()
    finally:
        run_docker_command("rm", "-f", CONTAINER_NAME, check=False)
