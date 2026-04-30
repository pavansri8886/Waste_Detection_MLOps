# Waste Detection MLOps

Drone waste detection application for the MLOps final project.

**Reference assignment repo**: https://github.com/sinaayyy/project_mlops  
**Submission repo**: https://github.com/pavansri8886/Waste_Detection_MLOps  
**Members**: NAGANABOINA Pavan Kumar | pavankumar.naganaboina@edu.ece.fr

![CI](https://github.com/pavansri8886/Waste_Detection_MLOps/actions/workflows/ci.yml/badge.svg)

The project provides a FastAPI inference API, a Streamlit/Folium operator UI, an Airflow drone synchronization pipeline, MLflow model registry integration, Prometheus metrics, Grafana dashboard configuration, Alertmanager routing, and GitHub Actions CI.

## Submission Status

This repository is prepared as a grading entry point: every required component has a reproducible command and a success example below.

| Area | Implemented evidence |
|---|---|
| Packaging | Root requirements file, API Dockerfile, App Dockerfile, full `docker-compose.yml` stack |
| API | `/health`, `/models`, `/predict`, `/history`, validation errors, SQLite persistence |
| Model management | 8 model entries exposed through MLflow-compatible registry URIs with local fallback models for reproducible grading |
| Streamlit | Model dropdown, upload form, prediction result, Folium history map, source/model/date filters |
| Airflow | `drone_mission_simulator` plus `drone_patrol_sync` with `extract -> transform -> load` |
| Observability | `/metrics`, JSONL prediction logs, Prometheus alert rules, Grafana dashboard with 4 panels |
| CI/CD | GitHub Actions workflow runs tests, Docker integration, image builds, and GHCR push |
| Quality | Runtime artifacts are ignored; tracked Python cache files were removed |

Local verification result before submission:

```text
python -m pytest api/tests -q
7 passed
```

Sample outputs below are representative. Timestamps, confidence values, row counts, and run IDs vary each time the stack is executed.

## Prerequisites

- Docker and Docker Compose
- Python 3.11 for local tests
- GitHub CLI optional, only for checking remote CI status

## Setup

```bash
git clone https://github.com/pavansri8886/Waste_Detection_MLOps.git
cd Waste_Detection_MLOps/project_mlops
python generate_patrol_db.py
```

Expected result: `drone_patrol.db` is created or updated with new unprocessed drone detections.

## Start The Stack

```bash
docker compose up -d --build
docker compose ps
```

Expected services:

- `api` on http://localhost:8000
- `app` on http://localhost:8501
- `airflow` on http://localhost:8080
- `mlflow` on http://localhost:5000
- `prometheus` on http://localhost:9090
- `grafana` on http://localhost:3000
- `alertmanager` on http://localhost:9093

Grafana default login: `admin` / `admin`.

## API Verification

```bash
curl -s http://localhost:8000/health
```

Expected:

```json
{"status":"ok"}
```

List models loaded through the MLflow registry interface:

```bash
curl -s http://localhost:8000/models | python -m json.tool
```

Expected: 8 model entries with `name`, `version`, `registered_at`, and `registry_uri`.

Example output:

```json
[
  {
    "name": "yolov8",
    "version": "1",
    "registered_at": "2026-04-30T12:00:00Z",
    "registry_uri": "models:/waste-detector-yolov8/Production"
  },
  {
    "name": "rtdetr",
    "version": "1",
    "registered_at": "2026-04-30T12:00:00Z",
    "registry_uri": "models:/waste-detector-rtdetr/Production"
  }
]
```

The real output contains all 8 configured models.

Run a prediction:

```bash
curl -s -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg" \
  -F "latitude=48.8566" \
  -F "longitude=2.3522" \
  -F "model_name=yolov8" | python -m json.tool
```

Expected fields: `rubbish`, `confiance`, `model_used`, `timestamp`.

Example output:

```json
{
  "rubbish": "rubbish",
  "confiance": 0.91,
  "model_used": "yolov8",
  "timestamp": "2026-04-30T14:22:11Z"
}
```

Verify model selection:

```bash
curl -s -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg" \
  -F "latitude=48.8566" \
  -F "longitude=2.3522" \
  -F "model_name=rtdetr" | python -m json.tool
```

Expected: `model_used` is `rtdetr`.

Example output:

```json
{
  "rubbish": "rubbish",
  "confiance": 0.88,
  "model_used": "rtdetr",
  "timestamp": "2026-04-30T14:24:03Z"
}
```

Verify validation errors:

```bash
curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg" \
  -F "latitude=48.8566" \
  -F "longitude=2.3522" \
  -F "model_name=unknown_model"
```

Expected: `422`.

Example output:

```text
422
```

```bash
curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8000/predict \
  -F "file=@requirements.txt" \
  -F "latitude=48.8566" \
  -F "longitude=2.3522" \
  -F "model_name=yolov8"
```

Expected: `422`.

Example output:

```text
422
```

```bash
curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg" \
  -F "latitude=999" \
  -F "longitude=2.3522" \
  -F "model_name=yolov8"
```

Expected: `422`.

Example output:

```text
422
```

Check history:

```bash
curl -s http://localhost:8000/history | python -m json.tool
```

Example output:

```json
[
  {
    "timestamp": "2026-04-30T14:22:11Z",
    "source": "manual",
    "latitude": 48.8566,
    "longitude": 2.3522,
    "model_name": "yolov8",
    "prediction": "rubbish",
    "confiance": 0.91,
    "filename": "upload_20260430142211000000_test_image.jpg",
    "drone_id": null
  }
]
```

## Automated Tests

Local unit tests:

```bash
python -m pytest api/tests/test_unit.py -v
```

Expected output:

```text
6 passed
```

Docker-based integration test:

```bash
python -m pytest api/tests/test_integration.py -v
```

The integration test builds the API Docker image, starts a container, sends real HTTP requests to `/health`, `/models`, `/predict`, and `/history`, then removes the container.

Expected output:

```text
1 passed
```

## Airflow Pipeline

DAG 1 simulates drone missions every 5 minutes:

```bash
docker compose exec airflow airflow dags list-runs \
  --dag-id drone_mission_simulator --output table
```

Example output:

```text
dag_id                   run_id      state    run_type
drone_mission_simulator  scheduled   success  scheduled
```

DAG 2 synchronizes drone detections every 10 minutes with three tasks:

```bash
docker compose exec airflow airflow dags trigger drone_patrol_sync
docker compose exec airflow airflow dags list-runs \
  --dag-id drone_patrol_sync --output table
```

Example output:

```text
dag_id             run_id      state    run_type
drone_patrol_sync  manual__... success  manual
```

Check task states for a run:

```bash
docker compose exec airflow airflow tasks states-for-dag-run \
  drone_patrol_sync <run_id> --output table
```

Expected tasks: `extract`, `transform`, `load`.

Example output:

```text
task_id    state
extract    success
transform  success
load       success
```

Verify confidence filtering and processed flags:

```bash
docker compose exec api sqlite3 /data/app_detections.db \
  "SELECT MIN(confiance) FROM app_detections WHERE source='drone_patrol';"
```

Expected: value is at least `0.65` when drone rows exist.

Example output:

```text
0.6512
```

```bash
docker compose exec airflow sqlite3 /data/drone_patrol.db \
  "SELECT processed, COUNT(*) FROM drone_detections GROUP BY processed;"
```

Expected: loaded rows are marked `processed = 1`.

Example output:

```text
0|23
1|67
```

## Streamlit Interface

Open http://localhost:8501.

Expected features:

- Model dropdown populated from `GET /models`
- Image upload and GPS inputs
- Prediction result with confidence and selected model
- Folium map with historical detections
- Filters by source, model, and date
- Red markers for manual uploads and orange markers for drone patrol detections

## Observability

Generate a few predictions, then verify API metrics:

```bash
curl -s http://localhost:8000/metrics | grep "^ml_"
```

Expected metrics include:

- `ml_predictions_total`
- `ml_inference_latency_seconds`
- `ml_predictions_by_model_total`
- `ml_validation_errors_total`

Example output:

```text
ml_predictions_total 3.0
ml_inference_latency_seconds_count 3.0
ml_predictions_by_model_total{model="yolov8"} 3.0
ml_validation_errors_total 0.0
```

Verify Prometheus scraping:

```bash
curl -s "http://localhost:9090/api/v1/query?query=ml_predictions_total" | python -m json.tool
```

Example output:

```json
{
  "status": "success",
  "data": {
    "resultType": "vector",
    "result": [
      {
        "metric": {
          "__name__": "ml_predictions_total",
          "job": "api"
        },
        "value": [
          1777557600.0,
          "3"
        ]
      }
    ]
  }
}
```

Verify structured prediction logs:

```bash
tail -5 logs/predictions.jsonl
```

Example output:

```json
{"timestamp":"2026-04-30T14:22:11Z","source":"manual","latitude":48.8566,"longitude":2.3522,"confiance":0.91,"model_name":"yolov8","latence_ms":42.7}
```

Verify Grafana dashboard file:

```bash
python -c "import json; d=json.load(open('monitoring/grafana/dashboard.json')); panels=d.get('panels', []); print(len(panels)); assert len(panels) >= 4"
```

Expected output:

```text
4
```

Open Grafana at http://localhost:3000 and check the `Waste Detection Dashboard`.

Verify Prometheus alert rules:

```bash
curl -s http://localhost:9090/api/v1/rules | python -m json.tool
curl -s http://localhost:9093/api/v2/status | python -m json.tool
```

Expected result:

```text
Prometheus returns at least one rule group, and Alertmanager returns a status response with cluster/status information.
```

## CI/CD

The GitHub Actions workflow runs on pushes and pull requests to `main`.

Expected CI steps:

- Install API dependencies
- Run unit tests
- Run Docker-based integration test
- Build and push API image to GHCR
- Build and push Streamlit image to GHCR

Check latest CI status:

```bash
gh run list --repo pavansri8886/Waste_Detection_MLOps --limit 1
```

Check public API image:

```bash
docker pull ghcr.io/pavansri8886/waste-api:latest
```

## Git And Quality Checks

Actual repository history check:

```bash
git shortlog -sn --all
git log --oneline -15
```

Example output:

```text
10  pavansri8886
 2  Yanis Bardes
 2  sinaayyy

1b8f92b Document expected verification outputs
f7d3cea Improve MLOps submission readiness
20a59f4 Update README.md
```

Check ignored runtime artifacts:

```bash
git ls-files | grep -E "(__pycache__|\.venv|\.db$|\.sqlite$|\.sqlite3$)"
```

Expected: no output.

Actual result after cleanup:

```text
# no tracked __pycache__, virtualenv, or SQLite runtime database files
```

The professor must be invited to the private repository before the deadline.

## Bonus

No bonus component is claimed in this README.
