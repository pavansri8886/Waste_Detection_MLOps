# Waste Detection MLOps

Drone waste detection application for the MLOps final project.

**Reference assignment repo**: https://github.com/sinaayyy/project_mlops  
**Submission repo**: https://github.com/pavansri8886/Waste_Detection_MLOps  
**Members**: NAGANABOINA Pavan Kumar | pavankumar.naganaboina@edu.ece.fr

![CI](https://github.com/pavansri8886/Waste_Detection_MLOps/actions/workflows/ci.yml/badge.svg)

The project provides a FastAPI inference API, a Streamlit/Folium operator UI, an Airflow drone synchronization pipeline, MLflow model registry integration, Prometheus metrics, Grafana dashboard configuration, Alertmanager routing, and GitHub Actions CI.

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

Run a prediction:

```bash
curl -s -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg" \
  -F "latitude=48.8566" \
  -F "longitude=2.3522" \
  -F "model_name=yolov8" | python -m json.tool
```

Expected fields: `rubbish`, `confiance`, `model_used`, `timestamp`.

Verify model selection:

```bash
curl -s -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg" \
  -F "latitude=48.8566" \
  -F "longitude=2.3522" \
  -F "model_name=rtdetr" | python -m json.tool
```

Expected: `model_used` is `rtdetr`.

Verify validation errors:

```bash
curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg" \
  -F "latitude=48.8566" \
  -F "longitude=2.3522" \
  -F "model_name=unknown_model"
```

Expected: `422`.

```bash
curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8000/predict \
  -F "file=@requirements.txt" \
  -F "latitude=48.8566" \
  -F "longitude=2.3522" \
  -F "model_name=yolov8"
```

Expected: `422`.

```bash
curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg" \
  -F "latitude=999" \
  -F "longitude=2.3522" \
  -F "model_name=yolov8"
```

Expected: `422`.

Check history:

```bash
curl -s http://localhost:8000/history | python -m json.tool
```

## Automated Tests

Local unit tests:

```bash
python -m pytest api/tests/test_unit.py -v
```

Docker-based integration test:

```bash
python -m pytest api/tests/test_integration.py -v
```

The integration test builds the API Docker image, starts a container, sends real HTTP requests to `/health`, `/models`, `/predict`, and `/history`, then removes the container.

## Airflow Pipeline

DAG 1 simulates drone missions every 5 minutes:

```bash
docker compose exec airflow airflow dags list-runs \
  --dag-id drone_mission_simulator --output table
```

DAG 2 synchronizes drone detections every 10 minutes with three tasks:

```bash
docker compose exec airflow airflow dags trigger drone_patrol_sync
docker compose exec airflow airflow dags list-runs \
  --dag-id drone_patrol_sync --output table
```

Check task states for a run:

```bash
docker compose exec airflow airflow tasks states-for-dag-run \
  drone_patrol_sync <run_id> --output table
```

Expected tasks: `extract`, `transform`, `load`.

Verify confidence filtering and processed flags:

```bash
docker compose exec api sqlite3 /data/app_detections.db \
  "SELECT MIN(confiance) FROM app_detections WHERE source='drone_patrol';"
```

Expected: value is at least `0.65` when drone rows exist.

```bash
docker compose exec airflow sqlite3 /data/drone_patrol.db \
  "SELECT processed, COUNT(*) FROM drone_detections GROUP BY processed;"
```

Expected: loaded rows are marked `processed = 1`.

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

Verify Prometheus scraping:

```bash
curl -s "http://localhost:9090/api/v1/query?query=ml_predictions_total" | python -m json.tool
```

Verify structured prediction logs:

```bash
tail -5 logs/predictions.jsonl
```

Verify Grafana dashboard file:

```bash
python -c "import json; d=json.load(open('monitoring/grafana/dashboard.json')); panels=d.get('panels', []); print(len(panels)); assert len(panels) >= 4"
```

Open Grafana at http://localhost:3000 and check the `Waste Detection Dashboard`.

Verify Prometheus alert rules:

```bash
curl -s http://localhost:9090/api/v1/rules | python -m json.tool
curl -s http://localhost:9093/api/v2/status | python -m json.tool
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

```bash
git shortlog -sn --all
git log --oneline -15
```

Check ignored runtime artifacts:

```bash
git ls-files | grep -E "(__pycache__|\.venv|\.db$|\.sqlite$|\.sqlite3$)"
```

Expected: no output.

The professor must be invited to the private repository before the deadline.

## Bonus

No bonus component is claimed in this README.
