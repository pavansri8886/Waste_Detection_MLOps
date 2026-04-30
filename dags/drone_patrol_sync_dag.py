from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pathlib import Path
import json
import sqlite3

PATROL_DB = Path("/data/drone_patrol.db")
APP_DB = Path("/data/app_detections.db")
MIN_CONFIDENCE = 0.65


def ensure_app_schema(conn: sqlite3.Connection) -> None:
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


def extract(**context) -> str:
    PATROL_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(PATROL_DB)) as patrol_conn:
        rows = patrol_conn.execute(
            """
            SELECT id, drone_id, timestamp, latitude, longitude, classe, confiance, image_filename
            FROM drone_detections
            WHERE processed = 0
            ORDER BY id
            """
        ).fetchall()

    payload = [
        {
            "id": row[0],
            "drone_id": row[1],
            "timestamp": row[2],
            "latitude": row[3],
            "longitude": row[4],
            "classe": row[5],
            "confiance": row[6],
            "image_filename": row[7],
        }
        for row in rows
    ]
    print(f"Extracted {len(payload)} unprocessed drone detections")
    return json.dumps(payload)


def transform(**context) -> str:
    extracted = context["ti"].xcom_pull(task_ids="extract")
    rows = json.loads(extracted or "[]")
    filtered = [row for row in rows if float(row["confiance"]) >= MIN_CONFIDENCE]
    print(f"Kept {len(filtered)} detections with confiance >= {MIN_CONFIDENCE}")
    return json.dumps(filtered)


def load(**context) -> None:
    transformed = context["ti"].xcom_pull(task_ids="transform")
    rows = json.loads(transformed or "[]")
    if not rows:
        print("No qualifying drone detections to load")
        return

    APP_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(APP_DB)) as app_conn:
        ensure_app_schema(app_conn)
        app_conn.executemany(
            """
            INSERT INTO app_detections
                (timestamp, source, latitude, longitude, model_name, prediction, confiance, filename, drone_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row["timestamp"],
                    "drone_patrol",
                    row["latitude"],
                    row["longitude"],
                    "drone_simulator",
                    row["classe"],
                    row["confiance"],
                    row["image_filename"],
                    row["drone_id"],
                )
                for row in rows
            ],
        )
        app_conn.commit()

    with sqlite3.connect(str(PATROL_DB)) as patrol_conn:
        patrol_conn.executemany(
            "UPDATE drone_detections SET processed = 1 WHERE id = ?",
            [(row["id"],) for row in rows],
        )
        patrol_conn.commit()

    print(f"Loaded {len(rows)} drone detections into {APP_DB}")


with DAG(
    dag_id="drone_patrol_sync",
    start_date=datetime(2026, 1, 1),
    schedule_interval="*/10 * * * *",
    catchup=False,
    default_args={"retries": 0, "retry_delay": timedelta(minutes=1)},
) as dag:
    extract_task = PythonOperator(task_id="extract", python_callable=extract)
    transform_task = PythonOperator(task_id="transform", python_callable=transform)
    load_task = PythonOperator(task_id="load", python_callable=load)

    extract_task >> transform_task >> load_task
