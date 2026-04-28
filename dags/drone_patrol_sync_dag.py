from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3

PATROL_DB = Path("/data/drone_patrol.db")
APP_DB = Path("/data/app_detections.db")


def sync_patrol_data() -> None:
    PATROL_DB.parent.mkdir(parents=True, exist_ok=True)
    APP_DB.parent.mkdir(parents=True, exist_ok=True)

    app_conn = sqlite3.connect(str(APP_DB))
    app_conn.execute(
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
    app_conn.commit()

    patrol_conn = sqlite3.connect(str(PATROL_DB))
    rows = patrol_conn.execute(
        "SELECT id, drone_id, timestamp, latitude, longitude, classe, confiance, image_filename FROM drone_detections WHERE processed = 0"
    ).fetchall()

    if not rows:
        print("No new patrol detections to sync")
        patrol_conn.close()
        app_conn.close()
        return

    insert_rows = [
        (
            row[2],
            f"drone:{row[1]}",
            row[3],
            row[4],
            "drone_simulator",
            row[5],
            row[6],
            row[7],
        )
        for row in rows
    ]

    app_conn.executemany(
        "INSERT INTO app_detections (timestamp, source, latitude, longitude, model_name, prediction, confiance, filename) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        insert_rows,
    )
    app_conn.commit()

    patrol_conn.executemany(
        "UPDATE drone_detections SET processed = 1 WHERE id = ?",
        [(row[0],) for row in rows],
    )
    patrol_conn.commit()

    print(f"Synced {len(rows)} patrol detections into {APP_DB}")
    patrol_conn.close()
    app_conn.close()


with DAG(
    dag_id="drone_patrol_sync",
    start_date=datetime(2026, 1, 1),
    schedule_interval="@hourly",
    catchup=False,
    default_args={"retries": 0, "retry_delay": timedelta(minutes=1)},
) as dag:
    sync_data = PythonOperator(task_id="sync_data", python_callable=sync_patrol_data)

    sync_data
