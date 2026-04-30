import argparse
import json
import sqlite3
from collections import Counter
from datetime import datetime, UTC
from pathlib import Path
from statistics import mean


DEFAULT_APP_DB = Path("data/app_detections.db")
DEFAULT_PATROL_DB = Path("drone_patrol.db")
DEFAULT_OUTPUT = Path("reports/data_quality_report.json")


def read_app_detections(db_path: Path) -> list[dict]:
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT timestamp, source, latitude, longitude, model_name,
                       confiance, drone_id
                FROM app_detections
                """
            ).fetchall()
        except sqlite3.Error:
            return []
    return [dict(row) for row in rows]


def read_patrol_detections(db_path: Path) -> list[dict]:
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT timestamp, 'drone_patrol' AS source, latitude, longitude,
                       'drone_simulator' AS model_name, confiance, drone_id,
                       processed
                FROM drone_detections
                """
            ).fetchall()
        except sqlite3.Error:
            return []
    return [dict(row) for row in rows]


def demo_records() -> list[dict]:
    return [
        {
            "timestamp": "2026-04-30T10:00:00Z",
            "source": "manual",
            "latitude": 48.8566,
            "longitude": 2.3522,
            "model_name": "yolov8",
            "confiance": 0.91,
            "drone_id": None,
        },
        {
            "timestamp": "2026-04-30T10:05:00Z",
            "source": "drone_patrol",
            "latitude": 48.861,
            "longitude": 2.36,
            "model_name": "drone_simulator",
            "confiance": 0.72,
            "drone_id": "drone_001",
        },
        {
            "timestamp": "2026-04-30T10:10:00Z",
            "source": "drone_patrol",
            "latitude": 45.76,
            "longitude": 4.84,
            "model_name": "drone_simulator",
            "confiance": 0.58,
            "drone_id": "drone_002",
        },
    ]


def validate_record(record: dict) -> list[str]:
    issues = []
    confidence = record.get("confiance")
    latitude = record.get("latitude")
    longitude = record.get("longitude")

    if confidence is None or not 0 <= float(confidence) <= 1:
        issues.append("confidence_out_of_range")
    if latitude is None or not -90 <= float(latitude) <= 90:
        issues.append("latitude_out_of_range")
    if longitude is None or not -180 <= float(longitude) <= 180:
        issues.append("longitude_out_of_range")
    if not record.get("source"):
        issues.append("missing_source")
    if not record.get("model_name"):
        issues.append("missing_model_name")
    return issues


def build_report(records: list[dict]) -> dict:
    issue_counter: Counter[str] = Counter()
    valid_records = []
    confidence_values = []

    for record in records:
        issues = validate_record(record)
        issue_counter.update(issues)
        if not issues:
            valid_records.append(record)
        if record.get("confiance") is not None:
            confidence_values.append(float(record["confiance"]))

    source_counts = Counter(record.get("source", "missing") for record in records)
    model_counts = Counter(record.get("model_name", "missing") for record in records)
    low_confidence_count = sum(1 for value in confidence_values if value < 0.65)
    quality_score = 100.0 if not records else round((len(valid_records) / len(records)) * 100, 2)

    status = "pass"
    if issue_counter or quality_score < 95 or low_confidence_count:
        status = "warning"
    if quality_score < 80:
        status = "fail"

    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "component": "data_quality_and_drift_monitoring",
        "status": status,
        "records_analyzed": len(records),
        "valid_records": len(valid_records),
        "quality_score_percent": quality_score,
        "average_confidence": round(mean(confidence_values), 4) if confidence_values else None,
        "low_confidence_threshold": 0.65,
        "low_confidence_count": low_confidence_count,
        "source_distribution": dict(source_counts),
        "model_distribution": dict(model_counts),
        "issues": dict(issue_counter),
        "recommendation": recommendation(status, issue_counter, low_confidence_count),
    }


def recommendation(status: str, issues: Counter[str], low_confidence_count: int) -> str:
    if status == "pass":
        return "Data quality is acceptable for the current monitoring window."
    if issues:
        return "Investigate invalid GPS, missing source, or missing model fields before using these detections operationally."
    if low_confidence_count:
        return "Review low-confidence detections; they may indicate model drift, new waste types, or poor image quality."
    return "Review the monitoring report before operational use."


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a data quality and drift report for waste detections.")
    parser.add_argument("--app-db", type=Path, default=DEFAULT_APP_DB)
    parser.add_argument("--patrol-db", type=Path, default=DEFAULT_PATROL_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--demo", action="store_true", help="Use built-in sample records so the command is always demoable.")
    args = parser.parse_args()

    if args.demo:
        records = demo_records()
    else:
        records = read_app_detections(args.app_db) + read_patrol_detections(args.patrol_db)

    report = build_report(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Data quality status: {report['status']}")
    print(f"Records analyzed: {report['records_analyzed']}")
    print(f"Quality score: {report['quality_score_percent']}%")
    print(f"Low confidence count: {report['low_confidence_count']}")
    print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()
