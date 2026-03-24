from __future__ import annotations

import csv
import json
from pathlib import Path

from app.config import RECORDS_FILE, SETTINGS_FILE
from app.models import Record


RECORD_HEADERS = ["id", "created_at", "stage", "payload", "source_ids"]


def ensure_records_file() -> None:
    if not RECORDS_FILE.exists():
        with RECORDS_FILE.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RECORD_HEADERS)
            writer.writeheader()


def append_record(record: Record) -> None:
    ensure_records_file()
    with RECORDS_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RECORD_HEADERS)
        writer.writerow(record.__dict__)


def read_records(limit: int = 500) -> list[dict]:
    ensure_records_file()
    with RECORDS_FILE.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-limit:][::-1]


def save_named_settings(name: str, payload: dict) -> None:
    data = load_saved_settings()
    data[name] = payload
    SETTINGS_FILE.write_text(json.dumps(data, indent=2))


def load_saved_settings() -> dict:
    if not SETTINGS_FILE.exists():
        return {}
    return json.loads(SETTINGS_FILE.read_text())
