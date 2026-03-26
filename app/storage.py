from __future__ import annotations

import csv
import json
from pathlib import Path

from app.config import DATA_DIR, OUTPUT_DIR, RECORDS_DIR, SETTINGS_FILE
from app.models import Record


RECORD_HEADERS = ["id", "created_at", "stage", "payload", "source_ids"]


def _sanitize_stage(stage: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in stage).strip("_") or "unknown"


def _stage_records_file(stage: str) -> Path:
    return RECORDS_DIR / f"{_sanitize_stage(stage)}.csv"


def _ensure_records_file(path: Path) -> None:
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RECORD_HEADERS)
            writer.writeheader()


def append_record(record: Record) -> None:
    target = _stage_records_file(record.stage)
    _ensure_records_file(target)
    with target.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RECORD_HEADERS)
        writer.writerow(record.__dict__)


def read_records(limit: int = 500) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(RECORDS_DIR.glob("*.csv")):
        with path.open("r", newline="", encoding="utf-8") as f:
            rows.extend(list(csv.DictReader(f)))
    rows.sort(key=lambda row: row.get("created_at", ""), reverse=True)
    return rows[:limit]


def save_named_settings(name: str, payload: dict) -> None:
    data = load_saved_settings()
    data[name] = payload
    SETTINGS_FILE.write_text(json.dumps(data, indent=2))


def load_saved_settings() -> dict:
    if not SETTINGS_FILE.exists():
        return {}
    return json.loads(SETTINGS_FILE.read_text())


def list_csv_files() -> list[dict]:
    roots = [DATA_DIR, OUTPUT_DIR]
    files: list[dict] = []
    for root in roots:
        if not root.exists():
            continue
        for csv_path in sorted(root.rglob("*.csv")):
            relative = csv_path.relative_to(root).as_posix()
            files.append(
                {
                    "key": f"{root.name}/{relative}",
                    "name": relative,
                    "location": root.name,
                }
            )
    return files


def resolve_csv_file(file_key: str) -> Path:
    safe_key = (file_key or "").strip()
    if not safe_key or "/" not in safe_key:
        raise ValueError("invalid_file_key")

    location, relative_path = safe_key.split("/", 1)
    if location not in {"data", "output"}:
        raise ValueError("invalid_location")
    if not relative_path.endswith(".csv"):
        raise ValueError("invalid_extension")
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("invalid_filename")

    root = DATA_DIR if location == "data" else OUTPUT_DIR
    target = (root / relative).resolve()
    if root.resolve() not in target.parents and target != root.resolve():
        raise ValueError("invalid_filename")
    if not target.exists() or not target.is_file():
        raise ValueError("file_not_found")
    return target


def read_csv_table(file_key: str) -> dict:
    target = resolve_csv_file(file_key)
    with target.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        headers = list(rows[0].keys()) if rows else []
        if not headers:
            f.seek(0)
            reader = csv.reader(f)
            headers = next(reader, [])
    return {"headers": headers, "rows": rows}


def update_csv_row(file_key: str, row_index: int, row_payload: dict) -> None:
    target = resolve_csv_file(file_key)
    with target.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)

    if row_index < 0 or row_index >= len(rows):
        raise IndexError("row_not_found")

    updated_row = {}
    for header in headers:
        updated_row[header] = str(row_payload.get(header, ""))
    rows[row_index] = updated_row

    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def delete_csv_row(file_key: str, row_index: int) -> None:
    target = resolve_csv_file(file_key)
    with target.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)

    if row_index < 0 or row_index >= len(rows):
        raise IndexError("row_not_found")

    del rows[row_index]

    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
