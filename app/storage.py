from __future__ import annotations

import csv
import json
from pathlib import Path
from uuid import uuid4

from app.config import DATA_DIR, OUTPUT_DIR, RECORDS_FILE, SETTINGS_FILE
from app.models import Record, Stage


RECORD_HEADERS = ["id", "created_at", "stage", "payload", "source_ids"]

STAGE_CSV_SCHEMAS: dict[str, list[str]] = {
    Stage.ELEMENT_GENERATION.value: ["id", "element_type", "name", "description", "reasoning_for_choosing", "created_at"],
    Stage.STORY_FORMAT_GENERATION.value: ["id", "name", "description", "reasoning_for_choosing", "created_at"],
    Stage.HEADLINE_GENERATION.value: ["id", "headline", "reasoning_for_choosing", "created_at"],
    Stage.HEADLINE_SELECTION.value: ["id", "headline", "reasoning_for_choosing", "created_at"],
    Stage.HOOK_GENERATION.value: ["id", "hook", "reasoning_for_choosing", "created_at"],
    Stage.STORY_PLANNING.value: ["id", "name", "description", "reasoning_for_choosing", "created_at"],
    Stage.STORY_WRITING.value: ["id", "title", "story", "reasoning_for_choosing", "created_at"],
    Stage.SHORT_SCRIPT_WRITING.value: ["id", "title", "script", "reasoning_for_choosing", "created_at"],
    Stage.VIDEO_HEADLINE_GENERATION.value: ["id", "video_headline", "reasoning_for_choosing", "created_at"],
    Stage.CAPTION_GENERATION.value: ["id", "caption", "reasoning_for_choosing", "created_at"],
}


def stage_csv_filename(stage_key: str) -> str:
    return f"{stage_key}.csv"


def stage_csv_path(stage_key: str) -> Path:
    if stage_key not in STAGE_CSV_SCHEMAS:
        raise ValueError("invalid_stage")
    return DATA_DIR / stage_csv_filename(stage_key)


def ensure_stage_csv(stage_key: str) -> None:
    target = stage_csv_path(stage_key)
    if target.exists():
        return
    headers = STAGE_CSV_SCHEMAS[stage_key]
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()


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


def list_csv_files() -> list[dict]:
    for stage_key in STAGE_CSV_SCHEMAS:
        ensure_stage_csv(stage_key)

    roots = [DATA_DIR, OUTPUT_DIR]
    files: list[dict] = []
    for root in roots:
        if not root.exists():
            continue
        for csv_path in sorted(root.glob("*.csv")):
            files.append(
                {
                    "key": f"{root.name}/{csv_path.name}",
                    "name": csv_path.name,
                    "location": root.name,
                }
            )
    return files


def resolve_csv_file(file_key: str) -> Path:
    safe_key = (file_key or "").strip()
    if not safe_key or "/" not in safe_key:
        raise ValueError("invalid_file_key")

    location, filename = safe_key.split("/", 1)
    if location not in {"data", "output"}:
        raise ValueError("invalid_location")
    if not filename.endswith(".csv"):
        raise ValueError("invalid_extension")
    if "/" in filename or "\\" in filename:
        raise ValueError("invalid_filename")

    root = DATA_DIR if location == "data" else OUTPUT_DIR
    target = root / filename
    if not target.exists():
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


def append_stage_rows(stage_key: str, rows: list[dict]) -> None:
    ensure_stage_csv(stage_key)
    headers = STAGE_CSV_SCHEMAS[stage_key]
    target = stage_csv_path(stage_key)

    normalized = []
    for row in rows:
        normalized.append({
            header: str(row.get(header, "")) if row.get(header) is not None else ""
            for header in headers
        })
        if not normalized[-1].get("id"):
            normalized[-1]["id"] = str(uuid4())

    with target.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writerows(normalized)
