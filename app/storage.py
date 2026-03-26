from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.config import DATA_DIR, OUTPUT_DIR, RECORDS_DIR, SETTINGS_FILE
from app.models import Record


RECORD_HEADERS = ["id", "created_at", "stage", "payload", "source_ids"]
CORE_STAGE_HEADERS = [
    "row_id",
    "run_id",
    "stage_name",
    "stage_step",
    "item_id",
    "parent_item_id",
    "source_item_ids",
    "context_item_ids",
    "created_at",
    "status",
    "selected_for_next_stage",
    "quality_score",
    "created_by_agent",
    "workflow_round",
    "custom_instruction",
    "output_text",
    "output_summary",
    "notes",
]
STAGE_FILE_MAP = {
    "element_generation": "elements.csv",
    "story_format_generation": "story_formats.csv",
    "headline_generation": "headlines.csv",
    "headline_board_review": "headline_board_votes.csv",
    "headline_selection": "headline_selections.csv",
    "hook_generation": "hooks.csv",
    "story_planning": "story_plans.csv",
    "story_writing": "stories.csv",
    "short_script_writing": "short_scripts.csv",
    "video_headline_generation": "video_headlines.csv",
    "caption_generation": "captions.csv",
}
STAGE_EXTRA_HEADERS = {
    "element_generation": ["element_group", "specificity_level", "novelty_note", "believability_note", "why_selected", "proposed_by_agent", "final_decider_score"],
    "story_format_generation": ["format_title", "format_blueprint", "supports_plot_twist", "supports_multi_part", "why_it_works", "why_selected"],
    "headline_generation": ["headline_text", "storyteller_identity", "storyteller_location", "click_strength_score", "believability_score", "novelty_score", "used_elements_ids", "used_format_ids", "passed_board", "board_pass_count"],
    "hook_generation": ["headline_id", "initial_hook_line", "extended_hook", "spoiler_risk", "curiosity_score", "why_selected"],
    "story_planning": ["headline_id", "hook_id", "setup", "character_building", "escalation_steps", "twist_or_reveal", "payoff", "part_breaks", "retention_notes", "supports_multi_part"],
    "story_writing": ["headline_id", "hook_id", "story_plan_id", "story_title", "story_text", "writer_version", "merged_from_story_ids", "readability_notes", "retention_notes", "final_story_score"],
    "short_script_writing": ["story_id", "script_scope", "script_full_text", "part_number", "part_text", "opening_line", "closing_line", "approx_duration_seconds", "language_level", "humanization_notes", "retention_notes"],
    "video_headline_generation": ["script_id", "video_headline_text", "readability_score", "curiosity_score", "selected_final"],
    "caption_generation": ["script_id", "caption_text", "caption_style_type", "curiosity_score", "comment_trigger_score", "selected_final"],
}
FEEDBACK_HEADERS = ["row_id", "run_id", "stage_name", "target_item_id", "feedback_id", "reviewer_agent", "workflow_round", "overall_verdict", "main_strengths", "main_issues", "suggestions", "ranking_payload", "created_at"]
DECIDER_HEADERS = ["row_id", "run_id", "stage_name", "decider_output_id", "selected_base_ids", "rejected_ids", "final_item_id", "why_this_won", "why_others_lost", "confidence", "created_at"]
RUN_LOG_HEADERS = ["run_id", "started_at", "finished_at", "run_mode", "selected_stages", "included_context_sources", "custom_instruction", "status", "error_message", "total_items_generated", "total_tokens_if_available", "notes"]
CONTEXT_LINK_HEADERS = ["row_id", "run_id", "target_item_id", "source_item_id", "source_stage", "relation_type"]


def _sanitize_stage(stage: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in stage).strip("_") or "unknown"


def _stage_records_file(stage: str) -> Path:
    return RECORDS_DIR / f"{_sanitize_stage(stage)}.csv"


def _ensure_records_file(path: Path) -> None:
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RECORD_HEADERS)
            writer.writeheader()


def _ensure_csv(path: Path, headers: list[str]) -> None:
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()


def _append_csv_row(path: Path, headers: list[str], row: dict) -> None:
    _ensure_csv(path, headers)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writerow({key: row.get(key, "") for key in headers})


def _json(value) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def persist_stage_output(run_id: str, stage_name: str, output: dict, custom_instruction: str, summary: str = "", summary_notes: str = "") -> None:
    file_name = STAGE_FILE_MAP.get(stage_name, f"{_sanitize_stage(stage_name)}.csv")
    file_path = DATA_DIR / file_name
    headers = CORE_STAGE_HEADERS + STAGE_EXTRA_HEADERS.get(stage_name, [])
    final_output = output.get("final_decider_output") or output
    item_id = str(final_output.get("id") or final_output.get("selected_headline_id") or uuid4())
    selected = final_output.get("selected", True)
    base_row = {
        "row_id": str(uuid4()),
        "run_id": run_id,
        "stage_name": stage_name,
        "stage_step": "decider_final",
        "item_id": item_id,
        "parent_item_id": "",
        "source_item_ids": _json(final_output.get("source_ids", [])),
        "context_item_ids": _json(final_output.get("context_ids", [])),
        "created_at": _now_iso(),
        "status": "final",
        "selected_for_next_stage": int(bool(selected)),
        "quality_score": final_output.get("confidence", ""),
        "created_by_agent": "decider",
        "workflow_round": "final",
        "custom_instruction": custom_instruction or "",
        "output_text": _json(final_output),
        "output_summary": summary,
        "notes": summary_notes,
    }
    _append_csv_row(file_path, headers, base_row)

    for round_idx, round_name in enumerate(["feedback_round_1", "feedback_round_2"], start=1):
        feedback = output.get(round_name, {}) or {}
        for reviewer_agent, payload in feedback.items():
            _append_csv_row(
                DATA_DIR / "feedback_log.csv",
                FEEDBACK_HEADERS,
                {
                    "row_id": str(uuid4()),
                    "run_id": run_id,
                    "stage_name": stage_name,
                    "target_item_id": item_id,
                    "feedback_id": str(uuid4()),
                    "reviewer_agent": reviewer_agent,
                    "workflow_round": round_idx,
                    "overall_verdict": payload.get("verdict", ""),
                    "main_strengths": _json(payload.get("strengths", [])),
                    "main_issues": _json(payload.get("issues", [])),
                    "suggestions": _json(payload.get("feedback", payload.get("suggestions", []))),
                    "ranking_payload": _json(payload.get("ranking", payload.get("rankings", []))),
                    "created_at": _now_iso(),
                },
            )

    _append_csv_row(
        DATA_DIR / "decider_log.csv",
        DECIDER_HEADERS,
        {
            "row_id": str(uuid4()),
            "run_id": run_id,
            "stage_name": stage_name,
            "decider_output_id": str(uuid4()),
            "selected_base_ids": _json(final_output.get("selected_ids", final_output.get("selected_base_ids", []))),
            "rejected_ids": _json(final_output.get("rejected_ids", [])),
            "final_item_id": item_id,
            "why_this_won": final_output.get("reasoning", final_output.get("selection_reason", "")),
            "why_others_lost": _json(final_output.get("why_others_lost", "")),
            "confidence": final_output.get("confidence", ""),
            "created_at": _now_iso(),
        },
    )

    board = output.get("board", {}) or {}
    votes = board.get("votes", {}) or {}
    for board_member, vote_payload in votes.items():
        _append_csv_row(
            DATA_DIR / "headline_board_votes.csv",
            ["row_id", "run_id", "headline_id", "board_member_name", "vote", "rank", "explanation", "improvement_if_fail", "created_at"],
            {
                "row_id": str(uuid4()),
                "run_id": run_id,
                "headline_id": item_id,
                "board_member_name": board_member,
                "vote": vote_payload.get("vote", ""),
                "rank": _json(vote_payload.get("ranking", [])),
                "explanation": vote_payload.get("explanation", ""),
                "improvement_if_fail": vote_payload.get("improvement_if_fail", ""),
                "created_at": _now_iso(),
            },
        )

    source_ids = final_output.get("source_ids", []) or []
    context_ids = final_output.get("context_ids", []) or []
    for source_id in source_ids:
        _append_csv_row(
            DATA_DIR / "context_links.csv",
            CONTEXT_LINK_HEADERS,
            {
                "row_id": str(uuid4()),
                "run_id": run_id,
                "target_item_id": item_id,
                "source_item_id": source_id,
                "source_stage": "",
                "relation_type": "direct_parent",
            },
        )
    for context_id in context_ids:
        _append_csv_row(
            DATA_DIR / "context_links.csv",
            CONTEXT_LINK_HEADERS,
            {
                "row_id": str(uuid4()),
                "run_id": run_id,
                "target_item_id": item_id,
                "source_item_id": context_id,
                "source_stage": "",
                "relation_type": "included_context",
            },
        )


def append_run_log(row: dict) -> None:
    _append_csv_row(DATA_DIR / "run_log.csv", RUN_LOG_HEADERS, row)


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
