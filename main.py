from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from app.agents import load_agents, save_agents
from app.models import DEFAULT_STAGE_ORDER, RunConfig, RunLogEvent, RunState, Stage
from app.pipeline import LOOP_DEFINITIONS, Pipeline
from app.storage import (
    STAGE_CSV_SCHEMAS,
    append_stage_rows,
    delete_csv_row,
    list_csv_files,
    load_saved_settings,
    read_csv_table,
    read_records,
    save_named_settings,
    update_csv_row,
)

load_dotenv()
app = Flask(__name__)

RUNS: dict[str, RunState] = {}
RUN_EVENTS: dict[str, list[dict]] = {}


PROMPT_STAGE_FILES = {
    "element_generation": "element_generation.txt",
    "story_format_generation": "story_format_generation.txt",
    "headline_generation": "headline_generation.txt",
    "headline_selection": "headline_generation.txt",
    "hook_generation": "hook_generation.txt",
    "story_planning": "story_planning.txt",
    "story_writing": "story_writing.txt",
    "short_script_writing": "short_script_writing.txt",
    "video_headline_generation": "video_headline_generation.txt",
    "caption_generation": "caption_generation.txt",
}


def prompt_paths_for_stage(stage: str) -> dict[str, Path]:
    stage_file = PROMPT_STAGE_FILES.get(stage)
    if not stage_file:
        raise ValueError("invalid_stage")
    return {
        "system_prompt_template": Path("prompts/shared/base_system_prompt.txt"),
        "task_prompt": Path("prompts/stages") / stage_file,
        "feedback_prompt": Path("prompts/shared/feedback_prompt.txt"),
        "decider_prompt": Path("prompts/shared/decider_prompt.txt"),
        "board_prompt": Path("prompts/shared/board_prompt.txt"),
    }



STAGE_LABELS = {
    "element_generation": "Elements",
    "story_format_generation": "Headline Format",
    "headline_generation": "Headline",
    "headline_selection": "Headline Selection",
    "hook_generation": "Hook",
    "story_planning": "Story Planning",
    "story_writing": "Story Writing",
    "short_script_writing": "Short Script",
    "video_headline_generation": "Video Headline",
    "caption_generation": "Caption",
}


def stage_catalog() -> list[dict]:
    items = []
    for stage in DEFAULT_STAGE_ORDER:
        loop = LOOP_DEFINITIONS[stage]
        items.append(
            {
                "key": stage.value,
                "label": STAGE_LABELS.get(stage.value, stage.value),
                "loop_explanation": (
                    f"Creators: {', '.join(loop.creator_agents) or 'none'} → "
                    f"Reviewers: {', '.join(loop.reviewer_agents) or 'none'} → "
                    f"Rewrite: {'yes' if loop.has_rewrite_round else 'no'} → "
                    f"Decider: {'yes' if loop.has_decider else 'no'}"
                ),
                "agents": sorted(set(loop.creator_agents + loop.reviewer_agents + (["decider"] if loop.has_decider else []))),
                "csv_key": f"data/{stage.value}.csv",
                "csv_headers": STAGE_CSV_SCHEMAS.get(stage.value, []),
            }
        )
    return items


def add_event(event):
    RUN_EVENTS.setdefault(event.run_id, []).append(asdict(event))


def parse_run_config(payload: dict) -> RunConfig:
    selected_raw = payload.get("selected_stages") or [s.value for s in DEFAULT_STAGE_ORDER]
    alias = {"elements":"element_generation","format_types":"story_format_generation","headlines":"headline_generation","hook":"hook_generation","story_plan":"story_planning","story":"story_writing","script":"short_script_writing","video_text":"video_headline_generation"}
    selected = [Stage(alias.get(s, s)) for s in selected_raw]
    selected_element_types = payload.get("selected_element_types") or [
        "main_characters",
        "side_characters",
        "locations",
        "situations",
        "protagonist_emotions",
        "audience_emotions",
        "absurd_situations",
        "narrator_locations",
    ]
    return RunConfig(
        selected_stages=selected,
        mode=payload.get("mode", "sequential"),
        model_map=payload.get("model_map", {}),
        agent_model_map=payload.get("agent_model_map", {}),
        temperature=float(payload.get("temperature", 0.7)),
        max_context_chars=int(payload.get("max_context_chars", 12000)),
        target_minutes=int(payload.get("target_minutes", 2)),
        target_parts=int(payload.get("target_parts", 3)),
        custom_instruction=payload.get("custom_instruction", ""),
        enable_data_specialist=bool(payload.get("enable_data_specialist", True)),
        enable_format_context=bool(payload.get("enable_format_context", True)),
        output_count=int(payload.get("output_count", 3)),
        selected_element_types=selected_element_types,
        context_selection=payload.get("context_selection", {}),
        existing_elements=payload.get("existing_elements", ""),
        include_existing_elements=bool(payload.get("include_existing_elements", False)),
    )


def run_in_background(run_id: str):
    pipeline = Pipeline(add_event)
    run = RUNS[run_id]

    async def _runner():
        updated = await pipeline.execute(run)
        RUNS[run_id] = updated

    asyncio.run(_runner())


@app.route("/")
def index():
    return render_template("index.html", stage_values=[s.value for s in DEFAULT_STAGE_ORDER])


@app.get("/api/bootstrap")
def bootstrap():
    return jsonify(
        {
            "agents": load_agents(),
            "records": read_records(200),
            "saved_settings": load_saved_settings(),
            "runs": [asdict(r) for r in RUNS.values()],
            "stage_order": [s.value for s in DEFAULT_STAGE_ORDER],
            "stage_catalog": stage_catalog(),
        }
    )


@app.post("/api/run")
def start_run():
    payload = request.json or {}
    cfg = parse_run_config(payload)
    run_id = str(uuid4())
    run = RunState(run_id=run_id, config=cfg)
    RUNS[run_id] = run
    RUN_EVENTS[run_id] = []
    thread = threading.Thread(target=run_in_background, args=(run_id,), daemon=True)
    thread.start()
    return jsonify({"run_id": run_id})


@app.get("/api/runs")
def list_runs():
    return jsonify([asdict(r) for r in sorted(RUNS.values(), key=lambda x: x.created_at, reverse=True)])


@app.get("/api/runs/<run_id>")
def get_run(run_id: str):
    run = RUNS.get(run_id)
    if not run:
        return jsonify({"error": "not_found"}), 404
    return jsonify({"run": asdict(run), "events": RUN_EVENTS.get(run_id, [])})


@app.get("/api/agents")
def get_agents():
    return jsonify(load_agents())


@app.post("/api/agents")
def set_agents():
    payload = request.json or {}
    save_agents(payload)
    return jsonify({"ok": True})


@app.post("/api/settings/<name>")
def save_setting(name: str):
    payload = request.json or {}
    save_named_settings(name, payload)
    return jsonify({"ok": True})


@app.get("/api/csv/files")
def get_csv_files():
    return jsonify({"files": list_csv_files()})


@app.get("/api/csv/table")
def get_csv_table():
    file_key = request.args.get("file", "")
    try:
        table = read_csv_table(file_key)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(table)


@app.put("/api/csv/table/row")
def put_csv_row():
    payload = request.json or {}
    file_key = payload.get("file", "")
    row_index = int(payload.get("row_index", -1))
    row = payload.get("row", {}) or {}
    try:
        update_csv_row(file_key, row_index, row)
        updated = read_csv_table(file_key)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except IndexError as e:
        return jsonify({"error": str(e)}), 404
    return jsonify({"ok": True, **updated})


@app.delete("/api/csv/table/row")
def remove_csv_row():
    payload = request.json or {}
    file_key = payload.get("file", "")
    row_index = int(payload.get("row_index", -1))
    try:
        delete_csv_row(file_key, row_index)
        updated = read_csv_table(file_key)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except IndexError as e:
        return jsonify({"error": str(e)}), 404
    return jsonify({"ok": True, **updated})


@app.get("/api/prompts/<stage>")
def get_stage_prompts(stage: str):
    try:
        prompt_files = prompt_paths_for_stage(stage)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    payload = {}
    for key, rel_path in prompt_files.items():
        full = Path(app.root_path).parent / rel_path
        payload[key] = full.read_text(encoding="utf-8") if full.exists() else ""
    return jsonify(payload)


@app.post("/api/prompts/<stage>")
def save_stage_prompts(stage: str):
    body = request.json or {}
    try:
        prompt_files = prompt_paths_for_stage(stage)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    for key, rel_path in prompt_files.items():
        if key not in body:
            continue
        full = Path(app.root_path).parent / rel_path
        full.write_text(str(body[key]), encoding="utf-8")
    return jsonify({"ok": True})


@app.post("/api/runs/<run_id>/db-update")
def db_update(run_id: str):
    payload = request.json or {}
    stage = payload.get("stage", "")
    action = payload.get("action", "confirm")
    rows = payload.get("rows", []) or []

    run = RUNS.get(run_id)
    if not run:
        return jsonify({"error": "not_found"}), 404
    if stage not in run.pending_updates:
        return jsonify({"error": "stage_not_pending"}), 400

    if action in {"confirm", "edit_submit"}:
        append_stage_rows(stage, rows)
        run.pending_updates[stage]["status"] = "saved"
        run.pending_updates[stage]["rows"] = rows
        add_event(RunLogEvent(run_id=run_id, stage=stage, agent="storage", role="save_update", message=f"saved rows={len(rows)} to data/{stage}.csv"))
        return jsonify({"ok": True, "status": "saved"})

    if action == "cancel":
        run.pending_updates[stage]["status"] = "rejected"
        add_event(RunLogEvent(run_id=run_id, stage=stage, agent="storage", role="save_update", message="pending update rejected"))
        return jsonify({"ok": True, "status": "rejected"})

    return jsonify({"error": "invalid_action"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000, debug=True)
