from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from uuid import uuid4

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from app.agents import load_agents, save_agents
from app.models import DEFAULT_STAGE_ORDER, RunConfig, RunState, Stage
from app.pipeline import Pipeline
from app.storage import load_saved_settings, read_records, save_named_settings

load_dotenv()
app = Flask(__name__)

RUNS: dict[str, RunState] = {}
RUN_EVENTS: dict[str, list[dict]] = {}


def add_event(event):
    RUN_EVENTS.setdefault(event.run_id, []).append(asdict(event))


def parse_run_config(payload: dict) -> RunConfig:
    selected_raw = payload.get("selected_stages") or [s.value for s in DEFAULT_STAGE_ORDER]
    selected = [Stage(s) for s in selected_raw]
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000, debug=True)
