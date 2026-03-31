from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import asdict

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request
from sqlalchemy import select

from app.models import (
    Agent,
    ConversationMessage,
    FlowStage,
    GenerationFlow,
    Headline,
    HeadlineFormat,
    Persona,
    PromptBlock,
    PromptBlockStage,
    Run,
    RunEvent,
    StageDefinition,
    StageKey,
    Element,
)
from app.pipeline import ELEMENT_GROUPS, run_pipeline_in_thread
from app.platform_agent import propose_configuration
from app.storage import init_db, session_scope

load_dotenv()
app = Flask(__name__)
init_db()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/bootstrap")
def bootstrap():
    with session_scope() as session:
        flows = session.scalars(select(GenerationFlow).order_by(GenerationFlow.updated_at.desc())).all()
        recent_runs = session.scalars(select(Run).order_by(Run.started_at.desc()).limit(20)).all()
        return jsonify(
            {
                "stages": [
                    {"key": s.key, "name": s.name, "part": s.part, "description": s.description}
                    for s in session.scalars(select(StageDefinition)).all()
                ],
                "flows": [
                    {"id": f.id, "name": f.name, "description": f.description, "execution_mode": f.execution_mode}
                    for f in flows
                ],
                "recent_runs": [
                    {"id": r.id, "flow_id": r.flow_id, "status": r.status, "started_at": r.started_at.isoformat()}
                    for r in recent_runs
                ],
            }
        )


@app.get("/api/personas")
def list_personas():
    with session_scope() as session:
        return jsonify([
            {"id": p.id, "name": p.name, "description": p.description, "persona_text": p.persona_text}
            for p in session.scalars(select(Persona)).all()
        ])


@app.post("/api/personas")
def save_persona():
    payload = request.json or {}
    with session_scope() as session:
        pid = payload.get("id")
        persona = session.get(Persona, pid) if pid else Persona()
        persona.name = payload.get("name", persona.name)
        persona.description = payload.get("description", "")
        persona.persona_text = payload.get("persona_text", "")
        session.add(persona)
        session.commit()
        return jsonify({"ok": True, "id": persona.id})


@app.get("/api/agents")
def list_agents():
    with session_scope() as session:
        return jsonify([
            {
                "id": a.id,
                "name": a.name,
                "role": a.role,
                "is_active": a.is_active,
                "model": a.model,
                "persona_id": a.persona_id,
                "override_prompt": a.override_prompt,
            }
            for a in session.scalars(select(Agent)).all()
        ])


@app.post("/api/agents")
def save_agent():
    payload = request.json or {}
    with session_scope() as session:
        aid = payload.get("id")
        agent = session.get(Agent, aid) if aid else Agent()
        for field in ["name", "role", "model", "override_prompt", "persona_id"]:
            if field in payload:
                setattr(agent, field, payload[field])
        if "is_active" in payload:
            agent.is_active = bool(payload["is_active"])
        session.add(agent)
        session.commit()
        return jsonify({"ok": True, "id": agent.id})


@app.get("/api/prompt-blocks")
def list_prompt_blocks():
    with session_scope() as session:
        refs = session.scalars(select(PromptBlockStage)).all()
        stage_map = {}
        for ref in refs:
            stage_map.setdefault(ref.prompt_block_id, []).append(ref.stage_key)
        return jsonify([
            {
                "id": b.id,
                "name": b.name,
                "description_for_agent": b.description_for_agent,
                "prompt_text": b.prompt_text,
                "scope": b.scope,
                "is_active": b.is_active,
                "version": b.version,
                "stage_keys": stage_map.get(b.id, []),
            }
            for b in session.scalars(select(PromptBlock)).all()
        ])


@app.post("/api/prompt-blocks")
def save_prompt_block():
    payload = request.json or {}
    with session_scope() as session:
        bid = payload.get("id")
        block = session.get(PromptBlock, bid) if bid else PromptBlock()
        block.name = payload.get("name", block.name)
        block.description_for_agent = payload.get("description_for_agent", "")
        block.prompt_text = payload.get("prompt_text", "")
        block.scope = payload.get("scope", "shared")
        block.is_active = bool(payload.get("is_active", True))
        if bid:
            block.version += 1
        session.add(block)
        session.flush()
        session.query(PromptBlockStage).filter(PromptBlockStage.prompt_block_id == block.id).delete()
        for stage_key in payload.get("stage_keys", []):
            session.add(PromptBlockStage(prompt_block_id=block.id, stage_key=stage_key))
        session.commit()
        return jsonify({"ok": True, "id": block.id})


@app.post("/api/prompt-blocks/<int:block_id>/duplicate")
def duplicate_prompt_block(block_id: int):
    with session_scope() as session:
        block = session.get(PromptBlock, block_id)
        if not block:
            return jsonify({"error": "not_found"}), 404
        clone = PromptBlock(
            name=f"{block.name} (copy)",
            description_for_agent=block.description_for_agent,
            prompt_text=block.prompt_text,
            scope=block.scope,
            is_active=block.is_active,
        )
        session.add(clone)
        session.commit()
        return jsonify({"ok": True, "id": clone.id})


@app.get("/api/flows")
def list_flows():
    with session_scope() as session:
        flows = session.scalars(select(GenerationFlow)).all()
        data = []
        for flow in flows:
            data.append(
                {
                    "id": flow.id,
                    "name": flow.name,
                    "description": flow.description,
                    "execution_mode": flow.execution_mode,
                    "default_settings": flow.default_settings,
                    "context_rules": flow.context_rules,
                    "stages": [
                        {
                            "id": s.id,
                            "stage_key": s.stage_key,
                            "stage_order": s.stage_order,
                            "enabled": s.enabled,
                            "stage_params": s.stage_params,
                            "agent_ids": s.agent_ids,
                            "prompt_block_ids": s.prompt_block_ids,
                            "context_sources": s.context_sources,
                        }
                        for s in sorted(flow.stages, key=lambda x: x.stage_order)
                    ],
                }
            )
        return jsonify(data)


@app.post("/api/flows")
def save_flow():
    payload = request.json or {}
    with session_scope() as session:
        fid = payload.get("id")
        flow = session.get(GenerationFlow, fid) if fid else GenerationFlow()
        flow.name = payload.get("name", flow.name)
        flow.description = payload.get("description", "")
        flow.execution_mode = payload.get("execution_mode", "sequential")
        flow.default_settings = payload.get("default_settings", {})
        flow.context_rules = payload.get("context_rules", {})
        session.add(flow)
        session.flush()
        session.query(FlowStage).filter(FlowStage.flow_id == flow.id).delete()
        for stage in payload.get("stages", []):
            session.add(
                FlowStage(
                    flow_id=flow.id,
                    stage_key=stage["stage_key"],
                    stage_order=stage.get("stage_order", 0),
                    enabled=bool(stage.get("enabled", True)),
                    stage_params=stage.get("stage_params", {}),
                    agent_ids=stage.get("agent_ids", []),
                    prompt_block_ids=stage.get("prompt_block_ids", []),
                    context_sources=stage.get("context_sources", []),
                )
            )
        session.commit()
        return jsonify({"ok": True, "id": flow.id})


@app.post("/api/flows/<int:flow_id>/duplicate")
def duplicate_flow(flow_id: int):
    with session_scope() as session:
        flow = session.get(GenerationFlow, flow_id)
        if not flow:
            return jsonify({"error": "not_found"}), 404
        new_flow = GenerationFlow(
            name=f"{flow.name} (copy)",
            description=flow.description,
            execution_mode=flow.execution_mode,
            default_settings=flow.default_settings,
            context_rules=flow.context_rules,
        )
        session.add(new_flow)
        session.flush()
        for s in flow.stages:
            session.add(
                FlowStage(
                    flow_id=new_flow.id,
                    stage_key=s.stage_key,
                    stage_order=s.stage_order,
                    enabled=s.enabled,
                    stage_params=s.stage_params,
                    agent_ids=s.agent_ids,
                    prompt_block_ids=s.prompt_block_ids,
                    context_sources=s.context_sources,
                )
            )
        session.commit()
        return jsonify({"ok": True, "id": new_flow.id})


@app.delete("/api/flows/<int:flow_id>")
def delete_flow(flow_id: int):
    with session_scope() as session:
        flow = session.get(GenerationFlow, flow_id)
        if not flow:
            return jsonify({"error": "not_found"}), 404
        session.delete(flow)
        session.commit()
        return jsonify({"ok": True})


@app.post("/api/runs")
def start_run():
    payload = request.json or {}
    with session_scope() as session:
        run = Run(flow_id=payload.get("flow_id"), config=payload.get("config", {}), status="queued")
        session.add(run)
        session.commit()
        rid = run.id
    thread = threading.Thread(target=run_pipeline_in_thread, args=(session_scope, rid), daemon=True)
    thread.start()
    return jsonify({"run_id": rid})


@app.get("/api/runs/<int:run_id>")
def run_detail(run_id: int):
    with session_scope() as session:
        run = session.get(Run, run_id)
        if not run:
            return jsonify({"error": "not_found"}), 404
        events = session.scalars(select(RunEvent).where(RunEvent.run_id == run_id).order_by(RunEvent.id)).all()
        chats = session.scalars(select(ConversationMessage).where(ConversationMessage.run_id == run_id).order_by(ConversationMessage.id)).all()
        return jsonify(
            {
                "run": {"id": run.id, "status": run.status, "flow_id": run.flow_id},
                "events": [
                    {
                        "id": e.id,
                        "ts": e.ts.isoformat(),
                        "event_type": e.event_type,
                        "stage_key": e.stage_key,
                        "agent_name": e.agent_name,
                        "message": e.message,
                    }
                    for e in events
                ],
                "conversation": [
                    {
                        "id": c.id,
                        "ts": c.ts.isoformat(),
                        "stage_key": c.stage_key,
                        "agent_name": c.agent_name,
                        "role": c.role,
                        "content": c.content,
                    }
                    for c in chats
                ],
            }
        )


@app.get("/api/database/content")
def database_content():
    with session_scope() as session:
        return jsonify(
            {
                "elements": [
                    {
                        "id": e.id,
                        "element_type": e.element_type,
                        "name": e.name,
                        "description": e.description,
                        "reasoning_for_choosing": e.reasoning_for_choosing,
                        "created_at": e.created_at.isoformat(),
                    }
                    for e in session.scalars(select(Element).order_by(Element.id.desc()).limit(100)).all()
                ],
                "headline_formats": [
                    {
                        "id": f.id,
                        "name": f.name,
                        "blueprint": f.blueprint,
                        "reasoning_for_choosing": f.reasoning_for_choosing,
                        "created_at": f.created_at.isoformat(),
                    }
                    for f in session.scalars(select(HeadlineFormat).order_by(HeadlineFormat.id.desc()).limit(100)).all()
                ],
                "headlines": [
                    {
                        "id": h.id,
                        "headline": h.headline,
                        "reasoning_for_choosing": h.reasoning_for_choosing,
                        "score": h.score,
                        "created_at": h.created_at.isoformat(),
                    }
                    for h in session.scalars(select(Headline).order_by(Headline.id.desc()).limit(100)).all()
                ],
            }
        )


@app.post("/api/platform-agent/propose")
def platform_propose():
    payload = request.json or {}
    intent = payload.get("intent", "")

    async def _run():
        with session_scope() as session:
            return await propose_configuration(session, intent)

    return jsonify(asyncio.run(_run()))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000, debug=True)
