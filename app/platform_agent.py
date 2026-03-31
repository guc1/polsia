from __future__ import annotations

import json

from sqlalchemy import select

from app.models import Agent, GenerationFlow, Persona, PromptBlock, StageDefinition, StageKey
from app.openrouter_client import OpenRouterClient

PLATFORM_SYSTEM_PROMPT = """
You are the platform-level agent for a modular AI content platform.
You must reason about stages, flows, personas, agents, and prompt blocks.
Respond as JSON with keys: intent_summary, suggested_flow, suggested_prompt_blocks, suggested_agent_layout, notes.
Keep recommendations implementation-ready.
""".strip()


async def propose_configuration(session, user_intent: str) -> dict:
    stages = [{"key": s.key, "name": s.name, "part": s.part} for s in session.scalars(select(StageDefinition)).all()]
    agents = [{"id": a.id, "name": a.name, "role": a.role} for a in session.scalars(select(Agent)).all()]
    personas = [{"id": p.id, "name": p.name} for p in session.scalars(select(Persona)).all()]
    blocks = [{"id": b.id, "name": b.name, "scope": b.scope} for b in session.scalars(select(PromptBlock)).all()]
    flows = [{"id": f.id, "name": f.name, "execution_mode": f.execution_mode} for f in session.scalars(select(GenerationFlow)).all()]

    user_prompt = json.dumps(
        {
            "user_intent": user_intent,
            "available_stages": stages,
            "available_agents": agents,
            "available_personas": personas,
            "available_prompt_blocks": blocks,
            "existing_flows": flows,
            "constraints": "Part A stages are implemented now. Part B stages are future placeholders.",
        },
        indent=2,
    )
    client = OpenRouterClient()
    response = await client.complete(model="openai/gpt-4o-mini", system_prompt=PLATFORM_SYSTEM_PROMPT, user_prompt=user_prompt)
    if response.startswith("[DRY RUN]"):
        return {
            "intent_summary": user_intent,
            "suggested_flow": {
                "name": "Work+School Part A Flow",
                "description": "Generates elements, formats, and headlines.",
                "stages": [StageKey.ELEMENTS.value, StageKey.HEADLINE_FORMATS.value, StageKey.HEADLINES.value],
                "execution_mode": "sequential",
            },
            "suggested_prompt_blocks": ["School/Work focus", "Quality bar"],
            "suggested_agent_layout": {"elements": ["creator_alpha", "creator_beta", "reviewer", "decider"]},
            "notes": ["Dry run mode; set OPENROUTER_API_KEY for richer suggestions."],
        }
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {"intent_summary": user_intent, "notes": [response]}
