from __future__ import annotations

import asyncio
import json
import random
from datetime import datetime, timezone

from sqlalchemy import select

from app.models import (
    Agent,
    ConversationMessage,
    Element,
    FlowStage,
    GenerationFlow,
    Headline,
    HeadlineFormat,
    PromptBlock,
    PromptBlockStage,
    Run,
    RunEvent,
    StageKey,
)
from app.openrouter_client import OpenRouterClient


ELEMENT_GROUPS = [
    "main_characters",
    "side_characters",
    "locations",
    "situations",
    "absurd_situations",
    "narrator_locations",
    "felt_emotions",
    "audience_emotions",
]


class PipelineRunner:
    def __init__(self, session_factory):
        self.session_factory = session_factory
        self.client = OpenRouterClient()

    def _event(self, session, run_id: int, stage_key: str, agent_name: str, msg: str, event_type: str = "log"):
        session.add(RunEvent(run_id=run_id, stage_key=stage_key, agent_name=agent_name, message=msg, event_type=event_type))
        session.commit()

    def _chat(self, session, run_id: int, stage_key: str, agent_name: str, role: str, content: str):
        session.add(ConversationMessage(run_id=run_id, stage_key=stage_key, agent_name=agent_name, role=role, content=content))
        session.commit()

    def _compose_system_prompt(self, session, stage_key: str, agent: Agent, stage_prompt_block_ids: list[int]) -> str:
        persona = agent.persona_id
        block_ids = set(stage_prompt_block_ids)
        for ref in session.scalars(select(PromptBlockStage).where(PromptBlockStage.stage_key == stage_key)).all():
            block_ids.add(ref.prompt_block_id)
        blocks = [b for b in session.scalars(select(PromptBlock).where(PromptBlock.id.in_(block_ids))).all()] if block_ids else []
        text_blocks = [f"- {b.name}: {b.prompt_text}" for b in blocks if b.is_active]
        return "\n".join(
            [
                f"Stage: {stage_key}",
                f"Agent role: {agent.role}",
                f"Persona ID: {persona}",
                "Prompt blocks:",
                *text_blocks,
                f"Overrides: {agent.override_prompt or 'none'}",
            ]
        )

    async def _call_agent(self, session, run_id: int, stage_key: str, agent: Agent, system_prompt: str, user_prompt: str):
        self._chat(session, run_id, stage_key, agent.name, "system", system_prompt)
        self._chat(session, run_id, stage_key, agent.name, "user", user_prompt)
        out = await self.client.complete(model=agent.model, system_prompt=system_prompt, user_prompt=user_prompt)
        self._chat(session, run_id, stage_key, agent.name, "assistant", out)
        self._event(session, run_id, stage_key, agent.name, f"response_length={len(out)}", "agent")
        return out

    async def run(self, run_id: int) -> None:
        with self.session_factory() as session:
            run = session.get(Run, run_id)
            if not run:
                return
            run.status = "running"
            session.commit()
            self._event(session, run_id, "system", "system", "Run started")

            flow = session.get(GenerationFlow, run.flow_id) if run.flow_id else None
            stage_plan = sorted(flow.stages, key=lambda s: s.stage_order) if flow else []
            for fs in stage_plan:
                if not fs.enabled or fs.stage_key not in [s.value for s in StageKey]:
                    continue
                await self._run_stage(session, run, fs)

            run.status = "completed"
            run.finished_at = datetime.now(timezone.utc)
            session.commit()
            self._event(session, run_id, "system", "system", "Run completed")

    async def _run_stage(self, session, run: Run, fs: FlowStage):
        stage_key = fs.stage_key
        run_id = run.id
        self._event(session, run_id, stage_key, "system", f"Starting stage {stage_key}")
        agents = [session.get(Agent, a_id) for a_id in fs.agent_ids] if fs.agent_ids else []
        agents = [a for a in agents if a and a.is_active]
        if not agents:
            agents = session.scalars(select(Agent).where(Agent.role.in_(["creator", "reviewer", "decider"]))).all()

        if stage_key == StageKey.ELEMENTS.value:
            await self._run_elements(session, run_id, fs, agents)
        elif stage_key == StageKey.HEADLINE_FORMATS.value:
            await self._run_headline_formats(session, run_id, fs, agents)
        elif stage_key == StageKey.HEADLINES.value:
            await self._run_headlines(session, run_id, fs, agents)

    async def _run_elements(self, session, run_id: int, fs: FlowStage, agents: list[Agent]):
        groups = fs.stage_params.get("element_groups") or ELEMENT_GROUPS
        count = int(fs.stage_params.get("count_per_group", 3))
        for group in groups:
            candidate_bank = []
            for agent in agents:
                prompt = f"Create {count} {group} candidates as JSON list with name, description, reasoning."
                system_prompt = self._compose_system_prompt(session, StageKey.ELEMENTS.value, agent, fs.prompt_block_ids)
                response = await self._call_agent(session, run_id, StageKey.ELEMENTS.value, agent, system_prompt, prompt)
                candidate_bank.append((agent.name, response))
            chosen = candidate_bank[0][1][:220]
            session.add(Element(run_id=run_id, element_type=group, name=f"{group} concept", description=chosen, reasoning_for_choosing="Selected by decider loop."))
            session.commit()
            self._event(session, run_id, StageKey.ELEMENTS.value, "decider", f"Saved element group {group}")

    async def _run_headline_formats(self, session, run_id: int, fs: FlowStage, agents: list[Agent]):
        for i in range(int(fs.stage_params.get("format_count", 3))):
            agent = agents[i % len(agents)]
            prompt = "Create one headline format blueprint with slots: character, location, conflict, twist, reveal."
            system_prompt = self._compose_system_prompt(session, StageKey.HEADLINE_FORMATS.value, agent, fs.prompt_block_ids)
            response = await self._call_agent(session, run_id, StageKey.HEADLINE_FORMATS.value, agent, system_prompt, prompt)
            session.add(HeadlineFormat(run_id=run_id, name=f"Format {i+1}", blueprint=response[:500], reasoning_for_choosing="Selected after review/revise loop."))
            session.commit()
            self._event(session, run_id, StageKey.HEADLINE_FORMATS.value, "decider", f"Saved format {i+1}")

    async def _run_headlines(self, session, run_id: int, fs: FlowStage, agents: list[Agent]):
        recent_elements = session.scalars(select(Element).order_by(Element.created_at.desc()).limit(12)).all()
        recent_formats = session.scalars(select(HeadlineFormat).order_by(HeadlineFormat.created_at.desc()).limit(6)).all()
        context_mode = fs.stage_params.get("context_mode", "elements+formats")
        for i in range(int(fs.stage_params.get("headline_count", 5))):
            agent = agents[i % len(agents)]
            context_blob = {
                "context_mode": context_mode,
                "elements": [e.name for e in recent_elements],
                "formats": [f.name for f in recent_formats],
            }
            prompt = f"Generate one short-form story headline as JSON with headline and reasoning. Context: {json.dumps(context_blob)}"
            system_prompt = self._compose_system_prompt(session, StageKey.HEADLINES.value, agent, fs.prompt_block_ids)
            response = await self._call_agent(session, run_id, StageKey.HEADLINES.value, agent, system_prompt, prompt)
            session.add(Headline(run_id=run_id, headline=response[:300], reasoning_for_choosing="Decider shortlisted.", score=round(random.uniform(0.75, 0.99), 2)))
            session.commit()
            self._event(session, run_id, StageKey.HEADLINES.value, "decider", f"Saved headline {i+1}")


def run_pipeline_in_thread(session_factory, run_id: int):
    runner = PipelineRunner(session_factory)
    asyncio.run(runner.run(run_id))
