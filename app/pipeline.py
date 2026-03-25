from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from app.agents import load_agents
from app.config import OUTPUT_DIR
from app.models import DEFAULT_STAGE_ORDER, Record, RunConfig, RunLogEvent, RunState, Stage
from app.openrouter_client import OpenRouterClient
from app.storage import append_record

DEFAULT_MODEL = "openai/gpt-4o-mini"


class Pipeline:
    def __init__(self, event_cb: Callable[[RunLogEvent], None]):
        self.event_cb = event_cb
        self.client = OpenRouterClient()
        self.agents = load_agents()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._chat_threads: dict[str, list[dict[str, str]]] = {}

    def _log(self, run_id: str, stage: str, agent: str, role: str, message: str) -> None:
        self.event_cb(RunLogEvent(run_id=run_id, stage=stage, agent=agent, role=role, message=message))

    def _agent_system_prompt(self, agent_key: str, local_context: str, stage: str) -> str:
        a = self.agents[agent_key]
        stage_block = a.get("stage_guidance", {}).get(stage, {})
        if isinstance(stage_block, dict):
            stage_local = stage_block.get("local_context", "")
            stage_workflow = stage_block.get("workflow", "")
            stage_output = stage_block.get("output_contract", "")
        else:
            stage_local = str(stage_block)
            stage_workflow = ""
            stage_output = ""

        agent_rule = a.get("agent_rule", "")
        safety_notes = a.get("safety_notes", "")
        return (
            f"Project context:\n{a['project_context']}\n\n"
            f"Local context:\n{local_context}\n\n"
            f"Persona:\n{a['persona']}\n\n"
            f"Goal:\n{a['goal']}\n\n"
            f"Agent-specific rule:\n{agent_rule}\n\n"
            f"Stage local context ({stage}):\n{stage_local}\n\n"
            f"Stage workflow ({stage}):\n{stage_workflow}\n\n"
            f"Stage output contract ({stage}):\n{stage_output}\n\n"
            f"Safety notes:\n{safety_notes}"
        )

    async def _call_agent(self, run_id: str, stage: Stage, agent_key: str, prompt: str, cfg: RunConfig) -> str:
        model = (
            cfg.agent_model_map.get(agent_key)
            or cfg.model_map.get(stage.value)
            or DEFAULT_MODEL
        )
        chat_key = f"{run_id}:{stage.value}:{agent_key}:{model}"
        if chat_key not in self._chat_threads:
            self._chat_threads[chat_key] = []
        chat_history = self._chat_threads[chat_key]

        history_excerpt = ""
        if chat_history:
            history_excerpt = "Prior thread context:\n" + "\n".join(
                [f"{m['role'].upper()}: {m['content']}" for m in chat_history[-6:]]
            )

        user_prompt = prompt[: cfg.max_context_chars]
        if history_excerpt:
            user_prompt = f"{history_excerpt}\n\nNew instruction:\n{user_prompt}"

        self._log(run_id, stage.value, agent_key, "meta", f"chat_id={chat_key}")
        self._log(run_id, stage.value, agent_key, "prompt", prompt[:600])
        result = await self.client.complete(
            model=model,
            system_prompt=self._agent_system_prompt(agent_key, f"Stage={stage.value}", stage.value),
            user_prompt=user_prompt[: cfg.max_context_chars],
            temperature=cfg.temperature,
        )
        chat_history.append({"role": "user", "content": prompt[:3000]})
        chat_history.append({"role": "assistant", "content": result[:3000]})
        self._log(run_id, stage.value, agent_key, "output", result[:1200])
        return result

    async def _stage_elements(self, run: RunState) -> dict:
        cfg = run.config
        agents = ["creative", "viral", "familiarity", "simplifier", "crazy_genius"]
        if cfg.enable_data_specialist:
            agents.append("data_specialist")

        selected_keys = cfg.selected_element_types or [
            "main_characters",
            "side_characters",
            "locations",
            "situations",
            "protagonist_emotions",
            "audience_emotions",
            "absurd_situations",
            "narrator_locations",
        ]
        existing_block = ""
        if cfg.include_existing_elements and cfg.existing_elements.strip():
            existing_block = (
                "Existing elements that can be reused, improved, or expanded (optional):\n"
                f"{cfg.existing_elements.strip()}\n\n"
            )

        prompt = (
            "Generate story elements as strict JSON.\n"
            f"Only generate these keys: {', '.join(selected_keys)}.\n"
            f"Generate {cfg.output_count * 2} items per selected key.\n"
            "Every item must include: id, name, description, retention_reason.\n"
            f"{existing_block}"
            f"Custom instruction: {cfg.custom_instruction or 'None'}."
        )

        tasks = [self._call_agent(run.run_id, Stage.ELEMENTS, a, prompt, cfg) for a in agents]
        outputs = await asyncio.gather(*tasks)
        round_one = dict(zip(agents, outputs))

        feedback_tasks = []
        for agent in agents:
            others = {k: v for k, v in round_one.items() if k != agent}
            feedback_prompt = (
                "These are the suggestions and arguments from other agents.\n"
                "Give one feedback round on all other proposals (exclude your own).\n"
                "Return strict JSON with:\n"
                "1) feedback_by_agent: object keyed by agent\n"
                "2) rankings: object keyed by each selected element key with ranked_ids\n"
                "3) recommendation_summary.\n\n"
                f"Selected element keys: {selected_keys}\n"
                f"Other proposals:\n{json.dumps(others)}"
            )
            feedback_tasks.append(self._call_agent(run.run_id, Stage.ELEMENTS, agent, feedback_prompt, cfg))
        feedback_outputs = await asyncio.gather(*feedback_tasks)
        round_two_feedback = dict(zip(agents, feedback_outputs))

        decider_context = {
            "selected_element_keys": selected_keys,
            "round_1_proposals": round_one,
            "round_2_feedback_and_rankings": round_two_feedback,
        }
        decider_prompt = (
            "You are the final decider. Use full context from all agents.\n"
            "Decide final picks and return strict JSON with:\n"
            "selected_elements (object by element key), decision_reasoning, winning_ids, confidence.\n\n"
            f"{json.dumps(decider_context)}"
        )
        decided = await self._call_agent(run.run_id, Stage.ELEMENTS, "decider", decider_prompt, cfg)
        return {
            "elements": decided,
            "source_agents": agents,
            "selected_element_types": selected_keys,
            "round_1_proposals": round_one,
            "round_2_feedback": round_two_feedback,
        }

    async def _stage_simple(self, run: RunState, stage: Stage, instruction: str, context: str) -> dict:
        cfg = run.config
        seed_agents = ["creative", "viral", "crazy_genius"]
        tasks = [
            self._call_agent(run.run_id, stage, a, f"{instruction}\n\nContext:\n{context}", cfg)
            for a in seed_agents
        ]
        drafts = await asyncio.gather(*tasks)
        feedback_prompt = "Give ranked critique of these drafts and suggested improvements:\n\n" + "\n\n".join(drafts)
        feedback = await self._call_agent(run.run_id, stage, "familiarity", feedback_prompt, cfg)
        decider_prompt = (
            "Use drafts + feedback to output final JSON with fields: title, reasoning, ids_used, storyteller.\n\n"
            f"Drafts:\n{json.dumps(drafts)}\n\nFeedback:\n{feedback}"
        )
        final = await self._call_agent(run.run_id, stage, "decider", decider_prompt, cfg)
        return {"drafts": drafts, "feedback": feedback, "final": final}

    async def _stage_story_pipeline(self, run: RunState, context: str) -> dict:
        cfg = run.config
        hook = await self._call_agent(
            run.run_id,
            Stage.HOOK,
            "structure_expert",
            f"Create initial_hook and hook body from headline context. Return JSON.\n\n{context}",
            cfg,
        )
        plan = await self._call_agent(
            run.run_id,
            Stage.STORY_PLAN,
            "structure_expert",
            (
                "Create story plan with sections, cliffhanger points, and part breakdown. "
                f"Target parts: {cfg.target_parts}, target minutes: {cfg.target_minutes}. Return JSON."
            ),
            cfg,
        )
        draft_a, draft_b = await asyncio.gather(
            self._call_agent(run.run_id, Stage.STORY, "creative", f"Write story from plan:\n{plan}", cfg),
            self._call_agent(run.run_id, Stage.STORY, "crazy_genius", f"Write alternative story from plan:\n{plan}", cfg),
        )
        critique = await self._call_agent(
            run.run_id,
            Stage.STORY,
            "retention_expert",
            f"Compare draft A and B and suggest merged improvements.\nA:\n{draft_a}\n\nB:\n{draft_b}",
            cfg,
        )
        final_story = await self._call_agent(
            run.run_id,
            Stage.STORY,
            "decider",
            f"Produce final story merging best ideas + critique.\n{critique}",
            cfg,
        )
        script = await self._call_agent(
            run.run_id,
            Stage.SCRIPT,
            "script_translator",
            (
                "Transform story into short script files by part. Must be B2 level, spoken style, numbers written out, "
                "and part-aware recap hooks. Return JSON with keys all_parts and parts[]."
                f"\n\nStory:\n{final_story}"
            ),
            cfg,
        )
        video_text = await self._call_agent(
            run.run_id,
            Stage.VIDEO_TEXT,
            "short_expert",
            "Generate per-part on-screen opening headline + social caption optimized for curiosity/FOMO. Return JSON.",
            cfg,
        )
        return {
            "hook": hook,
            "plan": plan,
            "story_a": draft_a,
            "story_b": draft_b,
            "critique": critique,
            "final_story": final_story,
            "script": script,
            "video_text": video_text,
        }

    async def execute(self, run: RunState) -> RunState:
        run.status = "running"
        selected = run.config.selected_stages
        prior_context = ""
        try:
            if Stage.ELEMENTS in selected:
                out = await self._stage_elements(run)
                run.outputs[Stage.ELEMENTS.value] = out
                append_record(Record.new(Stage.ELEMENTS.value, json.dumps(out)[:7000]))
                prior_context += f"\nELEMENTS:\n{out['elements']}"

            if Stage.FORMAT_TYPES in selected:
                out = await self._stage_simple(run, Stage.FORMAT_TYPES, "Create headline format blueprints.", prior_context)
                run.outputs[Stage.FORMAT_TYPES.value] = out
                append_record(Record.new(Stage.FORMAT_TYPES.value, json.dumps(out)[:7000]))
                prior_context += f"\nFORMATS:\n{out['final']}"

            if Stage.HEADLINES in selected:
                out = await self._stage_simple(run, Stage.HEADLINES, "Create viral story headlines.", prior_context)
                run.outputs[Stage.HEADLINES.value] = out
                append_record(Record.new(Stage.HEADLINES.value, json.dumps(out)[:7000]))
                prior_context += f"\nHEADLINES:\n{out['final']}"

            story_set = {Stage.HOOK, Stage.STORY_PLAN, Stage.STORY, Stage.SCRIPT, Stage.VIDEO_TEXT}
            if story_set.intersection(set(selected)):
                out = await self._stage_story_pipeline(run, prior_context)
                for key, val in out.items():
                    stage_key = key if key in [s.value for s in Stage] else key
                    run.outputs[stage_key] = val
                for stage_name, payload in out.items():
                    append_record(Record.new(stage_name, str(payload)[:7000]))

                run_dir = OUTPUT_DIR / run.run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "full_output.json").write_text(json.dumps(run.outputs, indent=2))

                script_dir = run_dir / "script_parts"
                script_dir.mkdir(exist_ok=True)
                (script_dir / "all_parts.txt").write_text(out["script"])

            run.status = "completed"
            return run
        except Exception as exc:  # pragma: no cover
            run.status = "failed"
            run.errors.append(str(exc))
            self._log(run.run_id, "system", "system", "error", str(exc))
            return run
