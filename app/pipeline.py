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
from app.prompt_builder import PromptBuilder, validate_json_response
from app.storage import append_record

DEFAULT_MODEL = "openai/gpt-4o-mini"


class Pipeline:
    def __init__(self, event_cb: Callable[[RunLogEvent], None]):
        self.event_cb = event_cb
        self.client = OpenRouterClient()
        self.agents = load_agents()
        self.prompt_builder = PromptBuilder()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._chat_threads: dict[str, list[dict[str, str]]] = {}

    def _log(self, run_id: str, stage: str, agent: str, role: str, message: str) -> None:
        self.event_cb(RunLogEvent(run_id=run_id, stage=stage, agent=agent, role=role, message=message))

    @staticmethod
    def _trim_for_log(text: str, limit: int = 4000) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + "\n...[truncated for log]..."

    async def _call_agent(self, run_id: str, stage: Stage, agent_key: str, prompt: str, cfg: RunConfig) -> dict:
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
        system_prompt = self.prompt_builder.build_system_prompt(agent_key, stage)
        self._log(run_id, stage.value, agent_key, "system_prompt", self._trim_for_log(system_prompt))
        self._log(run_id, stage.value, agent_key, "user_prompt", self._trim_for_log(user_prompt))
        result = await self.client.complete(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt[: cfg.max_context_chars],
            temperature=cfg.temperature,
        )
        try:
            parsed_result = validate_json_response(result)
        except ValueError:
            repair_prompt = (
                "Convert your previous answer into valid JSON only. "
                "Do not add markdown fences or extra commentary.\n\n"
                f"Previous answer:\n{result}"
            )
            self._log(run_id, stage.value, agent_key, "warning", "Model returned non-JSON. Requesting JSON repair pass.")
            repaired = await self.client.complete(
                model=model,
                system_prompt=system_prompt,
                user_prompt=repair_prompt[: cfg.max_context_chars],
                temperature=0,
            )
            parsed_result = validate_json_response(repaired)
            result = repaired
        chat_history.append({"role": "user", "content": prompt[:3000]})
        chat_history.append({"role": "assistant", "content": result[:3000]})
        self._log(run_id, stage.value, agent_key, "output", result[:1200])
        return parsed_result

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

        prompt = self.prompt_builder.build_user_prompt(
            task="create_elements",
            objective="Generate candidate elements for the requested element groups.",
            settings={
                "requested_groups": selected_keys,
                "final_needed_per_group": cfg.output_count,
                "candidate_multiplier": 2,
                "allow_broad_and_specific_mix": True,
                "data_analysis_enabled": cfg.enable_data_specialist,
            },
            available_context={
                "existing_elements": cfg.existing_elements.strip(),
                "context_note": existing_block.strip() or "none",
            },
            custom_instruction=cfg.custom_instruction or "Lean toward recognizable everyday conflict.",
            output_schema={
                "agent_name": "...",
                "stage": "element_generation",
                "task_type": "create_candidates",
                "status": "ok",
                "items": {"main_characters": [{"text": "..."}]},
                "reasoning_summary": "...",
                "used_context_ids": [],
                "confidence": 0.0,
                "warnings": [],
            },
        )

        tasks = [self._call_agent(run.run_id, Stage.ELEMENTS, a, prompt, cfg) for a in agents]
        outputs = await asyncio.gather(*tasks)
        round_one = dict(zip(agents, outputs))

        feedback_tasks = []
        for agent in agents:
            others = {k: v for k, v in round_one.items() if k != agent}
            feedback_prompt = (
                self.prompt_builder.build_user_prompt(
                    task="review_element_candidates",
                    objective="Review candidate elements created by another agent.",
                    settings={"rank_required": True, "max_keep_per_group": cfg.output_count},
                    available_context={"candidate_output_from_other_agent": others, "selected_element_keys": selected_keys},
                    custom_instruction="Be critical about weak, generic, or repetitive ideas.",
                    output_schema={
                        "agent_name": "...",
                        "stage": "element_generation",
                        "task_type": "feedback",
                        "status": "ok",
                        "group_feedback": {},
                        "reasoning_summary": "...",
                        "confidence": 0.0,
                        "warnings": [],
                    },
                )
            )
            feedback_tasks.append(self._call_agent(run.run_id, Stage.ELEMENTS, agent, feedback_prompt, cfg))
        feedback_outputs = await asyncio.gather(*feedback_tasks)
        round_two_feedback = dict(zip(agents, feedback_outputs))

        decider_context = {
            "selected_element_keys": selected_keys,
            "round_1_proposals": round_one,
            "round_2_feedback_and_rankings": round_two_feedback,
        }
        decider_prompt = self.prompt_builder.build_user_prompt(
            task="decide_final_elements",
            objective="Select the final element list using all candidate outputs and all feedback.",
            settings={"final_needed_per_group": cfg.output_count},
            available_context=decider_context,
            custom_instruction="Optimize for long-term reusability and story quality.",
            output_schema={
                "agent_name": "decider",
                "stage": "element_generation",
                "task_type": "final_selection",
                "status": "ok",
                "final_items": {},
                "selection_notes": "...",
                "confidence": 0.0,
                "warnings": [],
            },
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
            self._call_agent(
                run.run_id,
                stage,
                a,
                self.prompt_builder.build_user_prompt(
                    task=f"create_{stage.value}",
                    objective=instruction,
                    settings={"candidate_count_per_agent": 1},
                    available_context={"prior_context": context},
                    custom_instruction=cfg.custom_instruction or "None",
                    output_schema={
                        "agent_name": "...",
                        "stage": stage.value,
                        "task_type": "create_candidates",
                        "status": "ok",
                        "items": [],
                        "reasoning_summary": "...",
                        "confidence": 0.0,
                        "warnings": [],
                    },
                ),
                cfg,
            )
            for a in seed_agents
        ]
        drafts = await asyncio.gather(*tasks)
        feedback_prompt = self.prompt_builder.build_user_prompt(
            task="give_feedback",
            objective="Evaluate another agent's output and suggest concrete improvements.",
            settings={"must_rank": True, "be_direct": True},
            available_context={"candidate_output": drafts, "stage_goal": instruction, "relevant_context": context},
            custom_instruction="Focus on quality, retention, clarity, originality, and practical usability.",
            output_schema={
                "agent_name": "...",
                "stage": stage.value,
                "task_type": "feedback",
                "status": "ok",
                "overall_verdict": "strong|medium|weak",
                "confidence": 0.0,
                "warnings": [],
            },
        )
        feedback = await self._call_agent(run.run_id, stage, "familiarity", feedback_prompt, cfg)
        decider_prompt = self.prompt_builder.build_user_prompt(
            task="make_final_decision",
            objective="Create the strongest final output using all candidate outputs and all feedback.",
            settings={"must_choose_final": True},
            available_context={"all_candidates": drafts, "all_feedback": [feedback], "stage_goal": instruction},
            custom_instruction="Do not average weak ideas together. Choose the strongest path and refine it.",
            output_schema={
                "agent_name": "decider",
                "stage": stage.value,
                "task_type": "final_decision",
                "status": "ok",
                "selected_base": "...",
                "final_output": {},
                "confidence": 0.0,
                "warnings": [],
            },
        )
        final = await self._call_agent(run.run_id, stage, "decider", decider_prompt, cfg)
        return {"drafts": drafts, "feedback": feedback, "final": final}

    async def _stage_story_pipeline(self, run: RunState, context: str) -> dict:
        cfg = run.config
        hook = await self._call_agent(
            run.run_id,
            Stage.HOOK,
            "structure_expert",
            self.prompt_builder.build_user_prompt(
                task="create_hooks",
                objective="Generate hook candidates for the selected headline.",
                settings={"round": 1, "need_initial_hook_line": True, "need_extended_hook": True},
                available_context={"selected_headline": context},
                custom_instruction="Make it instantly intriguing without spoiling the twist.",
                output_schema={
                    "agent_name": "...",
                    "stage": "hook_generation",
                    "task_type": "create_candidates",
                    "status": "ok",
                    "hooks": [],
                    "confidence": 0.0,
                    "warnings": [],
                },
            ),
            cfg,
        )
        plan = await self._call_agent(
            run.run_id,
            Stage.STORY_PLAN,
            "structure_expert",
            self.prompt_builder.build_user_prompt(
                task="create_story_plan",
                objective="Create a strong structural plan for the selected headline and final hook.",
                settings={
                    "target_video_length_seconds": cfg.target_minutes * 60,
                    "target_parts": cfg.target_parts,
                    "allow_multi_day_updates": True,
                    "must_include_natural_cliffhangers": True,
                },
                available_context={"selected_headline": context, "final_hook": hook},
                custom_instruction="Design for strong part endings and a satisfying final payoff.",
                output_schema={
                    "agent_name": "...",
                    "stage": "story_planning",
                    "task_type": "create_plan",
                    "status": "ok",
                    "story_plan": {},
                    "reasoning_summary": "...",
                    "confidence": 0.0,
                    "warnings": [],
                },
            ),
            cfg,
        )
        draft_a, draft_b = await asyncio.gather(
            self._call_agent(
                run.run_id,
                Stage.STORY,
                "creative",
                self.prompt_builder.build_user_prompt(
                    task="write_story",
                    objective="Write a strong story draft based on the approved planning materials.",
                    settings={"target_reading_length": "medium", "allow_more_detail_than_final_script": True},
                    available_context={"final_story_plan": plan},
                    custom_instruction="Keep it believable, emotionally active, and easy to adapt into a spoken short.",
                    output_schema={"agent_name": "...", "stage": "story_writing", "task_type": "draft_story", "status": "ok", "story_text": "...", "confidence": 0.0, "warnings": []},
                ),
                cfg,
            ),
            self._call_agent(
                run.run_id,
                Stage.STORY,
                "crazy_genius",
                self.prompt_builder.build_user_prompt(
                    task="write_story",
                    objective="Write a strong story draft based on the approved planning materials.",
                    settings={"target_reading_length": "medium", "allow_more_detail_than_final_script": True},
                    available_context={"final_story_plan": plan},
                    custom_instruction="Keep it believable, emotionally active, and easy to adapt into a spoken short.",
                    output_schema={"agent_name": "...", "stage": "story_writing", "task_type": "draft_story", "status": "ok", "story_text": "...", "confidence": 0.0, "warnings": []},
                ),
                cfg,
            ),
        )
        critique = await self._call_agent(
            run.run_id,
            Stage.STORY,
            "retention_expert",
            self.prompt_builder.build_user_prompt(
                task="give_feedback",
                objective="Evaluate another agent's output and suggest concrete improvements.",
                settings={"must_rank": True, "be_direct": True},
                available_context={"candidate_output": {"A": draft_a, "B": draft_b}},
                custom_instruction="Focus on quality, retention, clarity, originality, and practical usability.",
                output_schema={"agent_name": "...", "stage": "story_writing", "task_type": "feedback", "status": "ok", "confidence": 0.0, "warnings": []},
            ),
            cfg,
        )
        final_story = await self._call_agent(
            run.run_id,
            Stage.STORY,
            "decider",
            self.prompt_builder.build_user_prompt(
                task="make_final_decision",
                objective="Create the strongest final output using all candidate outputs and all feedback.",
                settings={"must_choose_final": True},
                available_context={"all_candidates": [draft_a, draft_b], "all_feedback": [critique], "stage_goal": "write_story"},
                custom_instruction="Do not average weak ideas together. Choose the strongest path and refine it.",
                output_schema={"agent_name": "decider", "stage": "story_writing", "task_type": "final_decision", "status": "ok", "final_output": {}, "confidence": 0.0, "warnings": []},
            ),
            cfg,
        )
        script = await self._call_agent(
            run.run_id,
            Stage.SCRIPT,
            "script_translator",
            self.prompt_builder.build_user_prompt(
                task="write_short_form_script",
                objective="Transform the final story into a multi-part short-form spoken script.",
                settings={"target_part_count": cfg.target_parts, "target_length_seconds_per_part": 35, "language_level": "B2"},
                available_context={"final_story": final_story},
                custom_instruction="Make it sound like someone turned on the camera and started telling the story naturally.",
                output_schema={"agent_name": "story_to_short_translator", "stage": "short_script_writing", "task_type": "draft_script", "status": "ok", "full_script": "...", "parts": [], "confidence": 0.0, "warnings": []},
            ),
            cfg,
        )
        video_text = await self._call_agent(
            run.run_id,
            Stage.VIDEO_TEXT,
            "short_expert",
            self.prompt_builder.build_user_prompt(
                task="create_video_headline",
                objective="Create strong on-screen headline options for the produced short.",
                settings={"option_count": 5, "max_words": 12},
                available_context={"final_short_script": script},
                custom_instruction="Make it easy to read instantly on screen.",
                output_schema={"agent_name": "...", "stage": "video_headline_generation", "task_type": "create_options", "status": "ok", "options": [], "confidence": 0.0, "warnings": []},
            ),
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
                prior_context += f"\nELEMENTS:\n{json.dumps(out['elements'])}"

            if Stage.FORMAT_TYPES in selected:
                out = await self._stage_simple(run, Stage.FORMAT_TYPES, "Create headline format blueprints.", prior_context)
                run.outputs[Stage.FORMAT_TYPES.value] = out
                append_record(Record.new(Stage.FORMAT_TYPES.value, json.dumps(out)[:7000]))
                prior_context += f"\nFORMATS:\n{json.dumps(out['final'])}"

            if Stage.HEADLINES in selected:
                out = await self._stage_simple(run, Stage.HEADLINES, "Create viral story headlines.", prior_context)
                run.outputs[Stage.HEADLINES.value] = out
                append_record(Record.new(Stage.HEADLINES.value, json.dumps(out)[:7000]))
                prior_context += f"\nHEADLINES:\n{json.dumps(out['final'])}"

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
                (script_dir / "all_parts.txt").write_text(json.dumps(out["script"], indent=2))

            run.status = "completed"
            return run
        except Exception as exc:  # pragma: no cover
            run.status = "failed"
            run.errors.append(str(exc))
            self._log(run.run_id, "system", "system", "error", str(exc))
            return run
