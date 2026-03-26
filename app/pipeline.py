from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from app.agents import load_agents
from app.config import OUTPUT_DIR
from app.models import Record, RunConfig, RunLogEvent, RunState, Stage
from app.openrouter_client import OpenRouterClient
from app.prompt_builder import PromptBuilder, validate_json_response
from app.storage import append_record, append_run_log, persist_stage_output

DEFAULT_MODEL = "openai/gpt-4o-mini"


@dataclass
class LoopDefinition:
    workflow_type: str
    creator_agents: list[str]
    reviewer_agents: list[str]
    separate_creation: bool
    has_rewrite_round: bool
    has_second_review_round: bool
    has_decider: bool
    has_board: bool
    saves: list[str]
    ordered_reviewers: list[str] | None = None


LOOP_DEFINITIONS: dict[Stage, LoopDefinition] = {
    Stage.ELEMENT_GENERATION: LoopDefinition(
        workflow_type="parallel_generate_review_decide",
        creator_agents=["creative", "viral", "familiarity", "simplifier", "crazy_genius"],
        reviewer_agents=["creative", "viral", "familiarity", "simplifier", "crazy_genius"],
        separate_creation=True,
        has_rewrite_round=False,
        has_second_review_round=False,
        has_decider=True,
        has_board=False,
        saves=["final_selected_elements", "element_group", "source_stage", "decider_reasoning", "proposed_by_agents"],
    ),
    Stage.STORY_FORMAT_GENERATION: LoopDefinition(
        "generate_review_rewrite_review_decide",
        ["creative", "viral", "crazy_genius"],
        ["familiarity", "simplifier"],
        True,
        True,
        True,
        True,
        False,
        ["final_format", "format_explanation", "stage_metadata", "reasoning", "upstream_links"],
    ),
    Stage.HEADLINE_GENERATION: LoopDefinition(
        "generate_review_rewrite_review_decide+decider_board_retry",
        ["creative", "viral", "familiarity", "crazy_genius"],
        ["simplifier", "retention_expert", "humanizer", "short_expert"],
        True,
        True,
        True,
        True,
        True,
        ["approved_headlines", "rejected_headlines", "board_votes", "board_explanations", "context_ids"],
    ),
    Stage.HEADLINE_SELECTION: LoopDefinition(
        "decider_only_selection",
        [],
        [],
        True,
        False,
        False,
        True,
        False,
        ["selected_headline_id", "selection_reason", "rank"],
    ),
    Stage.HOOK_GENERATION: LoopDefinition(
        "generate_review_rewrite_review_decide",
        ["creative", "crazy_genius", "short_expert"],
        ["structure_expert", "retention_expert", "viral", "humanizer", "familiarity"],
        True,
        True,
        True,
        True,
        False,
        ["final_hook", "shortlist", "rationale", "upstream_headline_id"],
    ),
    Stage.STORY_PLANNING: LoopDefinition(
        "collaborative_draft_review_rewrite_review_decide",
        ["creative", "crazy_genius", "structure_expert"],
        ["viral", "familiarity", "retention_expert", "simplifier", "humanizer"],
        False,
        True,
        True,
        True,
        False,
        ["final_story_plan", "part_structure", "cliffhanger_design", "upstream_refs"],
    ),
    Stage.STORY_WRITING: LoopDefinition(
        "competing_drafts_compare_review_rewrite_review_decide",
        ["creative", "crazy_genius"],
        ["structure_expert", "viral", "retention_expert", "familiarity", "humanizer"],
        True,
        True,
        True,
        True,
        False,
        ["final_story", "draft_references", "revision_notes", "upstream_refs"],
    ),
    Stage.SHORT_SCRIPT_WRITING: LoopDefinition(
        "specialist_draft_ordered_review_rewrite_ordered_review_decide",
        ["script_translator"],
        ["creative", "humanizer", "short_expert", "viral", "structure_expert", "familiarity", "retention_expert"],
        True,
        True,
        True,
        True,
        False,
        ["full_script", "per_part_scripts", "part_numbering", "cta_logic", "duration_estimates", "upstream_story_id"],
        ordered_reviewers=["creative", "humanizer", "short_expert", "viral", "structure_expert", "familiarity", "retention_expert"],
    ),
    Stage.VIDEO_HEADLINE_GENERATION: LoopDefinition(
        "parallel_generate_review_decide",
        ["viral", "creative", "short_expert"],
        ["familiarity", "simplifier", "retention_expert"],
        True,
        False,
        False,
        True,
        False,
        ["selected_video_headlines", "readability_metadata", "associated_script_id"],
    ),
    Stage.CAPTION_GENERATION: LoopDefinition(
        "parallel_generate_review_decide",
        ["viral", "creative", "familiarity"],
        ["retention_expert", "short_expert", "simplifier"],
        True,
        False,
        False,
        True,
        False,
        ["selected_captions", "caption_style_type", "associated_script_or_video_headline_id"],
    ),
}


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
        model = cfg.agent_model_map.get(agent_key) or cfg.model_map.get(stage.value) or DEFAULT_MODEL
        chat_key = f"{run_id}:{stage.value}:{agent_key}:{model}"
        thread = self._chat_threads.setdefault(chat_key, [])

        history_excerpt = ""
        if thread:
            history_excerpt = "Prior thread context:\n" + "\n".join(
                [f"{m['role'].upper()}: {m['content']}" for m in thread[-6:]]
            )

        user_prompt = prompt[: cfg.max_context_chars]
        if history_excerpt:
            user_prompt = f"{history_excerpt}\n\nNew instruction:\n{user_prompt}"

        system_prompt = self.prompt_builder.build_system_prompt(agent_key, stage)
        self._log(run_id, stage.value, agent_key, "meta", f"chat_id={chat_key}")
        self._log(run_id, stage.value, agent_key, "acting", f"model={model}")
        self._log(run_id, stage.value, agent_key, "system_prompt", self._trim_for_log(system_prompt))
        self._log(run_id, stage.value, agent_key, "user_prompt", self._trim_for_log(user_prompt))
        result = await self.client.complete(model=model, system_prompt=system_prompt, user_prompt=user_prompt, temperature=cfg.temperature)
        safe_result = str(result or "")
        self._log(run_id, stage.value, agent_key, "output", self._trim_for_log(safe_result, 1200))
        thread.append({"role": "user", "content": prompt[:3000]})
        thread.append({"role": "assistant", "content": safe_result[:3000]})
        try:
            return validate_json_response(result)
        except Exception:
            return {
                "agent_name": agent_key,
                "stage": stage.value,
                "task_type": "fallback_parse",
                "status": "coerced_from_text",
                "raw_text": str(result)[: cfg.max_context_chars],
                "items": [{"text": line} for line in str(result).splitlines()[:40]],
                "warnings": ["Model returned non-JSON output; coerced locally."],
            }

    def _build_prompt(self, task: str, objective: str, context: dict, schema: dict, custom_instruction: str) -> str:
        return self.prompt_builder.build_user_prompt(
            task=task,
            objective=objective,
            settings={"structured_loop": True},
            available_context=context,
            custom_instruction=custom_instruction,
            output_schema=schema,
        )

    async def _parallel_generate_review_decide(self, run: RunState, stage: Stage, context: dict) -> dict:
        cfg = run.config
        loop = LOOP_DEFINITIONS[stage]
        creators = list(loop.creator_agents)
        if stage == Stage.ELEMENT_GENERATION and cfg.enable_data_specialist:
            creators.append("data_specialist")

        self._log(run.run_id, stage.value, "system", "loop_round", "round=1 draft generation")
        draft_tasks = [
            self._call_agent(
                run.run_id,
                stage,
                creator,
                self._build_prompt("create_candidates", f"Generate candidates for {stage.value}.", context, {"items": []}, cfg.custom_instruction or ""),
                cfg,
            )
            for creator in creators
        ]
        drafts = dict(zip(creators, await asyncio.gather(*draft_tasks)))

        self._log(run.run_id, stage.value, "system", "loop_round", "round=2 feedback and ranking")
        feedback_tasks = [
            self._call_agent(
                run.run_id,
                stage,
                reviewer,
                self._build_prompt("feedback", "Review all candidates and rank them.", {"drafts": drafts}, {"rankings": [], "feedback": []}, cfg.custom_instruction or ""),
                cfg,
            )
            for reviewer in loop.reviewer_agents
        ]
        feedback = dict(zip(loop.reviewer_agents, await asyncio.gather(*feedback_tasks)))

        self._log(run.run_id, stage.value, "decider", "loop_round", "round=3 decider final")
        final = await self._call_agent(
            run.run_id,
            stage,
            "decider",
            self._build_prompt("final_decision", "Create final output using all drafts and feedback.", {"drafts": drafts, "feedback": feedback, "what_to_save": loop.saves}, {"final_output": {}}, cfg.custom_instruction or ""),
            cfg,
        )
        return {"loop_definition": loop.__dict__, "drafts": drafts, "feedback": feedback, "final_decider_output": final}

    async def _generate_review_rewrite_review_decide(self, run: RunState, stage: Stage, context: dict) -> dict:
        cfg = run.config
        loop = LOOP_DEFINITIONS[stage]
        creators = list(loop.creator_agents)
        if stage == Stage.HEADLINE_GENERATION and cfg.enable_data_specialist:
            creators.append("data_specialist")

        self._log(run.run_id, stage.value, "system", "loop_round", "round=1 initial drafts")
        first = dict(zip(creators, await asyncio.gather(*[
            self._call_agent(run.run_id, stage, a, self._build_prompt("draft_v1", f"Create first draft for {stage.value}.", context, {"draft": {}}, cfg.custom_instruction or ""), cfg) for a in creators
        ])))

        self._log(run.run_id, stage.value, "system", "loop_round", "round=2 first feedback")
        fb1 = dict(zip(loop.reviewer_agents, await asyncio.gather(*[
            self._call_agent(run.run_id, stage, a, self._build_prompt("feedback_v1", "Review all draft v1 outputs.", {"drafts": first}, {"feedback": [], "ranking": []}, cfg.custom_instruction or ""), cfg) for a in loop.reviewer_agents
        ])))

        self._log(run.run_id, stage.value, "system", "loop_round", "round=3 rewrite")
        revised = dict(zip(creators, await asyncio.gather(*[
            self._call_agent(run.run_id, stage, a, self._build_prompt("draft_v2", "Rewrite your draft using review feedback.", {"your_v1": first.get(a), "feedback": fb1}, {"draft": {}}, cfg.custom_instruction or ""), cfg) for a in creators
        ])))

        self._log(run.run_id, stage.value, "system", "loop_round", "round=4 second feedback")
        fb2 = dict(zip(loop.reviewer_agents, await asyncio.gather(*[
            self._call_agent(run.run_id, stage, a, self._build_prompt("feedback_v2", "Review revised outputs and rank final candidates.", {"revised": revised}, {"feedback": [], "ranking": []}, cfg.custom_instruction or ""), cfg) for a in loop.reviewer_agents
        ])))

        self._log(run.run_id, stage.value, "decider", "loop_round", "round=5 decider final")
        final = await self._call_agent(
            run.run_id, stage, "decider",
            self._build_prompt("final_decision", "Create final output using full two-pass loop evidence.", {"initial_drafts": first, "feedback_round_1": fb1, "revised_drafts": revised, "feedback_round_2": fb2, "what_to_save": loop.saves}, {"final_output": {}}, cfg.custom_instruction or ""),
            cfg,
        )
        out = {
            "loop_definition": loop.__dict__,
            "draft_round_1": first,
            "feedback_round_1": fb1,
            "draft_round_2": revised,
            "feedback_round_2": fb2,
            "final_decider_output": final,
        }

        if loop.has_board:
            self._log(run.run_id, stage.value, "board", "board_status", "pending")
            board_members = ["board_reviewer_1", "board_reviewer_2", "board_reviewer_3"]
            board_votes = {}
            for member in board_members:
                board_votes[member] = await self._call_agent(
                    run.run_id,
                    stage,
                    "board_reviewer",
                    self._build_prompt("board_review", "Vote PASS/FAIL and rank shortlisted headlines.", {"decider_output": final}, {"vote": "PASS|FAIL", "ranking": []}, cfg.custom_instruction or ""),
                    cfg,
                )
            pass_votes = sum(1 for vote in board_votes.values() if str(vote.get("vote", "")).upper() == "PASS")
            board_status = "passed" if pass_votes >= 2 else "failed"
            if board_status == "failed":
                self._log(run.run_id, stage.value, "board", "board_status", "failed_retry_pending")
                retry = await self._call_agent(
                    run.run_id,
                    stage,
                    "decider",
                    self._build_prompt("board_retry", "Revise headline shortlist once using board feedback.", {"previous_final": final, "board_votes": board_votes}, {"revised_final": {}}, cfg.custom_instruction or ""),
                    cfg,
                )
                retry_votes = {}
                for member in board_members:
                    retry_votes[member] = await self._call_agent(
                        run.run_id,
                        stage,
                        "board_reviewer",
                        self._build_prompt("board_review_retry", "Second and final PASS/FAIL board vote.", {"revised": retry}, {"vote": "PASS|FAIL", "ranking": []}, cfg.custom_instruction or ""),
                        cfg,
                    )
                retry_pass = sum(1 for vote in retry_votes.values() if str(vote.get("vote", "")).upper() == "PASS")
                board_status = "passed" if retry_pass >= 2 else "failed"
                out["board_retry"] = {"decider_retry": retry, "votes": retry_votes}
            out["board"] = {"status": board_status, "votes": board_votes}
            self._log(run.run_id, stage.value, "board", "board_status", board_status)
        return out

    async def _headline_selection(self, run: RunState, context: dict) -> dict:
        cfg = run.config
        stage = Stage.HEADLINE_SELECTION
        loop = LOOP_DEFINITIONS[stage]
        self._log(run.run_id, stage.value, "decider", "loop_round", "round=1 selection")
        final = await self._call_agent(
            run.run_id,
            stage,
            "decider",
            self._build_prompt("select_headline", "Select production headline(s) from approved list.", {"approved_headlines": context.get("approved_headlines", [])}, {"selected_ids": [], "reasoning": ""}, cfg.custom_instruction or ""),
            cfg,
        )
        return {"loop_definition": loop.__dict__, "final_decider_output": final}

    async def _execute_stage(self, run: RunState, stage: Stage, context: dict) -> dict:
        loop = LOOP_DEFINITIONS[stage]
        if loop.workflow_type == "parallel_generate_review_decide":
            return await self._parallel_generate_review_decide(run, stage, context)
        if loop.workflow_type.startswith("generate_review_rewrite_review_decide"):
            return await self._generate_review_rewrite_review_decide(run, stage, context)
        if loop.workflow_type == "decider_only_selection":
            return await self._headline_selection(run, context)
        if loop.workflow_type == "collaborative_draft_review_rewrite_review_decide":
            return await self._generate_review_rewrite_review_decide(run, stage, context)
        if loop.workflow_type == "competing_drafts_compare_review_rewrite_review_decide":
            return await self._generate_review_rewrite_review_decide(run, stage, context)
        if loop.workflow_type == "specialist_draft_ordered_review_rewrite_ordered_review_decide":
            return await self._generate_review_rewrite_review_decide(run, stage, context)
        raise ValueError(f"Unsupported workflow type: {loop.workflow_type}")

    async def _summarize_for_storage(self, run: RunState, stage: Stage, stage_output: dict) -> tuple[str, str]:
        prompt = self._build_prompt(
            "storage_summary",
            "Summarize this stage output for CSV storage.",
            {"stage_output": stage_output},
            {"summary": "", "notes": "", "quality_score": ""},
            run.config.custom_instruction or "",
        )
        try:
            summary_payload = await self._call_agent(run.run_id, stage, "data_archivist", prompt, run.config)
            summary = str(summary_payload.get("summary", ""))[:600]
            notes = str(summary_payload.get("notes", ""))[:600]
            if summary:
                return summary, notes
        except Exception:
            pass
        fallback = json.dumps(stage_output.get("final_decider_output", stage_output), ensure_ascii=False)[:600]
        return fallback, "fallback_summary"

    async def execute(self, run: RunState) -> RunState:
        run.status = "running"
        aggregate_context: dict = {}
        append_run_log(
            {
                "run_id": run.run_id,
                "started_at": run.created_at,
                "finished_at": "",
                "run_mode": run.config.mode,
                "selected_stages": json.dumps([s.value for s in run.config.selected_stages]),
                "included_context_sources": json.dumps(run.config.context_selection),
                "custom_instruction": run.config.custom_instruction,
                "status": "running",
                "error_message": "",
                "total_items_generated": 0,
                "total_tokens_if_available": "",
                "notes": "",
            }
        )
        try:
            total_items_generated = 0
            for stage in run.config.selected_stages:
                out = await self._execute_stage(run, stage, aggregate_context)
                run.outputs[stage.value] = out
                append_record(Record.new(stage.value, json.dumps(out)))
                summary, summary_notes = await self._summarize_for_storage(run, stage, out)
                persist_stage_output(
                    run_id=run.run_id,
                    stage_name=stage.value,
                    output=out,
                    custom_instruction=run.config.custom_instruction,
                    summary=summary,
                    summary_notes=summary_notes,
                )
                total_items_generated += 1
                aggregate_context[stage.value] = out

            run_dir = OUTPUT_DIR / run.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "full_output.json").write_text(json.dumps(run.outputs, indent=2))
            run.status = "completed"
            append_run_log(
                {
                    "run_id": run.run_id,
                    "started_at": run.created_at,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "run_mode": run.config.mode,
                    "selected_stages": json.dumps([s.value for s in run.config.selected_stages]),
                    "included_context_sources": json.dumps(run.config.context_selection),
                    "custom_instruction": run.config.custom_instruction,
                    "status": "completed",
                    "error_message": "",
                    "total_items_generated": total_items_generated,
                    "total_tokens_if_available": "",
                    "notes": "",
                }
            )
            return run
        except Exception as exc:  # pragma: no cover
            run.status = "failed"
            run.errors.append(str(exc))
            self._log(run.run_id, "system", "system", "error", str(exc))
            append_run_log(
                {
                    "run_id": run.run_id,
                    "started_at": run.created_at,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "run_mode": run.config.mode,
                    "selected_stages": json.dumps([s.value for s in run.config.selected_stages]),
                    "included_context_sources": json.dumps(run.config.context_selection),
                    "custom_instruction": run.config.custom_instruction,
                    "status": "failed",
                    "error_message": str(exc),
                    "total_items_generated": len(run.outputs),
                    "total_tokens_if_available": "",
                    "notes": "",
                }
            )
            return run
