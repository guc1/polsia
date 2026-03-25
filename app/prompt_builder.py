from __future__ import annotations

import json
from pathlib import Path

from app.config import ROOT
from app.models import Stage

PROMPTS_DIR = ROOT / "prompts"


AGENT_PERSONA_FILES = {
    "creative": "creative.txt",
    "viral": "viral.txt",
    "familiarity": "familiarity.txt",
    "simplifier": "simplifier.txt",
    "decider": "decider.txt",
    "crazy_genius": "crazy_genius.txt",
    "data_specialist": "data_specialist.txt",
    "structure_expert": "structure_expert.txt",
    "retention_expert": "retention_expert.txt",
    "humanizer": "humanizer.txt",
    "short_expert": "short_expert.txt",
    "script_translator": "story_to_short_translator.txt",
    "board_reviewer": "board_member.txt",
}

STAGE_CONTEXT_FILES = {
    Stage.ELEMENTS: "element_generation.txt",
    Stage.FORMAT_TYPES: "story_format_generation.txt",
    Stage.HEADLINES: "headline_generation.txt",
    Stage.HOOK: "hook_generation.txt",
    Stage.STORY_PLAN: "story_planning.txt",
    Stage.STORY: "story_writing.txt",
    Stage.SCRIPT: "short_script_writing.txt",
    Stage.VIDEO_TEXT: "video_headline_generation.txt",
}


class PromptBuilder:
    def __init__(self) -> None:
        self.base_system_prompt = self._read_text(PROMPTS_DIR / "shared" / "base_system_prompt.txt")

    def _read_text(self, path: Path) -> str:
        return path.read_text().strip()

    def build_system_prompt(self, agent_key: str, stage: Stage) -> str:
        persona_file = AGENT_PERSONA_FILES.get(agent_key, "creative.txt")
        stage_file = STAGE_CONTEXT_FILES.get(stage, "caption_generation.txt")

        persona_block = self._read_text(PROMPTS_DIR / "personas" / persona_file)
        stage_context = self._read_text(PROMPTS_DIR / "stages" / stage_file)

        return (
            self.base_system_prompt.replace("{STAGE_CONTEXT}", stage_context)
            .replace("{PERSONA_BLOCK}", persona_block)
            .strip()
        )

    def build_user_prompt(
        self,
        *,
        task: str,
        objective: str,
        settings: dict,
        available_context: dict,
        custom_instruction: str,
        output_schema: dict,
    ) -> str:
        return (
            "TASK:\n"
            f"{task}\n\n"
            "OBJECTIVE:\n"
            f"{objective}\n\n"
            "SETTINGS:\n"
            f"{json.dumps(settings, ensure_ascii=False, indent=2)}\n\n"
            "AVAILABLE CONTEXT:\n"
            f"{json.dumps(available_context, ensure_ascii=False, indent=2)}\n\n"
            "CUSTOM INSTRUCTION:\n"
            f"{custom_instruction}\n\n"
            "OUTPUT SCHEMA:\n"
            f"{json.dumps(output_schema, ensure_ascii=False, indent=2)}\n\n"
            "IMPORTANT RULES:\n"
            "- Use only the context needed.\n"
            "- If you use specific source items, list their IDs.\n"
            "- Stay inside the requested stage.\n"
            "- Return valid JSON only."
        )


def validate_json_response(text: object) -> dict:
    if text is None:
        raise ValueError("Model output is null, expected JSON object text.")
    raw = str(text).strip()
    candidates = [raw]

    if "```" in raw:
        stripped = raw.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
        candidates.append(stripped)

    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidates.append(raw[first_brace:last_brace + 1].strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Model output must be a JSON object.")
