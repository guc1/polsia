from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from app.config import AGENTS_FILE

PROJECT_CONTEXT = (
    "You contribute to Project Viral: create storytime content for short-form video platforms. "
    "Optimize for high retention, curiosity, familiarity, and believable-but-absurd situations."
)

UNIVERSAL_GOAL = (
    "Use your specialty to improve quality while collaborating with other agents. "
    "Be concise, critical, and constructive. Respect constraints and output structured results."
)


@dataclass
class AgentProfile:
    name: str
    persona: str


DEFAULT_AGENTS = {
    "creative": AgentProfile(
        "creative",
        "You connect unusual dots and propose unique but usable ideas.",
    ),
    "viral": AgentProfile(
        "viral",
        "You optimize concepts for algorithmic performance and engagement.",
    ),
    "familiarity": AgentProfile(
        "familiarity",
        "You maximize recognizability and emotional resonance with audience experience.",
    ),
    "simplifier": AgentProfile(
        "simplifier",
        "You reduce complexity into clear actionable core ideas.",
    ),
    "decider": AgentProfile(
        "decider",
        "You make the final decision based on discussion quality and strategic fit.",
    ),
    "crazy_genius": AgentProfile(
        "crazy_genius",
        "You propose highly novel twists while keeping outputs understandable.",
    ),
    "data_specialist": AgentProfile(
        "data_specialist",
        "You reason from performance data patterns and evidence.",
    ),
    "structure_expert": AgentProfile(
        "structure_expert",
        "You enforce hook-body-reward structure and cliffhanger pacing.",
    ),
    "retention_expert": AgentProfile(
        "retention_expert",
        "You optimize for watch time and continuation to next part.",
    ),
    "humanizer": AgentProfile(
        "humanizer",
        "You rewrite text to sound natural and imperfectly human.",
    ),
    "short_expert": AgentProfile(
        "short_expert",
        "You adapt content specifically to short-form consumption behavior.",
    ),
    "script_translator": AgentProfile(
        "script_translator",
        "You transform stories into spoken short-video scripts with part-aware recaps.",
    ),
}


def load_agents() -> dict[str, dict]:
    if AGENTS_FILE.exists():
        return json.loads(AGENTS_FILE.read_text())
    payload = {
        k: {
            "name": v.name,
            "persona": v.persona,
            "project_context": PROJECT_CONTEXT,
            "goal": UNIVERSAL_GOAL,
        }
        for k, v in DEFAULT_AGENTS.items()
    }
    AGENTS_FILE.write_text(json.dumps(payload, indent=2))
    return payload


def save_agents(agents: dict) -> None:
    AGENTS_FILE.write_text(json.dumps(agents, indent=2))
