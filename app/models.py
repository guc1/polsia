from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class Stage(str, Enum):
    ELEMENTS = "elements"
    FORMAT_TYPES = "format_types"
    HEADLINES = "headlines"
    HOOK = "hook"
    STORY_PLAN = "story_plan"
    STORY = "story"
    SCRIPT = "script"
    VIDEO_TEXT = "video_text"


DEFAULT_STAGE_ORDER = [
    Stage.ELEMENTS,
    Stage.FORMAT_TYPES,
    Stage.HEADLINES,
    Stage.HOOK,
    Stage.STORY_PLAN,
    Stage.STORY,
    Stage.SCRIPT,
    Stage.VIDEO_TEXT,
]


@dataclass
class RunConfig:
    selected_stages: list[Stage]
    mode: str = "sequential"
    model_map: dict[str, str] = field(default_factory=dict)
    agent_model_map: dict[str, str] = field(default_factory=dict)
    temperature: float = 0.7
    max_context_chars: int = 12000
    target_minutes: int = 2
    target_parts: int = 3
    custom_instruction: str = ""
    enable_data_specialist: bool = True
    enable_format_context: bool = True
    output_count: int = 3
    selected_element_types: list[str] = field(default_factory=list)
    existing_elements: str = ""
    include_existing_elements: bool = False


@dataclass
class RunLogEvent:
    run_id: str
    stage: str
    agent: str
    role: str
    message: str
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class RunState:
    run_id: str
    config: RunConfig
    status: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    outputs: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class Record:
    id: str
    created_at: str
    stage: str
    payload: str
    source_ids: str = ""

    @staticmethod
    def new(stage: str, payload: str, source_ids: str = "") -> "Record":
        return Record(
            id=str(uuid4()),
            created_at=datetime.now(timezone.utc).isoformat(),
            stage=stage,
            payload=payload,
            source_ids=source_ids,
        )
