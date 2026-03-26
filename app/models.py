from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class Stage(str, Enum):
    ELEMENT_GENERATION = "element_generation"
    STORY_FORMAT_GENERATION = "story_format_generation"
    HEADLINE_GENERATION = "headline_generation"
    HEADLINE_SELECTION = "headline_selection"
    HOOK_GENERATION = "hook_generation"
    STORY_PLANNING = "story_planning"
    STORY_WRITING = "story_writing"
    SHORT_SCRIPT_WRITING = "short_script_writing"
    VIDEO_HEADLINE_GENERATION = "video_headline_generation"
    CAPTION_GENERATION = "caption_generation"


DEFAULT_STAGE_ORDER = [
    Stage.ELEMENT_GENERATION,
    Stage.STORY_FORMAT_GENERATION,
    Stage.HEADLINE_GENERATION,
    Stage.HEADLINE_SELECTION,
    Stage.HOOK_GENERATION,
    Stage.STORY_PLANNING,
    Stage.STORY_WRITING,
    Stage.SHORT_SCRIPT_WRITING,
    Stage.VIDEO_HEADLINE_GENERATION,
    Stage.CAPTION_GENERATION,
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
    context_selection: dict[str, dict[str, Any]] = field(default_factory=dict)
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
    pending_updates: dict[str, Any] = field(default_factory=dict)


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
