from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class StageKey(str, Enum):
    ELEMENTS = "elements"
    HEADLINE_FORMATS = "headline_formats"
    HEADLINES = "headlines"
    STORY_STRUCTURE = "story_structure"
    HOOK = "hook"
    STORY = "story"
    SCRIPT = "script"
    PACKAGE = "title_description_package"


PART_A_STAGES = [StageKey.ELEMENTS, StageKey.HEADLINE_FORMATS, StageKey.HEADLINES]


class StageDefinition(Base):
    __tablename__ = "stage_definitions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120))
    part: Mapped[str] = mapped_column(String(32), default="A")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    description: Mapped[str] = mapped_column(Text, default="")


class Persona(Base):
    __tablename__ = "personas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    description: Mapped[str] = mapped_column(Text, default="")
    persona_text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class Agent(Base):
    __tablename__ = "agents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    role: Mapped[str] = mapped_column(String(60), default="creator")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    model: Mapped[str] = mapped_column(String(120), default="openai/gpt-4o-mini")
    override_prompt: Mapped[str] = mapped_column(Text, default="")
    persona_id: Mapped[int | None] = mapped_column(ForeignKey("personas.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class PromptBlock(Base):
    __tablename__ = "prompt_blocks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    description_for_agent: Mapped[str] = mapped_column(Text, default="")
    prompt_text: Mapped[str] = mapped_column(Text)
    scope: Mapped[str] = mapped_column(String(24), default="shared")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class PromptBlockStage(Base):
    __tablename__ = "prompt_block_stages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prompt_block_id: Mapped[int] = mapped_column(ForeignKey("prompt_blocks.id"))
    stage_key: Mapped[str] = mapped_column(String(80), index=True)


class GenerationFlow(Base):
    __tablename__ = "generation_flows"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(140), unique=True)
    description: Mapped[str] = mapped_column(Text, default="")
    execution_mode: Mapped[str] = mapped_column(String(20), default="sequential")
    default_settings: Mapped[dict] = mapped_column(JSON, default=dict)
    context_rules: Mapped[dict] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)
    stages: Mapped[list["FlowStage"]] = relationship(back_populates="flow", cascade="all, delete-orphan")


class FlowStage(Base):
    __tablename__ = "flow_stages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    flow_id: Mapped[int] = mapped_column(ForeignKey("generation_flows.id"), index=True)
    stage_key: Mapped[str] = mapped_column(String(80))
    stage_order: Mapped[int] = mapped_column(Integer)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    stage_params: Mapped[dict] = mapped_column(JSON, default=dict)
    agent_ids: Mapped[list] = mapped_column(JSON, default=list)
    prompt_block_ids: Mapped[list] = mapped_column(JSON, default=list)
    context_sources: Mapped[list] = mapped_column(JSON, default=list)
    flow: Mapped[GenerationFlow] = relationship(back_populates="stages")


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    flow_id: Mapped[int | None] = mapped_column(ForeignKey("generation_flows.id"), nullable=True)
    status: Mapped[str] = mapped_column(String(24), default="queued")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    config: Mapped[dict] = mapped_column(JSON, default=dict)


class RunEvent(Base):
    __tablename__ = "run_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    event_type: Mapped[str] = mapped_column(String(40), default="log")
    stage_key: Mapped[str] = mapped_column(String(80), default="system")
    agent_name: Mapped[str] = mapped_column(String(120), default="system")
    message: Mapped[str] = mapped_column(Text)


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    stage_key: Mapped[str] = mapped_column(String(80))
    agent_name: Mapped[str] = mapped_column(String(120))
    role: Mapped[str] = mapped_column(String(40))
    content: Mapped[str] = mapped_column(Text)


class Element(Base):
    __tablename__ = "elements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("runs.id"), nullable=True)
    element_type: Mapped[str] = mapped_column(String(80), index=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text)
    reasoning_for_choosing: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class HeadlineFormat(Base):
    __tablename__ = "headline_formats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("runs.id"), nullable=True)
    name: Mapped[str] = mapped_column(String(200))
    blueprint: Mapped[str] = mapped_column(Text)
    reasoning_for_choosing: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Headline(Base):
    __tablename__ = "headlines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("runs.id"), nullable=True)
    headline: Mapped[str] = mapped_column(Text)
    reasoning_for_choosing: Mapped[str] = mapped_column(Text, default="")
    score: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
