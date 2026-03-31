from __future__ import annotations

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.config import DATABASE_URL
from app.models import (
    Agent,
    Base,
    Persona,
    PromptBlock,
    PromptBlockStage,
    StageDefinition,
    StageKey,
)

engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def session_scope() -> Session:
    return SessionLocal()


def init_db() -> None:
    Base.metadata.create_all(engine)
    with session_scope() as session:
        _seed_stages(session)
        _seed_personas_agents(session)
        _seed_prompt_blocks(session)
        session.commit()


def _seed_stages(session: Session) -> None:
    existing = {s.key for s in session.scalars(select(StageDefinition)).all()}
    defs = [
        (StageKey.ELEMENTS.value, "Elements", "A", "Reusable story ingredients."),
        (StageKey.HEADLINE_FORMATS.value, "Headline Formats", "A", "Structural templates for headlines/stories."),
        (StageKey.HEADLINES.value, "Headlines", "A", "Final short-form story headlines."),
        (StageKey.STORY_STRUCTURE.value, "Story Structure", "B", "Future stage."),
        (StageKey.HOOK.value, "Hook", "B", "Future stage."),
        (StageKey.STORY.value, "Story", "B", "Future stage."),
        (StageKey.SCRIPT.value, "Script", "B", "Future stage."),
        (StageKey.PACKAGE.value, "Title/Description Package", "B", "Future stage."),
    ]
    for key, name, part, desc in defs:
        if key not in existing:
            session.add(StageDefinition(key=key, name=name, part=part, description=desc))


def _seed_personas_agents(session: Session) -> None:
    starter = {
        "Creator": "Generates numerous candidate ideas quickly with clear variation.",
        "Reviewer": "Evaluates clarity, curiosity, and emotional pull. Gives precise feedback.",
        "Decider": "Selects final options with concise reasoning and consistency.",
        "Platform Architect": "Understands all platform entities and can propose configurations.",
    }
    personas = {p.name: p for p in session.scalars(select(Persona)).all()}
    for name, text in starter.items():
        if name not in personas:
            session.add(Persona(name=name, description=f"{name} persona", persona_text=text))
    session.flush()
    personas = {p.name: p for p in session.scalars(select(Persona)).all()}

    agents = {a.name for a in session.scalars(select(Agent)).all()}
    defaults = [
        ("creator_alpha", "creator", "Creator"),
        ("creator_beta", "creator", "Creator"),
        ("reviewer", "reviewer", "Reviewer"),
        ("decider", "decider", "Decider"),
        ("platform_agent", "platform", "Platform Architect"),
    ]
    for name, role, persona_name in defaults:
        if name not in agents:
            session.add(Agent(name=name, role=role, persona_id=personas[persona_name].id))


def _seed_prompt_blocks(session: Session) -> None:
    if session.scalars(select(PromptBlock)).first():
        return
    blocks = [
        PromptBlock(name="Quality bar", description_for_agent="Baseline quality", prompt_text="Prefer concrete, vivid, and socially plausible ideas."),
        PromptBlock(name="School/Work focus", description_for_agent="Domain focus", prompt_text="Bias outputs toward school and workplace social situations.", scope="stage"),
        PromptBlock(name="No cringe", description_for_agent="Tone safety", prompt_text="Avoid cliché clickbait phrasing and robotic voice."),
    ]
    session.add_all(blocks)
    session.flush()
    stage_block = next(b for b in blocks if b.name == "School/Work focus")
    session.add(PromptBlockStage(prompt_block_id=stage_block.id, stage_key=StageKey.ELEMENTS.value))
    session.add(PromptBlockStage(prompt_block_id=stage_block.id, stage_key=StageKey.HEADLINES.value))
