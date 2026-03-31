# AGENTS.md — Durable Architecture Guide

## Platform purpose
Polsia is a modular internal platform for creating AI-generated short-form storytime/commentary content. The current implementation intentionally focuses on **Part A** and acts as the foundation for future expansion.

## Scope of Part 1 (implemented)
- Elements
- Headline Formats
- Headlines

Future Part B stages are represented in the stage registry as placeholders:
- Story Structure
- Hook
- Story
- Script
- Headline/Title/Description package

## Stage model philosophy
- Stages are first-class database entities.
- Flows decide which stages run and in what order.
- Each stage can bind its own agent set, prompt blocks, context sources, and stage parameters.
- Stage outputs persist in dedicated relational tables for readability and inspection.

## Database philosophy
- SQL-first persistence only. CSV is not used as the core state layer.
- SQLAlchemy models are the system contract and should remain migration-friendly.
- Default runtime DB is SQLite for local frictionless startup.
- The schema is intentionally portable to PostgreSQL later via `DATABASE_URL`.

## Generation flow philosophy
- A generation flow is a reusable preset/workflow object.
- Flows support save/load/edit/duplicate/delete.
- Flow stages define execution order and stage-specific config.
- Flows are the primary user entrypoint from home screen.

## Prompt block philosophy
- Prompt blocks are reusable and editable modules.
- Keep prompt blocks concise, inspectable, and versioned (increment on edit).
- Stage applicability is explicit via mapping table.
- Avoid giant hardcoded prompt strings; compose at runtime.

## Persona philosophy
- Personas are separate entities from prompt blocks.
- Agents reference personas.
- Runtime prompt composition merges:
  1) stage context
  2) selected prompt blocks
  3) persona
  4) agent override prompt

## Platform-level agent philosophy
- `platform_agent` helps configure the platform itself (flows, blocks, assignments).
- It should output structured config suggestions, not final creative content.
- It must stay aware of all stages, including future placeholders.

## Must-not-break rules
1. Keep Part A stages fully runnable.
2. Preserve full live conversation visibility (no truncation in UI display layer).
3. Keep flow save/load/edit path straightforward.
4. Do not reintroduce CSV as core persistence.
5. Do not couple personas and prompt blocks into one object.
6. Do not hardcode one fixed flow as the only workflow.
7. Maintain future-stage-ready stage registry.

## Legacy supersession note
This repository previously used a legacy framework shape. That implementation was intentionally superseded. New work should follow this platform architecture instead of restoring the old assumptions.
