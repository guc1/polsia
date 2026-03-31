# Polsia Platform (Part A Foundation)

This repository now contains a **rebuilt platform architecture** for AI-generated short-form storytime/commentary content.

## What is implemented now (Part A)
- Stage 1: **Elements**
- Stage 2: **Headline Formats**
- Stage 3: **Headlines**

## Core platform capabilities
- SQL database persistence (SQLite by default; SQLAlchemy-based for future PostgreSQL migration).
- First-class generation flows (create/load/edit/duplicate/delete).
- Separate entities for personas, agents, and prompt blocks.
- Runtime prompt composition (stage context + prompt blocks + persona + agent override).
- Live run monitor with:
  - left pane: raw events/logs
  - right pane: full agent conversation history
- Platform-level assistant endpoint to propose flows/settings from intent.

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Open: `http://localhost:4000`

## Tech decisions
- **Flask** for quick local-first integrated backend/UI iteration.
- **SQLAlchemy** ORM for structured relational modeling and future DB portability.
- **SQLite** local file database for low-friction development (`data/platform.db`).

## Legacy status
The old CSV-centric orchestration framework was intentionally superseded by this architecture.
