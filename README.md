# Project Viral Agentic Framework

A local Python + Flask app for a **stage-first** multi-agent workflow that generates:
- Elements
- Headline Formats
- Headlines
- Hooks
- Story plans
- Stories
- Short scripts
- Video headline text
- Captions

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # optional
python main.py
```

Then open `http://localhost:4000`.

## OpenRouter setup

Add this to `.env`:

```bash
OPENROUTER_API_KEY=your_key_here
# optional (defaults to https://openrouter.ai)
# OPENROUTER_BASE_URL=https://openrouter.ai
```

Without a key, the app runs in **dry-run mode** and still writes pipeline artifacts.

## New stage-first UX

- **Home screen** lists all stages as clickable cards.
- Clicking a stage opens a **dedicated stage workspace** with:
  - agent toggles/inspection,
  - loop explanation,
  - prompt editor,
  - run settings,
  - context selection,
  - stage CSV table view.
- Running from a stage only triggers that stage.
- After run click, you are moved to a **run screen** with:
  - left: raw logs,
  - right: conversation-style full prompt/output view.
- When run completes, CSV updates are **not** auto-written:
  - Confirm,
  - Edit and submit,
  - Cancel/reject.

## CSV philosophy

The project uses **simple per-stage CSV files** under `data/`.

Example for `element_generation.csv`:
- `id`
- `element_type`
- `name`
- `description`
- `reasoning_for_choosing`
- `created_at`

Other stages follow the same lightweight, readable approach.

## Output structure

- Stage CSV files: `data/<stage>.csv`
- Legacy run records: `data/records.csv`
- Agent configuration: `data/agents.json`
- Saved UI settings: `data/saved_settings.json`
- Pipeline output per run: `output/<run_id>/full_output.json`
