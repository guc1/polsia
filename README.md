# Project Viral Agentic Framework

A local Python + Flask app for orchestrating multi-stage agent workflows that generate:
- Story elements
- Story format types
- Headlines
- Hooks
- Story plans
- Full stories
- Short scripts
- Video text assets (opening text + captions)

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

Recommended starter model in the UI is `openai/gpt-4o-mini`.

## Output structure

- CSV metadata: `data/records.csv`
- Agent configuration: `data/agents.json`
- Saved UI settings: `data/saved_settings.json`
- Pipeline output per run: `output/<run_id>/`

## Notes

- You can edit agent personas/system pieces in the UI under **Agent Prompt Editor**.
- You can set model per stage and optional per-agent model overrides in the UI.
- You can inspect live event logs per run under **Runs & Live Agent Conversations**.
