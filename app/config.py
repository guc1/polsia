from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
AGENTS_FILE = DATA_DIR / "agents.json"
SETTINGS_FILE = DATA_DIR / "saved_settings.json"
RECORDS_DIR = DATA_DIR / "records"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RECORDS_DIR.mkdir(parents=True, exist_ok=True)
