# Local Developer / Operator Cheat Sheet

## Clone
```bash
git clone <your-repo-url>
cd polsia
```

## Install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run app locally
```bash
python main.py
```
Open `http://localhost:4000`.

## Database setup
Default uses SQLite at `data/platform.db` (auto-created). No Docker required for default setup.

### Optional PostgreSQL later
Set:
```bash
export DATABASE_URL='postgresql+psycopg://user:pass@localhost:5432/polsia'
```
Then start app; schema tables auto-create at startup.

## Docker usage
No Docker is required for local default mode.
If you want containerized PostgreSQL, use your preferred compose stack and set `DATABASE_URL` accordingly.

## Migrations
Current baseline uses SQLAlchemy `create_all` for frictionless bootstrapping.
If introducing Alembic later, create migration scripts before team rollout.

## Create a branch
```bash
git checkout -b feat/part-a-platform
```

## Commit
```bash
git add .
git commit -m "Rebuild platform around Part A SQL architecture"
```

## Open a PR
```bash
git push -u origin feat/part-a-platform
# then open PR in your Git host UI
```

## Run tests/checks
```bash
python -m compileall main.py app
python main.py
```

## Reset local database
```bash
rm -f data/platform.db
python main.py
```
