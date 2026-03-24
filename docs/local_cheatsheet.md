# Local Repository Cheat Sheet

## 1) Clone and run

```bash
git clone <YOUR_REPO_URL>
cd polsia
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# add OPENROUTER_API_KEY to .env
python main.py
```

Open: `http://localhost:4000`

## 2) Daily git workflow

```bash
git checkout -b feat/your-branch-name
# edit files
git status
git add .
git commit -m "feat: describe your change"
git push -u origin feat/your-branch-name
```

## 3) Create a PR on GitHub

### Option A: GitHub UI

1. Push your branch.
2. Open your repository on GitHub.
3. Click **Compare & pull request**.
4. Fill title + description.
5. Create PR.

### Option B: GitHub CLI

```bash
gh auth login
gh pr create --fill
```

## 4) Sync latest main

```bash
git checkout main
git pull origin main
git checkout feat/your-branch-name
git rebase main
```
