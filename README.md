# AgentiCode

GitHub-Native SDLC Automation System - automates the software development lifecycle within GitHub.

## Features

- **Issue Moderator**: Classifies issues (bug/suggestion/documentation/question)
- **Code Agent**: Reads issues, generates code, creates PRs
- **Reviewer Agent**: Analyzes PRs, checks CI, publishes reviews
- **Iterative Cycle**: Automatic fixes based on reviewer feedback
- **Webhook Server**: Real-time GitHub event handling

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SDLC Automation Flow                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. ISSUE CREATED                                                       │
│     └──> Issue Moderator (classifies)                                   │
│     └──> Code Agent (creates PR)                                        │
│                                                                         │
│  2. PR CREATED/UPDATED                                                  │
│     └──> CI (tests, lint, type check)                                   │
│     └──> Reviewer Agent (isolated from Code Agent)                      │
│         └──> APPROVE or REQUEST_CHANGES                                 │
│                                                                         │
│  3. REVIEW: REQUEST_CHANGES                                             │
│     └──> Code Agent (fixes based on feedback)                           │
│         └──> Push (returns to step 2)                                   │
│                                                                         │
│  Loop until: APPROVED or MAX_ITERATIONS reached                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Important:** Code Agent and Reviewer Agent run in separate workflows for role isolation.

## Quick Start

### Docker (recommended)

```bash
# 1. Clone repository
git clone https://github.com/your-repo/agenticode.git
cd agenticode

# 2. Configure environment
cp .env.example .env
# Edit .env with your settings

# 3. Start the service
docker-compose up -d

# Check logs
docker-compose logs -f
```

Service will be available at `http://localhost:8000`

### Local Setup

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Configure environment
cp .env.example .env
# Edit .env with your settings

# 4. Run webhook server
uv run python webhook_server.py
```

## Configuration

Create `.env` file:

```bash
# GitHub
GITHUB_TOKEN=ghp_xxx                    # PAT with repo access
CODE_AGENT_TOKEN=ghp_xxx                # PAT for Code Agent
REVIEWER_AGENT_TOKEN=ghp_xxx            # PAT for Reviewer Agent (different account!)
GITHUB_REPOSITORY=owner/repo

# LLM Provider (choose one)
LLM_PROVIDER=openai                     # openai, grok, yandex
LLM_MODEL=gpt-4o-mini

# OpenAI
OPENAI_API_KEY=sk-xxx

# Or Grok (xAI)
# GROK_API_KEY=xxx

# Or YandexGPT
# YANDEX_API_KEY=xxx
# YANDEX_FOLDER_ID=xxx

# Optional
MAX_ITERATIONS=5                        # Max generation iterations
MAX_REVIEW_ROUNDS=3                     # Max review cycles
WEBHOOK_SECRET=xxx                      # GitHub webhook secret
```

> **Important:** `CODE_AGENT_TOKEN` and `REVIEWER_AGENT_TOKEN` must be from **different GitHub accounts** or use a GitHub App. GitHub doesn't allow users to approve their own PRs.

## GitHub Actions Workflows

### 1. CI (`ci.yaml`)
- Trigger: push/PR to main
- Runs: black, ruff, mypy, pytest

### 2. Issue Processing (`on-issue.yaml`)
- Trigger: issue created
- Actions:
  1. Issue Moderator classifies issue
  2. For bug/suggestion/documentation - Code Agent creates PR

### 3. PR Review (`on-pr.yaml`)
- Trigger: PR created/updated with `feat:` prefix
- Actions:
  1. CI checks
  2. Reviewer Agent analyzes and posts review

### 4. Review Feedback (`on-review-feedback.yaml`)
- Trigger: review with REQUEST_CHANGES
- Actions: Code Agent fixes code based on feedback

### Setup Secrets

Add to repository settings:
- `CODE_AGENT_TOKEN` - PAT for Code Agent
- `REVIEWER_AGENT_TOKEN` - PAT for Reviewer Agent
- `OPENAI_API_KEY` (or other LLM provider key)

## CLI Tools

```bash
# Issue Moderator
uv run python -m agents.issue_moderator --issue 123 --output-json

# Code Agent - from issue
uv run python -m agents.code_agent --issue 123 --output-json

# Code Agent - PR iteration
uv run python -m agents.code_agent --pr 456 --output-json

# Reviewer Agent
uv run python -m agents.reviewer_agent --pr 456 --output-json

# Orchestrator (full cycle)
uv run python -m agents.interaction_orchestrator --issue 123 --output-json
```

## Webhook Endpoints

- `POST /webhook` - GitHub webhook receiver
- `GET /health` - Health check
- `GET /queue/stats` - Queue statistics
- `POST /trigger/issue/<n>/moderate` - Manual moderator trigger
- `POST /trigger/issue/<n>/generate` - Manual Code Agent trigger
- `POST /trigger/pr/<n>/review` - Manual Reviewer Agent trigger

## Development

### Tests

```bash
uv run pytest tests/ -v
```

### Linting

```bash
uv run black agents/ tests/
uv run ruff check agents/ tests/
uv run mypy agents/
```

## Project Structure

```
agenticode/
├── agents/
│   ├── code_agent.py           # Code generation from issues
│   ├── reviewer_agent.py       # Automated PR review
│   ├── issue_moderator.py      # Issue classification
│   ├── interaction_orchestrator.py  # Full review cycle
│   ├── task_queue.py           # Task queue (Huey/SQLite)
│   └── utils/
│       ├── github_client.py    # GitHub API
│       ├── llm_client.py       # LLM client (OpenAI/Grok/Yandex)
│       ├── code_formatter.py   # Black, Ruff, Mypy
│       └── prompts.py          # LLM prompts
├── tests/                      # Tests (103 tests)
├── .github/workflows/          # GitHub Actions
├── webhook_server.py           # Flask webhook server
├── Dockerfile                  # Container build
├── docker-compose.yml          # Docker Compose config
└── pyproject.toml              # Project dependencies
```

## Technical Requirements

- Python 3.11+
- LLM: GPT-4o-mini, YandexGPT, Grok
- GitHub: PyGithub
- Code quality: ruff, black, mypy, pytest
- CI/CD: GitHub Actions

## License

MIT
