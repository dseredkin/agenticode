# AgentiCode

GitHub-Native SDLC Automation System - automates the software development lifecycle within GitHub.

## GitHub Apps

AgentiCode uses two separate GitHub Apps to maintain role isolation between the Code Agent and Reviewer Agent:

- **[agenticode-contributor](https://github.com/apps/agenticode-contributor)** - Used by the Code Agent to create branches, push code, and create pull requests
- **[agenticode-reviewer](https://github.com/apps/agenticode-reviewer)** - Used by the Reviewer Agent to review PRs and approve/request changes

Using separate GitHub Apps ensures that the Code Agent cannot approve its own PRs, maintaining proper separation of concerns in the review process.

### Installing the Apps

1. Visit the app pages above and click "Install"
2. Select the repositories you want to enable AgentiCode for
3. Configure the webhook URL to point to your AgentiCode server

## Features

- **Issue Moderator**: Classifies issues (bug/suggestion/documentation/question)
- **Code Agent**: Reads issues, generates code, creates PRs
- **Reviewer Agent**: Analyzes PRs, checks CI, publishes reviews
- **Iterative Cycle**: Automatic fixes based on reviewer feedback
- **Webhook Server**: Real-time GitHub event handling

## How Agents Work

### Issue Moderator
1. Fetches issue from GitHub
2. Checks if already moderated (skips if so)
3. Sends issue title/body to LLM for classification
4. Applies labels and posts a templated response

### Code Agent
1. Fetches issue details and repository structure
2. Finds relevant existing code for context (keyword-based search)
3. Sends context + issue to LLM to generate code
4. Validates generated code (syntax, linting, type checks)
5. If validation fails, retries with error feedback (up to `MAX_ITERATIONS`)
6. Creates PR when validation passes

### Reviewer Agent
1. Fetches PR details and linked issue (if any)
2. Waits for CI checks to complete
3. Sends diff + CI status + issue context to LLM for analysis
4. Posts review: `APPROVE` or `REQUEST_CHANGES` with specific feedback

When a review requests changes, the Code Agent automatically picks up the feedback and pushes fixes, restarting the review cycle.

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
git clone https://github.com/dseredkin/agenticode.git
cd agenticode

# 2. Configure environment
cp .env.example .env
# Edit .env with your settings (see Configuration section)

# 3. Start the service
docker-compose up -d

# Check logs
docker-compose logs -f
```

Service will be available at `http://localhost:8000`

### Local Setup

**Prerequisites:**
- Python 3.11+
- Redis (for task queue)
- [uv](https://github.com/astral-sh/uv) package manager

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and enter repository
git clone https://github.com/dseredkin/agenticode.git
cd agenticode

# 3. Install dependencies
uv sync

# 4. Start Redis (required for task queue)
# Using Docker:
docker run -d --name redis -p 6379:6379 redis:alpine
# Or install Redis locally and run: redis-server

# 5. Configure environment
cp .env.example .env
# Edit .env with your settings (see Configuration section)

# 6. Run the webhook server
uv run python webhook_server.py
```

The server will start at `http://localhost:8000`. For local development, you can use [ngrok](https://ngrok.com) or similar tools to expose the webhook endpoint to GitHub:

```bash
ngrok http 8000
```

Then configure the ngrok URL as your GitHub webhook endpoint.

## Configuration

Create `.env` file:

```bash
# GitHub Repository
GITHUB_REPOSITORY=owner/repo            # Repository in owner/repo format

# Authentication (choose one approach)

# Option 1: GitHub Apps (recommended)
# Install the apps and they handle authentication automatically via webhooks

# Option 2: Personal Access Tokens
CODE_AGENT_TOKEN=ghp_xxx                # PAT for Code Agent (creates PRs)
REVIEWER_AGENT_TOKEN=ghp_xxx            # PAT for Reviewer Agent (reviews PRs)
GITHUB_TOKEN=ghp_xxx                    # Fallback PAT (optional)

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

# Task Queue
REDIS_URL=redis://localhost:6379/0      # Redis connection URL

# Optional
MAX_ITERATIONS=5                        # Max code generation attempts
ITERATION_TIMEOUT=600                   # Seconds per iteration
WEBHOOK_PORT=8000                       # Webhook server port
WEBHOOK_SECRET=xxx                      # GitHub webhook secret

# Repository Traversal (Code Agent context)
REPO_STRUCTURE_LIMIT=100                # Max files in repo structure
FILES_TO_CHECK_LIMIT=30                 # Max files to scan for keywords
RELEVANT_FILES_LIMIT=10                 # Max relevant files as context
```

> **Important:** If using PATs, `CODE_AGENT_TOKEN` and `REVIEWER_AGENT_TOKEN` must be from **different GitHub accounts**. GitHub doesn't allow users to approve their own PRs. Using the GitHub Apps ([agenticode-contributor](https://github.com/apps/agenticode-contributor) and [agenticode-reviewer](https://github.com/apps/agenticode-reviewer)) is the recommended approach as they handle identity separation automatically.

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

## Validation & Metrics

Scripts for measuring AgentiCode performance:

```bash
# Create 20 test issues in a repository
uv run python validation_setup.py --repo owner/repo --token ghp_xxx

# Dry run (preview without creating)
uv run python validation_setup.py --repo owner/repo --token ghp_xxx --dry-run

# Calculate approval rate for validation issues
uv run python validation_metrics.py --repo owner/repo --token ghp_xxx

# Verbose output with details per issue
uv run python validation_metrics.py --repo owner/repo --token ghp_xxx -v

# Basic approval rate for all agent PRs
uv run python metrics.py --repo owner/repo --token ghp_xxx
```

**Approval Rate** = (Merged + Approved PRs) / Total Agent PRs

This metric shows how effectively the Code Agent resolves issues without manual intervention.

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
│   ├── task_queue.py           # Task queue (Huey/Redis)
│   └── utils/
│       ├── github_client.py    # GitHub API
│       ├── llm_client.py       # LLM client (OpenAI/Grok/Yandex)
│       ├── code_formatter.py   # Black, Ruff, Mypy
│       └── prompts.py          # LLM prompts
├── tests/                      # Tests (103 tests)
├── .github/workflows/          # GitHub Actions
├── webhook_server.py           # Flask webhook server
├── metrics.py                  # Basic approval rate calculation
├── validation_setup.py         # Create test issues for validation
├── validation_metrics.py       # Calculate metrics for validation issues
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
