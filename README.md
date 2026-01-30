# AgentiCode

GitHub-Native SDLC Automation System that automates the software development lifecycle within GitHub.

## Features

- **Code Agent**: Reads GitHub issues and generates PRs with code
- **Reviewer Agent**: Reviews PRs and approves/requests changes (posts all comments at once)
- **Event-Driven Architecture**: Agents coordinate via GitHub webhooks, not tight loops
- **Webhook Server**: Deployable server for real-time GitHub event handling
- **Multi-Provider LLM Support**: OpenAI, Grok (xAI), YandexGPT

## How It Works

The system uses an event-driven architecture where each agent responds to GitHub webhook events:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Event-Driven Flow                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. ISSUE EVENT (auto-generate label)                                   │
│     └──> Code Agent creates PR with generated code                      │
│                                                                         │
│  2. PULL REQUEST EVENT (opened/synchronize)                             │
│     └──> Reviewer Agent reviews PR                                      │
│         └──> Posts all comments at once                                 │
│         └──> Sets status: APPROVE or REQUEST_CHANGES                    │
│                                                                         │
│  3. PULL REQUEST REVIEW EVENT (changes_requested)                       │
│     └──> Code Agent iterates on feedback                                │
│         └──> Pushes new commit (triggers step 2 again)                  │
│                                                                         │
│  Loop continues until: APPROVED or MAX_ITERATIONS reached               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Install with dev dependencies
uv sync --dev
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

| Variable | Description |
|----------|-------------|
| `GITHUB_TOKEN` | GitHub PAT with repo access (fallback) |
| `CODE_AGENT_TOKEN` | PAT for Code Agent (creates branches, PRs) |
| `REVIEWER_AGENT_TOKEN` | PAT for Reviewer Agent (reviews, approves) |

> **Important:** `CODE_AGENT_TOKEN` and `REVIEWER_AGENT_TOKEN` must be from **different GitHub identities** (different accounts or a GitHub App). GitHub doesn't allow users to approve their own PRs, so the reviewer must be a different identity than the code author.
| `GITHUB_REPOSITORY` | Repository in owner/repo format |
| `LLM_PROVIDER` | Provider: `openai`, `grok`, `yandex` |
| `LLM_MODEL` | Model to use (provider-specific) |
| `OPENAI_API_KEY` | OpenAI API key (if using openai) |
| `MAX_ITERATIONS` | Max code generation attempts (default: 5) |
| `MAX_REVIEW_ROUNDS` | Max review-fix cycles (default: 3) |
| `WEBHOOK_SECRET` | GitHub webhook secret (optional) |

## Usage

### Interaction Orchestrator (Recommended)

Run the full automation loop from an issue or PR:

```bash
# From issue: creates PR + runs review loop
uv run python -m agents.interaction_orchestrator --issue 123

# From existing PR: continues review loop
uv run python -m agents.interaction_orchestrator --pr 456

# Options
--max-review-rounds 5    # Maximum review-fix cycles
--output-json            # Output result as JSON
```

### Individual Agents

```bash
# Code Agent - generate code from issue
uv run python -m agents.code_agent --issue 123

# Code Agent - iterate on PR based on review feedback
uv run python -m agents.code_agent --pr 456

# Reviewer Agent - review a PR
uv run python -m agents.reviewer_agent --pr 456
```

### Webhook Server (Local)

Run the webhook server locally for development:

```bash
uv run python webhook_server.py
```

Endpoints:
- `POST /webhook` - GitHub webhook receiver
- `POST /trigger/issue/<number>` - Manual orchestration from issue
- `POST /trigger/pr/<number>` - Manual orchestration from PR
- `GET /health` - Health check

### Docker

```bash
# Run webhook server
docker-compose up webhook-server

# Run orchestrator for an issue
ISSUE_NUMBER=123 docker-compose run orchestrator

# Run individual agents
ISSUE_NUMBER=123 docker-compose run code-agent
PR_NUMBER=456 docker-compose run reviewer-agent
```

## Deployment

### DigitalOcean App Platform

1. Push code to GitHub

2. Create app via [DigitalOcean Console](https://cloud.digitalocean.com/apps):
   - Click **Create App** -> Select your GitHub repo
   - It will auto-detect the Dockerfile

3. Add environment variables in **Settings**:
   - `GITHUB_TOKEN` (secret)
   - `GITHUB_REPOSITORY` (e.g., `yourname/yourrepo`)
   - `OPENAI_API_KEY` (secret)
   - `WEBHOOK_SECRET` (secret, optional)

4. Configure GitHub webhook:
   - Go to your repo -> **Settings** -> **Webhooks** -> **Add webhook**
   - **Payload URL**: `https://your-app-url.ondigitalocean.app/webhook`
   - **Content type**: `application/json`
   - **Secret**: Same as `WEBHOOK_SECRET`
   - **Events**: Select individual events:
     - `Issues`
     - `Pull requests`
     - `Pull request reviews`

### Using doctl CLI

```bash
# Install and authenticate
brew install doctl
doctl auth init

# Edit .do/app.yaml with your repo
# Then create the app
doctl apps create --spec .do/app.yaml
```

## GitHub Actions

The system includes workflows for CI/CD integration:

1. **CI** (`ci.yaml`): Runs tests, linting, and type checking on push/PR
2. **Code Generation** (`on-issue.yaml`): Triggers on issues with `auto-generate` label
3. **PR Review** (`on-pr.yaml`): Runs orchestrator on PRs with `feat:` prefix or `auto-review` label

### Setup

1. Add required secrets to your repository:
   - `CODE_AGENT_TOKEN`
   - `REVIEWER_AGENT_TOKEN`
   - `OPENAI_API_KEY` (or other provider keys)

2. Optionally configure variables:
   - `LLM_PROVIDER`: Default `openai`
   - `LLM_MODEL`: Default `gpt-4o-mini`
   - `MAX_ITERATIONS`: Default `5`
   - `MAX_REVIEW_ROUNDS`: Default `3`

## Development

### Running Tests

```bash
uv run pytest tests/ -v
```

### Code Quality

```bash
# Format code
uv run black agents/ tests/

# Lint code
uv run ruff check agents/ tests/

# Type check
uv run mypy agents/
```

### Adding Dependencies

```bash
# Add a runtime dependency
uv add package-name

# Add a dev dependency
uv add --dev package-name
```

## Architecture

```
agenticode/
├── agents/
│   ├── code_agent.py           # Code generation from issues
│   ├── reviewer_agent.py       # PR review automation
│   ├── interaction_orchestrator.py  # Full review-fix loop
│   └── utils/
│       ├── github_client.py    # GitHub API wrapper
│       ├── github_app.py       # GitHub App integration
│       ├── llm_client.py       # Multi-provider LLM client
│       ├── code_formatter.py   # Black, Ruff, Mypy integration
│       └── prompts.py          # LLM prompt templates
├── webhook_server.py           # Flask webhook server
├── .github/workflows/          # GitHub Actions
├── .do/app.yaml               # DigitalOcean App Platform config
├── Dockerfile                  # Container image
├── docker-compose.yml          # Local development
├── src/                        # Generated application code
└── tests/                      # Test suite
```

## License

MIT
