# AgentiCode

GitHub-Native SDLC Automation System that automates the software development lifecycle within GitHub.

## Features

- **Code Agent**: CLI tool that reads GitHub issues and generates PRs with code
- **Reviewer Agent**: Reviews PRs and approves/requests changes
- **Iteration Loop**: Automatic re-generation based on review feedback (max 5 iterations)
- **Multi-Provider LLM Support**: OpenAI, Grok (xAI), YandexGPT

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Required environment variables:

| Variable | Description |
|----------|-------------|
| `GITHUB_TOKEN` | GitHub PAT with repo access |
| `GITHUB_REPOSITORY` | Repository in owner/repo format |
| `LLM_PROVIDER` | Provider: `openai`, `grok`, `yandex` |
| `LLM_MODEL` | Model to use (provider-specific) |
| `OPENAI_API_KEY` | OpenAI API key (if using openai) |
| `GROK_API_KEY` | Grok/xAI API key (if using grok) |
| `YANDEX_API_KEY` | YandexGPT API key (if using yandex) |

## Usage

### Code Agent

Generate code from a GitHub issue:

```bash
python -m agents.code_agent --issue 123
```

Options:
- `--max-iterations`: Maximum number of generation attempts (default: 5)
- `--output-json`: Output result as JSON

### Reviewer Agent

Review a pull request:

```bash
python -m agents.reviewer_agent --pr 456
```

Options:
- `--no-wait-ci`: Don't wait for CI to complete
- `--output-json`: Output result as JSON

### Docker

```bash
# Run Code Agent
ISSUE_NUMBER=123 docker-compose run code-agent

# Run Reviewer Agent
PR_NUMBER=456 docker-compose run reviewer-agent
```

## GitHub Actions

The system includes three workflows:

1. **CI** (`ci.yaml`): Runs tests, linting, and type checking on push/PR
2. **Code Generation** (`on-issue.yaml`): Triggers on issues with `auto-generate` label
3. **PR Review** (`on-pr.yaml`): Reviews PRs with `feat:` prefix or `auto-review` label

### Setup

1. Add required secrets to your repository:
   - `OPENAI_API_KEY` (or other provider keys)

2. Optionally configure variables:
   - `LLM_PROVIDER`: Default `openai`
   - `LLM_MODEL`: Default `gpt-4o-mini`
   - `MAX_ITERATIONS`: Default `5`

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Quality

```bash
# Format code
black agents/ tests/

# Lint code
ruff check agents/ tests/

# Type check
mypy agents/
```

## Architecture

```
agenticode/
├── agents/
│   ├── code_agent.py      # Code generation from issues
│   ├── reviewer_agent.py  # PR review automation
│   └── utils/
│       ├── github_client.py   # GitHub API wrapper
│       ├── llm_client.py      # Multi-provider LLM client
│       ├── code_formatter.py  # Black, Ruff, Mypy integration
│       └── prompts.py         # LLM prompt templates
├── .github/workflows/     # GitHub Actions
├── src/                   # Generated application code
└── tests/                 # Test suite
```

## License

MIT
