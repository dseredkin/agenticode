FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_SYSTEM_PYTHON=1
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml .
RUN uv sync --no-dev

COPY . .

CMD ["uv", "run", "python", "-m", "agents.code_agent"]
