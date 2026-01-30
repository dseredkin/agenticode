#!/bin/bash
# Start local development server with optional GitHub webhook forwarding

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PORT="${WEBHOOK_PORT:-8000}"
REPO="${GITHUB_REPOSITORY:-}"

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -r, --repo OWNER/REPO    GitHub repository for webhook forwarding"
    echo "  -p, --port PORT          Server port (default: 8000)"
    echo "  -f, --forward            Enable GitHub webhook forwarding (requires gh CLI)"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                           # Start server only"
    echo "  $0 -f -r owner/repo          # Start server + forward webhooks"
    echo "  $0 --port 9000               # Start server on port 9000"
    echo ""
    echo "Manual triggers (while server is running):"
    echo "  curl -X POST http://localhost:$PORT/trigger/issue/1"
    echo "  curl -X POST http://localhost:$PORT/trigger/pr/2"
    echo "  curl -X POST http://localhost:$PORT/trigger/pr/2/iterate"
}

FORWARD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--repo)
            REPO="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -f|--forward)
            FORWARD=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

export WEBHOOK_PORT="$PORT"

cleanup() {
    echo ""
    echo "Shutting down..."
    if [[ -n "${FORWARD_PID:-}" ]]; then
        kill "$FORWARD_PID" 2>/dev/null || true
    fi
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "========================================"
echo "  AgentiCode Local Development Server"
echo "========================================"
echo ""

if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed"
    exit 1
fi

echo "Starting webhook server on port $PORT..."
uv run python webhook_server.py &
SERVER_PID=$!

sleep 2

if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Error: Server failed to start"
    exit 1
fi

echo "Server started (PID: $SERVER_PID)"

if [[ "$FORWARD" == true ]]; then
    if [[ -z "$REPO" ]]; then
        echo "Error: --repo is required when using --forward"
        cleanup
        exit 1
    fi

    if ! command -v gh &> /dev/null; then
        echo "Error: gh CLI is not installed"
        echo "Install with: brew install gh"
        cleanup
        exit 1
    fi

    echo ""
    echo "Starting GitHub webhook forwarding for $REPO..."
    echo "Events: issues, pull_request"
    echo ""

    gh webhook forward \
        --repo="$REPO" \
        --events=issues,pull_request \
        --url="http://localhost:$PORT/webhook" &
    FORWARD_PID=$!

    sleep 2

    if ! kill -0 "$FORWARD_PID" 2>/dev/null; then
        echo "Error: Webhook forwarding failed to start"
        echo "Make sure you're authenticated: gh auth login"
        cleanup
        exit 1
    fi

    echo "Webhook forwarding started (PID: $FORWARD_PID)"
fi

echo ""
echo "========================================"
echo "  Server is running!"
echo "========================================"
echo ""
echo "Endpoints:"
echo "  Health:    http://localhost:$PORT/health"
echo "  Webhook:   http://localhost:$PORT/webhook"
echo ""
echo "Manual triggers:"
echo "  Issue:     curl -X POST http://localhost:$PORT/trigger/issue/<number>"
echo "  PR Review: curl -X POST http://localhost:$PORT/trigger/pr/<number>"
echo "  PR Iterate: curl -X POST http://localhost:$PORT/trigger/pr/<number>/iterate"
echo ""
echo "Press Ctrl+C to stop"
echo ""

wait
