"""Local webhook server for GitHub events."""

import hashlib
import hmac
import logging
import os
import subprocess
import sys
from threading import Thread

from dotenv import load_dotenv
from flask import Flask, jsonify, request

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")


def verify_signature(payload: bytes, signature: str) -> bool:
    """Verify GitHub webhook signature.

    Args:
        payload: Raw request body.
        signature: X-Hub-Signature-256 header value.

    Returns:
        True if signature is valid or no secret configured.
    """
    if not WEBHOOK_SECRET:
        return True

    if not signature:
        return False

    expected = (
        "sha256="
        + hmac.new(
            WEBHOOK_SECRET.encode(),
            payload,
            hashlib.sha256,
        ).hexdigest()
    )

    return hmac.compare_digest(expected, signature)


def run_agent(command: list[str], event_type: str, event_id: str) -> None:
    """Run agent in background thread.

    Args:
        command: Command to execute.
        event_type: GitHub event type for logging.
        event_id: Event identifier for logging.
    """
    logger.info(f"[{event_id}] Running: {' '.join(command)}")
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            logger.info(f"[{event_id}] {event_type} completed successfully")
            if result.stdout:
                logger.info(f"[{event_id}] Output: {result.stdout[:500]}")
        else:
            logger.error(f"[{event_id}] {event_type} failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.error(f"[{event_id}] {event_type} timed out after 600s")
    except Exception as e:
        logger.error(f"[{event_id}] {event_type} error: {e}")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


@app.route("/webhook", methods=["POST"])
def webhook():
    """Handle GitHub webhook events."""
    signature = request.headers.get("X-Hub-Signature-256", "")
    if not verify_signature(request.data, signature):
        logger.warning("Invalid webhook signature")
        return jsonify({"error": "Invalid signature"}), 401

    event = request.headers.get("X-GitHub-Event", "")
    delivery_id = request.headers.get("X-GitHub-Delivery", "unknown")

    try:
        payload = request.json
    except Exception as e:
        logger.error(f"Failed to parse payload: {e}")
        return jsonify({"error": "Invalid JSON"}), 400

    logger.info(f"[{delivery_id}] Received event: {event}")

    if event == "issues":
        return handle_issue_event(payload, delivery_id)
    elif event == "pull_request":
        return handle_pr_event(payload, delivery_id)
    elif event == "ping":
        logger.info(f"[{delivery_id}] Ping received from GitHub")
        return jsonify({"status": "pong"})
    else:
        logger.info(f"[{delivery_id}] Ignoring event: {event}")
        return jsonify({"status": "ignored", "event": event})


def handle_issue_event(payload: dict, delivery_id: str):
    """Handle issue events.

    Args:
        payload: GitHub event payload.
        delivery_id: Delivery ID for logging.
    """
    action = payload.get("action", "")
    issue = payload.get("issue", {})
    issue_number = issue.get("number")

    logger.info(f"[{delivery_id}] Issue #{issue_number}: {action}")

    if action not in ["opened", "labeled"]:
        return jsonify({"status": "ignored", "reason": f"action={action}"})

    labels = [label.get("name", "") for label in issue.get("labels", [])]

    if "auto-generate" in labels:
        # Use orchestrator for full loop: issue -> PR -> review -> iterate
        command = [
            sys.executable,
            "-m",
            "agents.interaction_orchestrator",
            "--issue",
            str(issue_number),
            "--output-json",
        ]
        thread = Thread(
            target=run_agent,
            args=(command, "interaction_orchestrator", delivery_id),
        )
        thread.start()
        return jsonify({"status": "processing", "agent": "interaction_orchestrator"})

    return jsonify({"status": "ignored", "reason": "no matching trigger"})


def handle_pr_event(payload: dict, delivery_id: str):
    """Handle pull request events.

    Args:
        payload: GitHub event payload.
        delivery_id: Delivery ID for logging.
    """
    action = payload.get("action", "")
    pr = payload.get("pull_request", {})
    pr_number = pr.get("number")
    title = pr.get("title", "")

    logger.info(f"[{delivery_id}] PR #{pr_number}: {action} - {title}")

    if action not in ["opened", "synchronize"]:
        return jsonify({"status": "ignored", "reason": f"action={action}"})

    labels = [label.get("name", "") for label in pr.get("labels", [])]

    if not (title.startswith("feat:") or "auto-review" in labels):
        return jsonify({"status": "ignored", "reason": "no auto-review trigger"})

    # Use orchestrator for full review loop
    command = [
        sys.executable,
        "-m",
        "agents.interaction_orchestrator",
        "--pr",
        str(pr_number),
        "--output-json",
    ]
    thread = Thread(
        target=run_agent,
        args=(command, "interaction_orchestrator", delivery_id),
    )
    thread.start()

    return jsonify({"status": "processing", "agent": "interaction_orchestrator"})


@app.route("/trigger/issue/<int:issue_number>", methods=["POST"])
def trigger_issue(issue_number: int):
    """Manually trigger full orchestration for an issue.

    Args:
        issue_number: GitHub issue number.
    """
    delivery_id = f"manual-issue-{issue_number}"
    command = [
        sys.executable,
        "-m",
        "agents.interaction_orchestrator",
        "--issue",
        str(issue_number),
        "--output-json",
    ]
    thread = Thread(
        target=run_agent,
        args=(command, "interaction_orchestrator", delivery_id),
    )
    thread.start()
    return jsonify(
        {
            "status": "processing",
            "agent": "interaction_orchestrator",
            "issue": issue_number,
        }
    )


@app.route("/trigger/pr/<int:pr_number>", methods=["POST"])
def trigger_pr_orchestration(pr_number: int):
    """Manually trigger full orchestration for a PR.

    Args:
        pr_number: GitHub PR number.
    """
    delivery_id = f"manual-pr-{pr_number}"
    command = [
        sys.executable,
        "-m",
        "agents.interaction_orchestrator",
        "--pr",
        str(pr_number),
        "--output-json",
    ]
    thread = Thread(
        target=run_agent,
        args=(command, "interaction_orchestrator", delivery_id),
    )
    thread.start()
    return jsonify(
        {
            "status": "processing",
            "agent": "interaction_orchestrator",
            "pr": pr_number,
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("WEBHOOK_PORT", "8000"))
    logger.info(f"Starting webhook server on port {port}")
    logger.info("Endpoints:")
    logger.info("  POST /webhook - GitHub webhook receiver")
    logger.info("  POST /trigger/issue/<number> - Manual orchestration from issue")
    logger.info("  POST /trigger/pr/<number> - Manual orchestration from PR")
    logger.info("  GET  /health - Health check")
    app.run(host="0.0.0.0", port=port, debug=False)
