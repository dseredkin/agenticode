"""Webhook server for GitHub events - event-driven agent coordination."""

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
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "5"))


def verify_signature(payload: bytes, signature: str) -> bool:
    """Verify GitHub webhook signature."""
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
    """Run agent in background thread."""
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
    elif event == "pull_request_review":
        return handle_pr_review_event(payload, delivery_id)
    elif event == "ping":
        logger.info(f"[{delivery_id}] Ping received from GitHub")
        return jsonify({"status": "pong"})
    else:
        logger.info(f"[{delivery_id}] Ignoring event: {event}")
        return jsonify({"status": "ignored", "event": event})


def handle_issue_event(payload: dict, delivery_id: str):
    """Handle issue events.

    Flow:
    - Issue with 'auto-generate' label -> Code Agent creates PR
    - New issue without 'auto-generate' -> Issue Moderator classifies and responds
    """
    action = payload.get("action", "")
    issue = payload.get("issue", {})
    issue_number = issue.get("number")

    logger.info(f"[{delivery_id}] Issue #{issue_number}: {action}")

    if action not in ["opened", "labeled"]:
        return jsonify({"status": "ignored", "reason": f"action={action}"})

    labels = [label.get("name", "") for label in issue.get("labels", [])]

    # If auto-generate label, trigger Code Agent
    if "auto-generate" in labels:
        command = [
            sys.executable,
            "-m",
            "agents.code_agent",
            "--issue",
            str(issue_number),
            "--output-json",
        ]
        thread = Thread(
            target=run_agent,
            args=(command, "code_agent", delivery_id),
        )
        thread.start()
        return jsonify({"status": "processing", "agent": "code_agent"})

    # For new issues without auto-generate, run Issue Moderator
    if action == "opened":
        command = [
            sys.executable,
            "-m",
            "agents.issue_moderator",
            "--issue",
            str(issue_number),
            "--output-json",
        ]
        thread = Thread(
            target=run_agent,
            args=(command, "issue_moderator", delivery_id),
        )
        thread.start()
        return jsonify({"status": "processing", "agent": "issue_moderator"})

    return jsonify({"status": "ignored", "reason": "no matching trigger"})


def handle_pr_event(payload: dict, delivery_id: str):
    """Handle PR events - triggers Reviewer Agent to review.

    Flow: PR opened/updated with 'feat:' prefix -> Reviewer reviews
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

    # Check iteration count from labels
    iteration = get_iteration_from_labels(labels)
    if iteration >= MAX_ITERATIONS:
        logger.warning(f"[{delivery_id}] Max iterations reached for PR #{pr_number}")
        return jsonify({"status": "ignored", "reason": "max iterations reached"})

    command = [
        sys.executable,
        "-m",
        "agents.reviewer_agent",
        "--pr",
        str(pr_number),
        "--output-json",
    ]
    thread = Thread(
        target=run_agent,
        args=(command, "reviewer_agent", delivery_id),
    )
    thread.start()

    return jsonify({"status": "processing", "agent": "reviewer_agent"})


def handle_pr_review_event(payload: dict, delivery_id: str):
    """Handle PR review events - triggers Code Agent to iterate on feedback.

    Flow: Review with REQUEST_CHANGES -> Code Agent iterates
    """
    action = payload.get("action", "")
    review = payload.get("review", {})
    pr = payload.get("pull_request", {})
    pr_number = pr.get("number")
    review_state = review.get("state", "")
    reviewer = review.get("user", {}).get("login", "unknown")

    logger.info(f"[{delivery_id}] PR #{pr_number} review by {reviewer}: {review_state}")

    if action != "submitted":
        return jsonify({"status": "ignored", "reason": f"action={action}"})

    if review_state != "changes_requested":
        logger.info(f"[{delivery_id}] Review state is {review_state}, not iterating")
        return jsonify({"status": "ignored", "reason": f"state={review_state}"})

    title = pr.get("title", "")
    labels = [label.get("name", "") for label in pr.get("labels", [])]

    if not (title.startswith("feat:") or "auto-review" in labels):
        return jsonify({"status": "ignored", "reason": "not an auto-review PR"})

    # Check iteration count
    iteration = get_iteration_from_labels(labels)
    if iteration >= MAX_ITERATIONS:
        logger.warning(f"[{delivery_id}] Max iterations reached for PR #{pr_number}")
        return jsonify({"status": "ignored", "reason": "max iterations reached"})

    command = [
        sys.executable,
        "-m",
        "agents.code_agent",
        "--pr",
        str(pr_number),
        "--output-json",
    ]
    thread = Thread(
        target=run_agent,
        args=(command, "code_agent", delivery_id),
    )
    thread.start()

    return jsonify({"status": "processing", "agent": "code_agent"})


def get_iteration_from_labels(labels: list[str]) -> int:
    """Extract iteration count from PR labels."""
    for label in labels:
        if label.startswith("iteration-"):
            try:
                return int(label.split("-")[1])
            except (IndexError, ValueError):
                pass
    return 0


@app.route("/trigger/issue/<int:issue_number>/moderate", methods=["POST"])
def trigger_issue_moderate(issue_number: int):
    """Manually trigger Issue Moderator to classify an issue."""
    delivery_id = f"manual-moderate-{issue_number}"
    command = [
        sys.executable,
        "-m",
        "agents.issue_moderator",
        "--issue",
        str(issue_number),
        "--output-json",
    ]
    thread = Thread(
        target=run_agent,
        args=(command, "issue_moderator", delivery_id),
    )
    thread.start()
    return jsonify(
        {
            "status": "processing",
            "agent": "issue_moderator",
            "issue": issue_number,
        }
    )


@app.route("/trigger/issue/<int:issue_number>/generate", methods=["POST"])
def trigger_issue_generate(issue_number: int):
    """Manually trigger Code Agent to generate code from an issue."""
    delivery_id = f"manual-generate-{issue_number}"
    command = [
        sys.executable,
        "-m",
        "agents.code_agent",
        "--issue",
        str(issue_number),
        "--output-json",
    ]
    thread = Thread(
        target=run_agent,
        args=(command, "code_agent", delivery_id),
    )
    thread.start()
    return jsonify(
        {
            "status": "processing",
            "agent": "code_agent",
            "issue": issue_number,
        }
    )


@app.route("/trigger/pr/<int:pr_number>/review", methods=["POST"])
def trigger_pr_review(pr_number: int):
    """Manually trigger Reviewer Agent for a PR."""
    delivery_id = f"manual-review-{pr_number}"
    command = [
        sys.executable,
        "-m",
        "agents.reviewer_agent",
        "--pr",
        str(pr_number),
        "--output-json",
    ]
    thread = Thread(
        target=run_agent,
        args=(command, "reviewer_agent", delivery_id),
    )
    thread.start()
    return jsonify(
        {
            "status": "processing",
            "agent": "reviewer_agent",
            "pr": pr_number,
        }
    )


@app.route("/trigger/pr/<int:pr_number>/iterate", methods=["POST"])
def trigger_pr_iteration(pr_number: int):
    """Manually trigger Code Agent to iterate on PR feedback."""
    delivery_id = f"manual-iterate-{pr_number}"
    command = [
        sys.executable,
        "-m",
        "agents.code_agent",
        "--pr",
        str(pr_number),
        "--output-json",
    ]
    thread = Thread(
        target=run_agent,
        args=(command, "code_agent", delivery_id),
    )
    thread.start()
    return jsonify(
        {
            "status": "processing",
            "agent": "code_agent",
            "pr": pr_number,
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("WEBHOOK_PORT", "8000"))
    logger.info(f"Starting webhook server on port {port}")
    logger.info("Event-driven flow:")
    logger.info("  Issue (opened) -> Issue Moderator classifies & responds")
    logger.info("  Issue (auto-generate) -> Code Agent creates PR")
    logger.info("  PR (opened/sync) -> Reviewer Agent reviews")
    logger.info("  Review (changes_requested) -> Code Agent iterates")
    logger.info("")
    logger.info("Endpoints:")
    logger.info("  POST /webhook - GitHub webhook receiver")
    logger.info("  POST /trigger/issue/<n>/moderate - Issue Moderator")
    logger.info("  POST /trigger/issue/<n>/generate - Code Agent from issue")
    logger.info("  POST /trigger/pr/<n>/review - Reviewer Agent")
    logger.info("  POST /trigger/pr/<n>/iterate - Code Agent iterate")
    logger.info("  GET  /health - Health check")
    app.run(host="0.0.0.0", port=port, debug=False)
