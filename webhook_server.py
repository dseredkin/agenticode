"""Webhook server for GitHub events with SQLite queue."""

import hashlib
import hmac
import logging
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request

from agents.task_queue import (
    QueueConnectionError,
    QueueEnqueueError,
    QueueError,
    TaskQueueManager,
    start_consumer_thread,
)
from agents.utils.installation_store import get_installation_store

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Start embedded task consumer (no separate worker needed)
_embed_worker = os.environ.get("ENABLE_EMBEDDED_WORKER", "true").lower()
if _embed_worker == "true":
    start_consumer_thread(workers=2)

WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "5"))

queue_manager: TaskQueueManager | None = None


def get_queue_manager() -> TaskQueueManager:
    """Get or create queue manager singleton."""
    global queue_manager
    if queue_manager is None:
        queue_manager = TaskQueueManager()
    return queue_manager


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


def extract_installation_context(payload: dict) -> tuple[int | None, str | None]:
    """Extract installation ID and repository from webhook payload.

    Args:
        payload: GitHub webhook payload.

    Returns:
        Tuple of (installation_id, repository_full_name).
    """
    installation = payload.get("installation", {})
    installation_id = installation.get("id")

    repository = payload.get("repository", {})
    repo_full_name = repository.get("full_name")

    return installation_id, repo_full_name


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    try:
        qm = get_queue_manager()
        if qm.is_healthy():
            stats = qm.get_queue_stats()
            return jsonify({"status": "ok", "queue": stats})
        return jsonify({"status": "degraded", "error": "Queue not healthy"}), 503
    except QueueConnectionError as e:
        return jsonify({"status": "degraded", "error": str(e)}), 503
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


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

    try:
        if event == "installation":
            return handle_installation_event(payload, delivery_id)
        elif event == "installation_repositories":
            return handle_installation_repositories_event(payload, delivery_id)
        elif event == "issues":
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
    except QueueConnectionError as e:
        logger.error(f"[{delivery_id}] Queue connection error: {e}")
        return jsonify({
            "status": "error",
            "error": "Queue unavailable",
            "details": str(e),
        }), 503
    except QueueEnqueueError as e:
        logger.error(f"[{delivery_id}] Failed to enqueue: {e}")
        return jsonify({
            "status": "error",
            "error": "Failed to queue task",
            "details": str(e),
        }), 503
    except Exception as e:
        logger.error(f"[{delivery_id}] Unexpected error: {e}")
        return jsonify({
            "status": "error",
            "error": "Internal error",
            "details": str(e),
        }), 500


def handle_installation_event(payload: dict, delivery_id: str):
    """Handle GitHub App installation events.

    Triggered when:
    - App is installed (created)
    - App is uninstalled (deleted)
    - App is suspended/unsuspended
    """
    action = payload.get("action", "")
    installation = payload.get("installation", {})
    installation_id = installation.get("id")
    account = installation.get("account", {})
    account_login = account.get("login", "")
    account_type = account.get("type", "")

    logger.info(
        f"[{delivery_id}] Installation {installation_id} ({account_login}): {action}"
    )

    store = get_installation_store()

    if action == "created":
        repositories = payload.get("repositories", [])
        repo_names = [repo.get("full_name") for repo in repositories]

        store.add_installation(
            installation_id=installation_id,
            account_login=account_login,
            account_type=account_type,
            repositories=repo_names,
        )

        return jsonify({
            "status": "created",
            "installation_id": installation_id,
            "account": account_login,
            "repositories": len(repo_names),
        })

    elif action == "deleted":
        store.remove_installation(installation_id)
        return jsonify({
            "status": "deleted",
            "installation_id": installation_id,
        })

    elif action == "suspend":
        store.suspend_installation(installation_id)
        return jsonify({
            "status": "suspended",
            "installation_id": installation_id,
        })

    elif action == "unsuspend":
        store.unsuspend_installation(installation_id)
        return jsonify({
            "status": "unsuspended",
            "installation_id": installation_id,
        })

    return jsonify({"status": "ignored", "action": action})


def handle_installation_repositories_event(payload: dict, delivery_id: str):
    """Handle repository additions/removals from an installation.

    Triggered when repositories are added to or removed from an installation.
    """
    action = payload.get("action", "")
    installation = payload.get("installation", {})
    installation_id = installation.get("id")

    logger.info(f"[{delivery_id}] Installation {installation_id} repos: {action}")

    store = get_installation_store()

    if action == "added":
        repositories = payload.get("repositories_added", [])
        repo_names = [repo.get("full_name") for repo in repositories]
        count = store.add_repositories(installation_id, repo_names)
        return jsonify({
            "status": "repos_added",
            "installation_id": installation_id,
            "count": count,
        })

    elif action == "removed":
        repositories = payload.get("repositories_removed", [])
        repo_names = [repo.get("full_name") for repo in repositories]
        count = store.remove_repositories(installation_id, repo_names)
        return jsonify({
            "status": "repos_removed",
            "installation_id": installation_id,
            "count": count,
        })

    return jsonify({"status": "ignored", "action": action})


def handle_issue_event(payload: dict, delivery_id: str):
    """Handle issue events.

    Flow:
    - Issue with 'auto-generate' label -> Code Agent creates PR
    - New issue without 'auto-generate' -> Issue Moderator classifies
    """
    action = payload.get("action", "")
    issue = payload.get("issue", {})
    issue_number = issue.get("number")

    installation_id, repository = extract_installation_context(payload)

    logger.info(
        f"[{delivery_id}] Issue #{issue_number}: {action} "
        f"(installation={installation_id}, repo={repository})"
    )

    if action not in ["opened", "labeled"]:
        return jsonify({"status": "ignored", "reason": f"action={action}"})

    labels = [label.get("name", "") for label in issue.get("labels", [])]
    qm = get_queue_manager()

    if "auto-generate" in labels:
        job = qm.enqueue_code_generation(
            issue_number,
            installation_id=installation_id,
            repository=repository,
        )
        if job:
            return jsonify({
                "status": "queued",
                "agent": "code_agent",
                "job_id": job.id,
            })
        return jsonify({
            "status": "deduplicated",
            "agent": "code_agent",
            "reason": "already processing",
        })

    if action == "opened":
        job = qm.enqueue_issue_moderate(
            issue_number,
            installation_id=installation_id,
            repository=repository,
        )
        if job:
            return jsonify({
                "status": "queued",
                "agent": "issue_moderator",
                "job_id": job.id,
            })
        return jsonify({
            "status": "deduplicated",
            "agent": "issue_moderator",
            "reason": "already processing",
        })

    return jsonify({"status": "ignored", "reason": "no matching trigger"})


def handle_pr_event(payload: dict, delivery_id: str):
    """Handle PR events - triggers Reviewer Agent to review.

    Flow: PR opened/updated with 'feat:' prefix -> Reviewer reviews
    """
    action = payload.get("action", "")
    pr = payload.get("pull_request", {})
    pr_number = pr.get("number")
    title = pr.get("title", "")

    installation_id, repository = extract_installation_context(payload)

    logger.info(
        f"[{delivery_id}] PR #{pr_number}: {action} - {title} "
        f"(installation={installation_id}, repo={repository})"
    )

    if action not in ["opened", "synchronize"]:
        return jsonify({"status": "ignored", "reason": f"action={action}"})

    labels = [label.get("name", "") for label in pr.get("labels", [])]

    if not (title.startswith("feat:") or "auto-review" in labels):
        return jsonify({"status": "ignored", "reason": "no auto-review trigger"})

    iteration = get_iteration_from_labels(labels)
    if iteration >= MAX_ITERATIONS:
        logger.warning(f"[{delivery_id}] Max iterations reached for PR #{pr_number}")
        return jsonify({"status": "ignored", "reason": "max iterations reached"})

    qm = get_queue_manager()
    job = qm.enqueue_pr_review(
        pr_number,
        installation_id=installation_id,
        repository=repository,
    )

    if job:
        return jsonify({
            "status": "queued",
            "agent": "reviewer_agent",
            "job_id": job.id,
        })
    return jsonify({
        "status": "deduplicated",
        "agent": "reviewer_agent",
        "reason": "already processing",
    })


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

    installation_id, repository = extract_installation_context(payload)

    logger.info(
        f"[{delivery_id}] PR #{pr_number} review by {reviewer}: {review_state} "
        f"(installation={installation_id}, repo={repository})"
    )

    if action != "submitted":
        return jsonify({"status": "ignored", "reason": f"action={action}"})

    if review_state != "changes_requested":
        logger.info(f"[{delivery_id}] Review state is {review_state}, not iterating")
        return jsonify({"status": "ignored", "reason": f"state={review_state}"})

    title = pr.get("title", "")
    labels = [label.get("name", "") for label in pr.get("labels", [])]

    if not (title.startswith("feat:") or "auto-review" in labels):
        return jsonify({"status": "ignored", "reason": "not an auto-review PR"})

    iteration = get_iteration_from_labels(labels)
    if iteration >= MAX_ITERATIONS:
        logger.warning(f"[{delivery_id}] Max iterations reached for PR #{pr_number}")
        return jsonify({"status": "ignored", "reason": "max iterations reached"})

    qm = get_queue_manager()
    job = qm.enqueue_pr_iteration(
        pr_number,
        installation_id=installation_id,
        repository=repository,
    )

    if job:
        return jsonify({
            "status": "queued",
            "agent": "code_agent",
            "job_id": job.id,
        })
    return jsonify({
        "status": "deduplicated",
        "agent": "code_agent",
        "reason": "already processing",
    })


def get_iteration_from_labels(labels: list[str]) -> int:
    """Extract iteration count from PR labels."""
    for label in labels:
        if label.startswith("iteration-"):
            try:
                return int(label.split("-")[1])
            except (IndexError, ValueError):
                pass
    return 0


@app.route("/queue/stats", methods=["GET"])
def queue_stats():
    """Get queue statistics."""
    try:
        qm = get_queue_manager()
        return jsonify(qm.get_queue_stats())
    except QueueError as e:
        return jsonify({"error": "Queue unavailable", "details": str(e)}), 503
    except Exception as e:
        logger.error(f"Failed to get queue stats: {e}")
        return jsonify({"error": "Internal error"}), 500


@app.route("/queue/job/<job_id>", methods=["GET"])
def job_status(job_id: str):
    """Get status of a specific job."""
    try:
        qm = get_queue_manager()
        status = qm.get_job_status(job_id)
        if status:
            return jsonify(status)
        return jsonify({"error": "Job not found"}), 404
    except QueueError as e:
        return jsonify({"error": "Queue unavailable", "details": str(e)}), 503
    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        return jsonify({"error": "Internal error"}), 500


@app.route("/trigger/issue/<int:issue_number>/moderate", methods=["POST"])
def trigger_issue_moderate(issue_number: int):
    """Manually trigger Issue Moderator to classify an issue."""
    try:
        qm = get_queue_manager()
        job = qm.enqueue_issue_moderate(issue_number, deduplicate=False)
        return jsonify({
            "status": "queued",
            "agent": "issue_moderator",
            "issue": issue_number,
            "job_id": job.id if job else None,
        })
    except QueueError as e:
        return jsonify({
            "status": "error",
            "error": "Queue unavailable",
            "details": str(e),
        }), 503


@app.route("/trigger/issue/<int:issue_number>/generate", methods=["POST"])
def trigger_issue_generate(issue_number: int):
    """Manually trigger Code Agent to generate code from an issue."""
    try:
        qm = get_queue_manager()
        job = qm.enqueue_code_generation(issue_number, deduplicate=False)
        return jsonify({
            "status": "queued",
            "agent": "code_agent",
            "issue": issue_number,
            "job_id": job.id if job else None,
        })
    except QueueError as e:
        return jsonify({
            "status": "error",
            "error": "Queue unavailable",
            "details": str(e),
        }), 503


@app.route("/trigger/pr/<int:pr_number>/review", methods=["POST"])
def trigger_pr_review(pr_number: int):
    """Manually trigger Reviewer Agent for a PR."""
    try:
        qm = get_queue_manager()
        job = qm.enqueue_pr_review(pr_number, deduplicate=False)
        return jsonify({
            "status": "queued",
            "agent": "reviewer_agent",
            "pr": pr_number,
            "job_id": job.id if job else None,
        })
    except QueueError as e:
        return jsonify({
            "status": "error",
            "error": "Queue unavailable",
            "details": str(e),
        }), 503


@app.route("/trigger/pr/<int:pr_number>/iterate", methods=["POST"])
def trigger_pr_iteration(pr_number: int):
    """Manually trigger Code Agent to iterate on PR feedback."""
    try:
        qm = get_queue_manager()
        job = qm.enqueue_pr_iteration(pr_number, deduplicate=False)
        return jsonify({
            "status": "queued",
            "agent": "code_agent",
            "pr": pr_number,
            "job_id": job.id if job else None,
        })
    except QueueError as e:
        return jsonify({
            "status": "error",
            "error": "Queue unavailable",
            "details": str(e),
        }), 503


if __name__ == "__main__":
    port = int(os.environ.get("WEBHOOK_PORT", "8000"))
    logger.info(f"Starting webhook server on port {port}")
    logger.info("Event-driven flow with SQLite queue:")
    logger.info("  Issue (opened) -> Queue -> Issue Moderator classifies & responds")
    logger.info("  Issue (auto-generate) -> Queue -> Code Agent creates PR")
    logger.info("  PR (opened/sync) -> Queue -> Reviewer Agent reviews")
    logger.info("  Review (changes_requested) -> Queue -> Code Agent iterates")
    logger.info("")
    logger.info("Endpoints:")
    logger.info("  POST /webhook - GitHub webhook receiver")
    logger.info("  GET  /queue/stats - Queue statistics")
    logger.info("  GET  /queue/job/<id> - Job status")
    logger.info("  POST /trigger/issue/<n>/moderate - Issue Moderator")
    logger.info("  POST /trigger/issue/<n>/generate - Code Agent from issue")
    logger.info("  POST /trigger/pr/<n>/review - Reviewer Agent")
    logger.info("  POST /trigger/pr/<n>/iterate - Code Agent iterate")
    logger.info("  GET  /health - Health check")
    app.run(host="0.0.0.0", port=port, debug=False)
