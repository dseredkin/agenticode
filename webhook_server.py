"""Webhook server for GitHub events with SQLite queue."""

import hashlib
import hmac
import logging
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from agents.task_queue import (
    QueueConnectionError,
    QueueEnqueueError,
    QueueError,
    TaskQueueManager,
    start_consumer_thread,
)
from agents.utils.github_app import get_installation_id_for_repo, load_private_key

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Request size limit (1MB max payload)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1MB

# Rate limiting configuration
# Default: 60 requests/minute globally, 30/minute per installation
RATE_LIMIT_GLOBAL = os.environ.get("RATE_LIMIT_GLOBAL", "60/minute")
RATE_LIMIT_PER_INSTALLATION = os.environ.get("RATE_LIMIT_PER_INSTALLATION", "30/minute")


def get_installation_id_for_limit() -> str:
    """Extract installation ID from request for rate limiting."""
    try:
        payload = request.get_json(silent=True)
        if payload:
            installation = payload.get("installation", {})
            installation_id = installation.get("id")
            if installation_id:
                return str(installation_id)
    except Exception:
        pass
    # Fall back to IP address
    return get_remote_address()


limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[RATE_LIMIT_GLOBAL],
    storage_uri="memory://",
)


@app.errorhandler(429)
def rate_limit_exceeded(e):
    """Handle rate limit exceeded errors."""
    logger.warning(f"Rate limit exceeded: {e.description}")
    return jsonify({
        "error": "Rate limit exceeded",
        "message": str(e.description),
        "retry_after": e.description,
    }), 429


@app.errorhandler(413)
def request_too_large(e):
    """Handle request entity too large errors."""
    logger.warning("Request payload too large")
    return jsonify({
        "error": "Request too large",
        "message": "Payload exceeds 1MB limit",
    }), 413


# Start embedded task consumer (no separate worker needed)
_embed_worker = os.environ.get("ENABLE_EMBEDDED_WORKER", "true").lower()
if _embed_worker == "true":
    _num_workers = int(os.environ.get("QUEUE_WORKERS", "4"))
    start_consumer_thread(workers=_num_workers)

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


# Cache for contributor installation IDs per repository
_contributor_installation_cache: dict[str, int] = {}


def is_contributor_app_webhook(installation_id: int | None, repository: str | None) -> bool:
    """Check if the webhook is from the contributor app's installation.

    This prevents duplicate processing when both contributor and reviewer apps
    are installed on the same repository.

    Args:
        installation_id: Installation ID from webhook payload.
        repository: Repository in owner/repo format.

    Returns:
        True if this is the contributor app's installation, False otherwise.
    """
    if not installation_id or not repository:
        return True  # Allow processing if we can't verify

    # Check cache first
    if repository in _contributor_installation_cache:
        return _contributor_installation_cache[repository] == installation_id

    # Look up the contributor app's installation ID
    contributor_app_id = os.environ.get("GITHUB_APP_CONTRIBUTOR_ID")
    contributor_app_key_env = os.environ.get("GITHUB_APP_CONTRIBUTOR_PRIVATE_KEY")

    if not contributor_app_id or not contributor_app_key_env:
        return True  # Allow processing if credentials not configured

    try:
        if contributor_app_key_env.startswith("-----BEGIN"):
            contributor_app_key = contributor_app_key_env
        else:
            contributor_app_key = load_private_key(contributor_app_key_env)

        contributor_installation_id = get_installation_id_for_repo(
            contributor_app_id, contributor_app_key, repository
        )
        _contributor_installation_cache[repository] = contributor_installation_id

        return contributor_installation_id == installation_id
    except Exception as e:
        logger.warning(f"Failed to verify contributor installation: {e}")
        return True  # Allow processing on error


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
@limiter.limit(RATE_LIMIT_PER_INSTALLATION, key_func=get_installation_id_for_limit)
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
    except QueueConnectionError as e:
        logger.error(f"[{delivery_id}] Queue connection error: {e}")
        return (
            jsonify(
                {
                    "status": "error",
                    "error": "Queue unavailable",
                    "details": str(e),
                }
            ),
            503,
        )
    except QueueEnqueueError as e:
        logger.error(f"[{delivery_id}] Failed to enqueue: {e}")
        return (
            jsonify(
                {
                    "status": "error",
                    "error": "Failed to queue task",
                    "details": str(e),
                }
            ),
            503,
        )
    except Exception as e:
        logger.error(f"[{delivery_id}] Unexpected error: {e}")
        return (
            jsonify(
                {
                    "status": "error",
                    "error": "Internal error",
                    "details": str(e),
                }
            ),
            500,
        )


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
            return jsonify(
                {
                    "status": "queued",
                    "agent": "code_agent",
                    "job_id": job.id,
                }
            )
        return jsonify(
            {
                "status": "deduplicated",
                "agent": "code_agent",
                "reason": "already processing",
            }
        )

    if action == "opened":
        job = qm.enqueue_issue_moderate(
            issue_number,
            installation_id=installation_id,
            repository=repository,
        )
        if job:
            return jsonify(
                {
                    "status": "queued",
                    "agent": "issue_moderator",
                    "job_id": job.id,
                }
            )
        return jsonify(
            {
                "status": "deduplicated",
                "agent": "issue_moderator",
                "reason": "already processing",
            }
        )

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

    # Only process webhooks from the contributor app to avoid duplicates
    if not is_contributor_app_webhook(installation_id, repository):
        logger.info(f"[{delivery_id}] Ignoring PR webhook from non-contributor app")
        return jsonify({"status": "ignored", "reason": "not contributor app"})

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
        return jsonify(
            {
                "status": "queued",
                "agent": "reviewer_agent",
                "job_id": job.id,
            }
        )
    return jsonify(
        {
            "status": "deduplicated",
            "agent": "reviewer_agent",
            "reason": "already processing",
        }
    )


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

    # Only process webhooks from the contributor app to avoid duplicates
    if not is_contributor_app_webhook(installation_id, repository):
        logger.info(f"[{delivery_id}] Ignoring review webhook from non-contributor app")
        return jsonify({"status": "ignored", "reason": "not contributor app"})

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
        return jsonify(
            {
                "status": "queued",
                "agent": "code_agent",
                "job_id": job.id,
            }
        )
    return jsonify(
        {
            "status": "deduplicated",
            "agent": "code_agent",
            "reason": "already processing",
        }
    )


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
@limiter.limit("10/minute")
def trigger_issue_moderate(issue_number: int):
    """Manually trigger Issue Moderator to classify an issue."""
    try:
        qm = get_queue_manager()
        job = qm.enqueue_issue_moderate(issue_number, deduplicate=False)
        return jsonify(
            {
                "status": "queued",
                "agent": "issue_moderator",
                "issue": issue_number,
                "job_id": job.id if job else None,
            }
        )
    except QueueError as e:
        return (
            jsonify(
                {
                    "status": "error",
                    "error": "Queue unavailable",
                    "details": str(e),
                }
            ),
            503,
        )


@app.route("/trigger/issue/<int:issue_number>/generate", methods=["POST"])
@limiter.limit("10/minute")
def trigger_issue_generate(issue_number: int):
    """Manually trigger Code Agent to generate code from an issue."""
    try:
        qm = get_queue_manager()
        job = qm.enqueue_code_generation(issue_number, deduplicate=False)
        return jsonify(
            {
                "status": "queued",
                "agent": "code_agent",
                "issue": issue_number,
                "job_id": job.id if job else None,
            }
        )
    except QueueError as e:
        return (
            jsonify(
                {
                    "status": "error",
                    "error": "Queue unavailable",
                    "details": str(e),
                }
            ),
            503,
        )


@app.route("/trigger/pr/<int:pr_number>/review", methods=["POST"])
@limiter.limit("10/minute")
def trigger_pr_review(pr_number: int):
    """Manually trigger Reviewer Agent for a PR."""
    try:
        qm = get_queue_manager()
        job = qm.enqueue_pr_review(pr_number, deduplicate=False)
        return jsonify(
            {
                "status": "queued",
                "agent": "reviewer_agent",
                "pr": pr_number,
                "job_id": job.id if job else None,
            }
        )
    except QueueError as e:
        return (
            jsonify(
                {
                    "status": "error",
                    "error": "Queue unavailable",
                    "details": str(e),
                }
            ),
            503,
        )


@app.route("/trigger/pr/<int:pr_number>/iterate", methods=["POST"])
@limiter.limit("10/minute")
def trigger_pr_iteration(pr_number: int):
    """Manually trigger Code Agent to iterate on PR feedback."""
    try:
        qm = get_queue_manager()
        job = qm.enqueue_pr_iteration(pr_number, deduplicate=False)
        return jsonify(
            {
                "status": "queued",
                "agent": "code_agent",
                "pr": pr_number,
                "job_id": job.id if job else None,
            }
        )
    except QueueError as e:
        return (
            jsonify(
                {
                    "status": "error",
                    "error": "Queue unavailable",
                    "details": str(e),
                }
            ),
            503,
        )


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
