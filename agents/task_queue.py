"""Task queue for processing GitHub events with Huey and Redis."""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from huey import RedisHuey
from huey.api import Result
from huey.consumer import Consumer

logger = logging.getLogger(__name__)

_consumer_thread: threading.Thread | None = None
_consumer_started = threading.Event()

# Default Redis URL
DEFAULT_REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")


class TaskType(str, Enum):
    """Types of tasks that can be queued."""

    ISSUE_MODERATE = "issue_moderate"
    ISSUE_GENERATE = "issue_generate"
    PR_REVIEW = "pr_review"
    PR_ITERATE = "pr_iterate"


class QueueError(Exception):
    """Base exception for queue errors."""

    pass


class QueueConnectionError(QueueError):
    """Raised when queue connection fails."""

    pass


class QueueEnqueueError(QueueError):
    """Raised when enqueueing a job fails."""

    pass


@dataclass
class QueueConfig:
    """Configuration for the task queue."""

    redis_url: str = field(
        default_factory=lambda: os.environ.get("REDIS_URL", DEFAULT_REDIS_URL)
    )
    default_timeout: int = 600  # 10 minutes
    result_ttl: int = 3600  # 1 hour
    max_retries: int = 3


def get_huey(redis_url: str | None = None) -> RedisHuey:
    """Get or create the Huey instance.

    Args:
        redis_url: Redis connection URL.

    Returns:
        RedisHuey instance.
    """
    url = redis_url or DEFAULT_REDIS_URL
    return RedisHuey(url=url, immediate=False)


# Global huey instance for task registration
huey = get_huey()


def start_consumer_thread(workers: int = 48) -> None:
    """Start Huey consumer in a background thread.

    This allows running the task queue without a separate worker process.
    Call this once when your application starts.

    Args:
        workers: Number of worker threads.
    """
    global _consumer_thread

    if _consumer_started.is_set():
        logger.info("[Queue] Consumer already running")
        return

    # Clear any pending tasks from previous runs
    try:
        pending_count = len(huey.pending())
        if pending_count > 0:
            logger.info(
                f"[Queue] Clearing {pending_count} pending tasks from previous run"
            )
            huey.flush()
    except Exception as e:
        logger.warning(f"[Queue] Failed to clear pending tasks: {e}")

    def run_consumer() -> None:
        try:
            consumer = Consumer(
                huey,
                workers=workers,
                worker_type="thread",
                setup_signals=False,
            )
            _consumer_started.set()
            logger.info(f"[Queue] Starting embedded consumer with {workers} workers")
            consumer.run()
        except Exception as e:
            logger.error(f"[Queue] Consumer error: {e}")
            _consumer_started.clear()

    _consumer_thread = threading.Thread(target=run_consumer, daemon=True)
    _consumer_thread.start()

    _consumer_started.wait(timeout=5)
    if _consumer_started.is_set():
        logger.info("[Queue] Consumer started successfully")
    else:
        logger.warning("[Queue] Consumer may not have started properly")


@huey.task(retries=3, retry_delay=60)  # type: ignore[untyped-decorator]
def run_issue_moderator(
    issue_number: int,
    installation_id: int | None = None,
    repository: str | None = None,
) -> dict[str, Any]:
    """Task: Run issue moderator on an issue.

    After classification, automatically triggers code generation for bugs.

    Args:
        issue_number: GitHub issue number to moderate.
        installation_id: GitHub App installation ID (for multi-tenant support).
        repository: Repository in owner/repo format.

    Returns:
        Result dictionary with success status and classification.
    """
    from agents.issue_moderator import IssueModerator

    logger.info(
        f"[Queue] Running issue moderator for issue #{issue_number} "
        f"(installation={installation_id}, repo={repository})"
    )

    try:
        moderator = IssueModerator(
            installation_id=installation_id,
            repository=repository,
        )
        result = moderator.run(issue_number)

        if result.success and result.classification:
            issue_type = result.classification.issue_type
            logger.info(
                f"[Queue] Issue #{issue_number} classified as {issue_type}, "
                "triggering code generation"
            )
            run_code_agent_issue(
                issue_number,
                installation_id=installation_id,
                repository=repository,
            )

        return {
            "success": result.success,
            "issue_number": result.issue_number,
            "error": result.error,
            "classification": (
                {
                    "type": result.classification.issue_type,
                    "severity": result.classification.severity,
                    "labels": result.classification.labels,
                }
                if result.classification
                else None
            ),
        }
    except Exception as e:
        logger.error(f"[Queue] Issue moderator failed for #{issue_number}: {e}")
        return {
            "success": False,
            "issue_number": issue_number,
            "error": str(e),
            "classification": None,
        }


@huey.task(retries=3, retry_delay=60)  # type: ignore[untyped-decorator]
def run_code_agent_issue(
    issue_number: int,
    installation_id: int | None = None,
    repository: str | None = None,
) -> dict[str, Any]:
    """Task: Run code agent to generate code from an issue.

    Args:
        issue_number: GitHub issue number.
        installation_id: GitHub App installation ID (for multi-tenant support).
        repository: Repository in owner/repo format.

    Returns:
        Result dictionary with success status and PR number.
    """
    from agents.code_agent import CodeAgent

    logger.info(
        f"[Queue] Running code agent for issue #{issue_number} "
        f"(installation={installation_id}, repo={repository})"
    )

    try:
        agent = CodeAgent(
            installation_id=installation_id,
            repository=repository,
        )
        result = agent.run(issue_number)

        return {
            "success": result.success,
            "issue_number": issue_number,
            "pr_number": result.pr_number,
            "error": result.error,
            "iterations": len(result.iterations),
            "files": [f.path for f in result.final_files],
        }
    except Exception as e:
        logger.error(f"[Queue] Code agent failed for issue #{issue_number}: {e}")
        return {
            "success": False,
            "issue_number": issue_number,
            "pr_number": None,
            "error": str(e),
            "iterations": 0,
            "files": [],
        }


@huey.task(retries=3, retry_delay=60)  # type: ignore[untyped-decorator]
def run_code_agent_pr(
    pr_number: int,
    installation_id: int | None = None,
    repository: str | None = None,
) -> dict[str, Any]:
    """Task: Run code agent to iterate on PR feedback.

    Args:
        pr_number: GitHub PR number.
        installation_id: GitHub App installation ID (for multi-tenant support).
        repository: Repository in owner/repo format.

    Returns:
        Result dictionary with success status.
    """
    from agents.code_agent import CodeAgent

    logger.info(
        f"[Queue] Running code agent iteration for PR #{pr_number} "
        f"(installation={installation_id}, repo={repository})"
    )

    try:
        agent = CodeAgent(
            installation_id=installation_id,
            repository=repository,
        )
        result = agent.run_pr_iteration(pr_number)

        return {
            "success": result.success,
            "pr_number": pr_number,
            "error": result.error,
            "iterations": len(result.iterations),
        }
    except Exception as e:
        logger.error(f"[Queue] Code agent PR iteration failed for #{pr_number}: {e}")
        return {
            "success": False,
            "pr_number": pr_number,
            "error": str(e),
            "iterations": 0,
        }


@huey.task(retries=3, retry_delay=60)  # type: ignore[untyped-decorator]
def run_reviewer_agent(
    pr_number: int,
    installation_id: int | None = None,
    repository: str | None = None,
) -> dict[str, Any]:
    """Task: Run reviewer agent on a PR.

    Args:
        pr_number: GitHub PR number.
        installation_id: GitHub App installation ID (for multi-tenant support).
        repository: Repository in owner/repo format.

    Returns:
        Result dictionary with review decision.
    """
    from agents.reviewer_agent import ReviewerAgent

    logger.info(
        f"[Queue] Running reviewer agent for PR #{pr_number} "
        f"(installation={installation_id}, repo={repository})"
    )

    try:
        agent = ReviewerAgent(
            installation_id=installation_id,
            repository=repository,
        )
        result = agent.run(pr_number)

        return {
            "success": result.success,
            "pr_number": pr_number,
            "error": result.error,
            "decision": result.decision.status if result.decision else None,
        }
    except Exception as e:
        logger.error(f"[Queue] Reviewer agent failed for PR #{pr_number}: {e}")
        return {
            "success": False,
            "pr_number": pr_number,
            "error": str(e),
            "decision": None,
        }


class TaskQueueManager:
    """Manager for enqueueing and tracking tasks."""

    def __init__(
        self,
        redis_url: str | None = None,
        config: QueueConfig | None = None,
    ) -> None:
        """Initialize task queue manager.

        Args:
            redis_url: Redis connection URL.
            config: Queue configuration.
        """
        self._config = config or QueueConfig()
        self._redis_url = redis_url or self._config.redis_url
        self._huey = get_huey(self._redis_url)
        self._processing: set[str] = set()

    def _get_task_id(self, task_type: TaskType, identifier: int) -> str:
        """Generate unique task ID for deduplication."""
        return f"{task_type.value}:{identifier}"

    def _is_processing(self, task_id: str) -> bool:
        """Check if a task is already processing or queued."""
        if task_id in self._processing:
            return True

        try:
            pending = self._huey.pending()
            for task in pending:
                if task.id == task_id:
                    return True
        except Exception:
            pass

        return False

    def _mark_processing(self, task_id: str) -> None:
        """Mark a task as processing."""
        self._processing.add(task_id)

    def _unmark_processing(self, task_id: str) -> None:
        """Remove task from processing set."""
        self._processing.discard(task_id)

    def enqueue_issue_moderate(
        self,
        issue_number: int,
        deduplicate: bool = True,
        installation_id: int | None = None,
        repository: str | None = None,
    ) -> Result[Any] | None:
        """Enqueue issue moderation task.

        Args:
            issue_number: GitHub issue number.
            deduplicate: Skip if already queued/processing.
            installation_id: GitHub App installation ID (for multi-tenant support).
            repository: Repository in owner/repo format.

        Returns:
            Result instance or None if deduplicated.

        Raises:
            QueueEnqueueError: If enqueueing fails.
        """
        task_id = self._get_task_id(TaskType.ISSUE_MODERATE, issue_number)

        if deduplicate and self._is_processing(task_id):
            logger.info(f"[Queue] Issue #{issue_number} already processing")
            return None

        try:
            self._mark_processing(task_id)
            result = run_issue_moderator(
                issue_number,
                installation_id=installation_id,
                repository=repository,
            )
            logger.info(f"[Queue] Enqueued issue moderation for #{issue_number}")
            return result

        except Exception as e:
            self._unmark_processing(task_id)
            logger.error(f"Failed to enqueue issue moderation: {e}")
            raise QueueEnqueueError(f"Failed to enqueue: {e}") from e

    def enqueue_code_generation(
        self,
        issue_number: int,
        deduplicate: bool = True,
        installation_id: int | None = None,
        repository: str | None = None,
    ) -> Result[Any] | None:
        """Enqueue code generation task.

        Args:
            issue_number: GitHub issue number.
            deduplicate: Skip if already queued/processing.
            installation_id: GitHub App installation ID (for multi-tenant support).
            repository: Repository in owner/repo format.

        Returns:
            Result instance or None if deduplicated.

        Raises:
            QueueEnqueueError: If enqueueing fails.
        """
        task_id = self._get_task_id(TaskType.ISSUE_GENERATE, issue_number)

        if deduplicate and self._is_processing(task_id):
            logger.info(f"[Queue] Code gen #{issue_number} already processing")
            return None

        try:
            self._mark_processing(task_id)
            result = run_code_agent_issue(
                issue_number,
                installation_id=installation_id,
                repository=repository,
            )
            logger.info(f"[Queue] Enqueued code generation for issue #{issue_number}")
            return result

        except Exception as e:
            self._unmark_processing(task_id)
            logger.error(f"Failed to enqueue code generation: {e}")
            raise QueueEnqueueError(f"Failed to enqueue: {e}") from e

    def enqueue_pr_review(
        self,
        pr_number: int,
        deduplicate: bool = True,
        installation_id: int | None = None,
        repository: str | None = None,
    ) -> Result[Any] | None:
        """Enqueue PR review task.

        Args:
            pr_number: GitHub PR number.
            deduplicate: Skip if already queued/processing.
            installation_id: GitHub App installation ID (for multi-tenant support).
            repository: Repository in owner/repo format.

        Returns:
            Result instance or None if deduplicated.

        Raises:
            QueueEnqueueError: If enqueueing fails.
        """
        task_id = self._get_task_id(TaskType.PR_REVIEW, pr_number)

        if deduplicate and self._is_processing(task_id):
            logger.info(f"[Queue] PR #{pr_number} review already processing")
            return None

        try:
            self._mark_processing(task_id)
            result = run_reviewer_agent(
                pr_number,
                installation_id=installation_id,
                repository=repository,
            )
            logger.info(f"[Queue] Enqueued PR review for #{pr_number}")
            return result

        except Exception as e:
            self._unmark_processing(task_id)
            logger.error(f"Failed to enqueue PR review: {e}")
            raise QueueEnqueueError(f"Failed to enqueue: {e}") from e

    def enqueue_pr_iteration(
        self,
        pr_number: int,
        deduplicate: bool = True,
        installation_id: int | None = None,
        repository: str | None = None,
    ) -> Result[Any] | None:
        """Enqueue PR iteration task.

        Args:
            pr_number: GitHub PR number.
            deduplicate: Skip if already queued/processing.
            installation_id: GitHub App installation ID (for multi-tenant support).
            repository: Repository in owner/repo format.

        Returns:
            Result instance or None if deduplicated.

        Raises:
            QueueEnqueueError: If enqueueing fails.
        """
        task_id = self._get_task_id(TaskType.PR_ITERATE, pr_number)

        if deduplicate and self._is_processing(task_id):
            logger.info(f"[Queue] PR #{pr_number} iteration already processing")
            return None

        try:
            self._mark_processing(task_id)
            result = run_code_agent_pr(
                pr_number,
                installation_id=installation_id,
                repository=repository,
            )
            logger.info(f"[Queue] Enqueued PR iteration for #{pr_number}")
            return result

        except Exception as e:
            self._unmark_processing(task_id)
            logger.error(f"Failed to enqueue PR iteration: {e}")
            raise QueueEnqueueError(f"Failed to enqueue: {e}") from e

    def get_job_status(self, task_id: str) -> dict[str, Any] | None:
        """Get status of a task.

        Args:
            task_id: The task ID.

        Returns:
            Status dictionary or None if not found.
        """
        try:
            result = self._huey.result(task_id, preserve=True)
            if result is not None:
                return {
                    "id": task_id,
                    "status": "completed",
                    "result": result,
                }

            pending = self._huey.pending()
            for task in pending:
                if task.id == task_id:
                    return {
                        "id": task_id,
                        "status": "queued",
                        "result": None,
                    }

            return None
        except Exception:
            return None

    def get_queue_stats(self) -> dict[str, int]:
        """Get queue statistics.

        Returns:
            Dictionary with queue stats.
        """
        try:
            pending = self._huey.pending()
            return {
                "queued": len(list(pending)),
                "processing": len(self._processing),
            }
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {"queued": 0, "processing": 0}

    def is_healthy(self) -> bool:
        """Check if queue is healthy.

        Returns:
            True if Redis is accessible.
        """
        try:
            self._huey.pending()
            return True
        except Exception:
            return False
