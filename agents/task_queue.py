"""Task queue for processing GitHub events with Redis and RQ."""

import logging
import os
from dataclasses import dataclass
from enum import Enum

from redis import Redis
from rq import Queue
from rq.job import Job

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Types of tasks that can be queued."""

    ISSUE_MODERATE = "issue_moderate"
    ISSUE_GENERATE = "issue_generate"
    PR_REVIEW = "pr_review"
    PR_ITERATE = "pr_iterate"


@dataclass
class QueueConfig:
    """Configuration for the task queue."""

    redis_url: str = "redis://localhost:6379"
    default_timeout: int = 600  # 10 minutes
    result_ttl: int = 3600  # 1 hour
    failure_ttl: int = 86400  # 24 hours
    max_retries: int = 3


def get_redis_connection(url: str | None = None) -> Redis:
    """Get Redis connection from URL or environment."""
    redis_url = url or os.environ.get("REDIS_URL", "redis://localhost:6379")
    return Redis.from_url(redis_url)


def get_queue(name: str = "default", redis_conn: Redis | None = None) -> Queue:
    """Get or create a queue."""
    conn = redis_conn or get_redis_connection()
    return Queue(name, connection=conn)


def run_issue_moderator(issue_number: int) -> dict:
    """Task: Run issue moderator on an issue.

    Args:
        issue_number: GitHub issue number to moderate.

    Returns:
        Result dictionary with success status and classification.
    """
    from agents.issue_moderator import IssueModerator

    logger.info(f"[Queue] Running issue moderator for issue #{issue_number}")

    moderator = IssueModerator()
    result = moderator.run(issue_number)

    return {
        "success": result.success,
        "issue_number": result.issue_number,
        "error": result.error,
        "classification": {
            "type": result.classification.issue_type,
            "severity": result.classification.severity,
            "labels": result.classification.labels,
        }
        if result.classification
        else None,
    }


def run_code_agent_issue(issue_number: int) -> dict:
    """Task: Run code agent to generate code from an issue.

    Args:
        issue_number: GitHub issue number.

    Returns:
        Result dictionary with success status and PR number.
    """
    from agents.code_agent import CodeAgent

    logger.info(f"[Queue] Running code agent for issue #{issue_number}")

    agent = CodeAgent()
    result = agent.run(issue_number)

    return {
        "success": result.success,
        "issue_number": issue_number,
        "pr_number": result.pr_number,
        "error": result.error,
        "iterations": len(result.iterations),
        "files": [f.path for f in result.final_files],
    }


def run_code_agent_pr(pr_number: int) -> dict:
    """Task: Run code agent to iterate on PR feedback.

    Args:
        pr_number: GitHub PR number.

    Returns:
        Result dictionary with success status.
    """
    from agents.code_agent import CodeAgent

    logger.info(f"[Queue] Running code agent iteration for PR #{pr_number}")

    agent = CodeAgent()
    result = agent.run_pr_iteration(pr_number)

    return {
        "success": result.success,
        "pr_number": pr_number,
        "error": result.error,
        "iterations": len(result.iterations),
    }


def run_reviewer_agent(pr_number: int) -> dict:
    """Task: Run reviewer agent on a PR.

    Args:
        pr_number: GitHub PR number.

    Returns:
        Result dictionary with review decision.
    """
    from agents.reviewer_agent import ReviewerAgent

    logger.info(f"[Queue] Running reviewer agent for PR #{pr_number}")

    agent = ReviewerAgent()
    result = agent.run(pr_number)

    return {
        "success": result.success,
        "pr_number": pr_number,
        "error": result.error,
        "decision": result.decision.value if result.decision else None,
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
        self._redis = get_redis_connection(redis_url or self._config.redis_url)
        self._queue = Queue("agents", connection=self._redis)
        self._processing_key = "agenticode:processing"

    def _get_job_id(self, task_type: TaskType, identifier: int) -> str:
        """Generate unique job ID for deduplication."""
        return f"{task_type.value}:{identifier}"

    def _is_processing(self, job_id: str) -> bool:
        """Check if a job is already processing or queued."""
        if self._redis.sismember(self._processing_key, job_id):
            return True

        job = Job.fetch(job_id, connection=self._redis)
        if job and job.get_status() in ["queued", "started", "deferred"]:
            return True

        return False

    def _mark_processing(self, job_id: str, ttl: int = 3600) -> None:
        """Mark a job as processing with TTL."""
        self._redis.sadd(self._processing_key, job_id)
        self._redis.expire(self._processing_key, ttl)

    def _unmark_processing(self, job_id: str) -> None:
        """Remove job from processing set."""
        self._redis.srem(self._processing_key, job_id)

    def enqueue_issue_moderate(
        self,
        issue_number: int,
        deduplicate: bool = True,
    ) -> Job | None:
        """Enqueue issue moderation task.

        Args:
            issue_number: GitHub issue number.
            deduplicate: Skip if already queued/processing.

        Returns:
            Job instance or None if deduplicated.
        """
        job_id = self._get_job_id(TaskType.ISSUE_MODERATE, issue_number)

        if deduplicate:
            try:
                if self._is_processing(job_id):
                    logger.info(f"[Queue] Issue #{issue_number} already processing")
                    return None
            except Exception:
                pass

        self._mark_processing(job_id)

        job = self._queue.enqueue(
            run_issue_moderator,
            issue_number,
            job_id=job_id,
            job_timeout=self._config.default_timeout,
            result_ttl=self._config.result_ttl,
            failure_ttl=self._config.failure_ttl,
            on_success=lambda *args: self._unmark_processing(job_id),
            on_failure=lambda *args: self._unmark_processing(job_id),
        )

        logger.info(f"[Queue] Enqueued issue moderation for #{issue_number}")
        return job

    def enqueue_code_generation(
        self,
        issue_number: int,
        deduplicate: bool = True,
    ) -> Job | None:
        """Enqueue code generation task.

        Args:
            issue_number: GitHub issue number.
            deduplicate: Skip if already queued/processing.

        Returns:
            Job instance or None if deduplicated.
        """
        job_id = self._get_job_id(TaskType.ISSUE_GENERATE, issue_number)

        if deduplicate:
            try:
                if self._is_processing(job_id):
                    logger.info(f"[Queue] Code gen #{issue_number} already processing")
                    return None
            except Exception:
                pass

        self._mark_processing(job_id)

        job = self._queue.enqueue(
            run_code_agent_issue,
            issue_number,
            job_id=job_id,
            job_timeout=self._config.default_timeout,
            result_ttl=self._config.result_ttl,
            failure_ttl=self._config.failure_ttl,
            on_success=lambda *args: self._unmark_processing(job_id),
            on_failure=lambda *args: self._unmark_processing(job_id),
        )

        logger.info(f"[Queue] Enqueued code generation for issue #{issue_number}")
        return job

    def enqueue_pr_review(
        self,
        pr_number: int,
        deduplicate: bool = True,
    ) -> Job | None:
        """Enqueue PR review task.

        Args:
            pr_number: GitHub PR number.
            deduplicate: Skip if already queued/processing.

        Returns:
            Job instance or None if deduplicated.
        """
        job_id = self._get_job_id(TaskType.PR_REVIEW, pr_number)

        if deduplicate:
            try:
                if self._is_processing(job_id):
                    logger.info(f"[Queue] PR #{pr_number} review already processing")
                    return None
            except Exception:
                pass

        self._mark_processing(job_id)

        job = self._queue.enqueue(
            run_reviewer_agent,
            pr_number,
            job_id=job_id,
            job_timeout=self._config.default_timeout,
            result_ttl=self._config.result_ttl,
            failure_ttl=self._config.failure_ttl,
            on_success=lambda *args: self._unmark_processing(job_id),
            on_failure=lambda *args: self._unmark_processing(job_id),
        )

        logger.info(f"[Queue] Enqueued PR review for #{pr_number}")
        return job

    def enqueue_pr_iteration(
        self,
        pr_number: int,
        deduplicate: bool = True,
    ) -> Job | None:
        """Enqueue PR iteration task.

        Args:
            pr_number: GitHub PR number.
            deduplicate: Skip if already queued/processing.

        Returns:
            Job instance or None if deduplicated.
        """
        job_id = self._get_job_id(TaskType.PR_ITERATE, pr_number)

        if deduplicate:
            try:
                if self._is_processing(job_id):
                    logger.info(f"[Queue] PR #{pr_number} iteration already processing")
                    return None
            except Exception:
                pass

        self._mark_processing(job_id)

        job = self._queue.enqueue(
            run_code_agent_pr,
            pr_number,
            job_id=job_id,
            job_timeout=self._config.default_timeout,
            result_ttl=self._config.result_ttl,
            failure_ttl=self._config.failure_ttl,
            on_success=lambda *args: self._unmark_processing(job_id),
            on_failure=lambda *args: self._unmark_processing(job_id),
        )

        logger.info(f"[Queue] Enqueued PR iteration for #{pr_number}")
        return job

    def get_job_status(self, job_id: str) -> dict | None:
        """Get status of a job.

        Args:
            job_id: The job ID.

        Returns:
            Status dictionary or None if not found.
        """
        try:
            job = Job.fetch(job_id, connection=self._redis)
            return {
                "id": job.id,
                "status": job.get_status(),
                "result": job.result,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "ended_at": job.ended_at.isoformat() if job.ended_at else None,
            }
        except Exception:
            return None

    def get_queue_stats(self) -> dict:
        """Get queue statistics.

        Returns:
            Dictionary with queue stats.
        """
        return {
            "queued": len(self._queue),
            "started": self._queue.started_job_registry.count,
            "finished": self._queue.finished_job_registry.count,
            "failed": self._queue.failed_job_registry.count,
            "deferred": self._queue.deferred_job_registry.count,
        }
