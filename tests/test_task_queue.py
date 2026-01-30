"""Tests for Task Queue."""

from unittest.mock import MagicMock, patch

import pytest

from agents.task_queue import (
    QueueConfig,
    TaskQueueManager,
    TaskType,
    run_code_agent_issue,
    run_issue_moderator,
    run_reviewer_agent,
)


class TestQueueConfig:
    """Tests for QueueConfig dataclass."""

    def test_default_config(self, monkeypatch):
        """Test default configuration values."""
        monkeypatch.delenv("REDIS_URL", raising=False)
        config = QueueConfig()
        assert config.redis_url == "redis://localhost:6379"
        assert config.default_timeout == 600
        assert config.result_ttl == 3600
        assert config.failure_ttl == 86400
        assert config.max_retries == 3

    def test_config_from_env(self, monkeypatch):
        """Test configuration reads REDIS_URL from environment."""
        monkeypatch.setenv("REDIS_URL", "redis://from-env:6379")
        config = QueueConfig()
        assert config.redis_url == "redis://from-env:6379"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = QueueConfig(
            redis_url="redis://custom:6380",
            default_timeout=300,
        )
        assert config.redis_url == "redis://custom:6380"
        assert config.default_timeout == 300


class TestTaskType:
    """Tests for TaskType enum."""

    def test_task_types(self):
        """Test all task types exist."""
        assert TaskType.ISSUE_MODERATE.value == "issue_moderate"
        assert TaskType.ISSUE_GENERATE.value == "issue_generate"
        assert TaskType.PR_REVIEW.value == "pr_review"
        assert TaskType.PR_ITERATE.value == "pr_iterate"


class TestTaskFunctions:
    """Tests for task worker functions."""

    @patch("agents.issue_moderator.IssueModerator")
    def test_run_issue_moderator(self, mock_moderator_class):
        """Test issue moderator task function."""
        mock_moderator = MagicMock()
        mock_moderator.run.return_value = MagicMock(
            success=True,
            issue_number=123,
            error=None,
            classification=MagicMock(
                issue_type="bug",
                severity="minor",
                labels=["bug"],
            ),
        )
        mock_moderator_class.return_value = mock_moderator

        result = run_issue_moderator(123)

        assert result["success"]
        assert result["issue_number"] == 123
        assert result["classification"]["type"] == "bug"
        mock_moderator.run.assert_called_once_with(123)

    @patch("agents.code_agent.CodeAgent")
    def test_run_code_agent_issue(self, mock_agent_class):
        """Test code agent task function."""
        mock_agent = MagicMock()
        mock_agent.run.return_value = MagicMock(
            success=True,
            pr_number=42,
            error=None,
            iterations=[MagicMock()],
            final_files=[MagicMock(path="src/app.py")],
        )
        mock_agent_class.return_value = mock_agent

        result = run_code_agent_issue(10)

        assert result["success"]
        assert result["issue_number"] == 10
        assert result["pr_number"] == 42
        assert result["files"] == ["src/app.py"]
        mock_agent.run.assert_called_once_with(10)

    @patch("agents.reviewer_agent.ReviewerAgent")
    def test_run_reviewer_agent(self, mock_agent_class):
        """Test reviewer agent task function."""
        mock_agent = MagicMock()
        mock_agent.run.return_value = MagicMock(
            success=True,
            pr_number=55,
            error=None,
            decision=MagicMock(value="approve"),
        )
        mock_agent_class.return_value = mock_agent

        result = run_reviewer_agent(55)

        assert result["success"]
        assert result["pr_number"] == 55
        assert result["decision"] == "approve"
        mock_agent.run.assert_called_once_with(55)


class TestTaskQueueManager:
    """Tests for TaskQueueManager class."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis connection."""
        redis = MagicMock()
        redis.sismember.return_value = False
        return redis

    @pytest.fixture
    def mock_queue(self):
        """Create a mock RQ queue."""
        queue = MagicMock()
        queue.__len__ = MagicMock(return_value=5)
        queue.started_job_registry.count = 2
        queue.finished_job_registry.count = 100
        queue.failed_job_registry.count = 3
        queue.deferred_job_registry.count = 0
        return queue

    @pytest.fixture
    def manager(self, mock_redis, mock_queue):
        """Create TaskQueueManager with mocked dependencies."""
        with patch("agents.task_queue.get_redis_connection", return_value=mock_redis):
            with patch("agents.task_queue.Queue", return_value=mock_queue):
                manager = TaskQueueManager()
                manager._redis = mock_redis
                manager._queue = mock_queue
                return manager

    def test_get_job_id(self, manager):
        """Test job ID generation."""
        job_id = manager._get_job_id(TaskType.ISSUE_MODERATE, 123)
        assert job_id == "issue_moderate:123"

        job_id = manager._get_job_id(TaskType.PR_REVIEW, 456)
        assert job_id == "pr_review:456"

    def test_enqueue_issue_moderate(self, manager, mock_queue):
        """Test enqueueing issue moderation."""
        mock_job = MagicMock()
        mock_job.id = "issue_moderate:123"
        mock_queue.enqueue.return_value = mock_job

        job = manager.enqueue_issue_moderate(123)

        assert job is not None
        assert job.id == "issue_moderate:123"
        mock_queue.enqueue.assert_called_once()

    def test_enqueue_issue_moderate_deduplicate(self, manager, mock_redis):
        """Test deduplication prevents duplicate jobs."""
        mock_redis.sismember.return_value = True

        job = manager.enqueue_issue_moderate(123, deduplicate=True)

        assert job is None

    def test_enqueue_code_generation(self, manager, mock_queue):
        """Test enqueueing code generation."""
        mock_job = MagicMock()
        mock_job.id = "issue_generate:456"
        mock_queue.enqueue.return_value = mock_job

        job = manager.enqueue_code_generation(456)

        assert job is not None
        mock_queue.enqueue.assert_called_once()

    def test_enqueue_pr_review(self, manager, mock_queue):
        """Test enqueueing PR review."""
        mock_job = MagicMock()
        mock_job.id = "pr_review:789"
        mock_queue.enqueue.return_value = mock_job

        job = manager.enqueue_pr_review(789)

        assert job is not None
        mock_queue.enqueue.assert_called_once()

    def test_enqueue_pr_iteration(self, manager, mock_queue):
        """Test enqueueing PR iteration."""
        mock_job = MagicMock()
        mock_job.id = "pr_iterate:101"
        mock_queue.enqueue.return_value = mock_job

        job = manager.enqueue_pr_iteration(101)

        assert job is not None
        mock_queue.enqueue.assert_called_once()

    def test_get_queue_stats(self, manager):
        """Test getting queue statistics."""
        stats = manager.get_queue_stats()

        assert stats["queued"] == 5
        assert stats["started"] == 2
        assert stats["finished"] == 100
        assert stats["failed"] == 3
        assert stats["deferred"] == 0

    @patch("agents.task_queue.Job")
    def test_get_job_status(self, mock_job_class, manager, mock_redis):
        """Test getting job status."""
        mock_job = MagicMock()
        mock_job.id = "issue_moderate:123"
        mock_job.get_status.return_value = "finished"
        mock_job.result = {"success": True}
        mock_job.created_at = None
        mock_job.started_at = None
        mock_job.ended_at = None
        mock_job_class.fetch.return_value = mock_job

        status = manager.get_job_status("issue_moderate:123")

        assert status["id"] == "issue_moderate:123"
        assert status["status"] == "finished"
        assert status["result"]["success"]

    @patch("agents.task_queue.Job")
    def test_get_job_status_not_found(self, mock_job_class, manager):
        """Test getting status of non-existent job."""
        mock_job_class.fetch.side_effect = Exception("Job not found")

        status = manager.get_job_status("nonexistent")

        assert status is None

    def test_mark_processing(self, manager, mock_redis):
        """Test marking job as processing."""
        manager._mark_processing("test_job")

        mock_redis.sadd.assert_called_once()
        mock_redis.expire.assert_called_once()

    def test_unmark_processing(self, manager, mock_redis):
        """Test unmarking job from processing."""
        manager._unmark_processing("test_job")

        mock_redis.srem.assert_called_once()


class TestHighLoadScenario:
    """Tests simulating high load scenarios."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis connection."""
        redis = MagicMock()
        processing_set = set()

        def sismember(key, value):
            return value in processing_set

        def sadd(key, value):
            processing_set.add(value)

        def srem(key, value):
            processing_set.discard(value)

        redis.sismember.side_effect = sismember
        redis.sadd.side_effect = sadd
        redis.srem.side_effect = srem
        return redis

    @pytest.fixture
    def mock_queue(self):
        """Create a mock RQ queue."""
        queue = MagicMock()
        queue.__len__ = MagicMock(return_value=0)
        queue.started_job_registry.count = 0
        queue.finished_job_registry.count = 0
        queue.failed_job_registry.count = 0
        queue.deferred_job_registry.count = 0
        return queue

    def test_deduplicate_multiple_same_issues(self, mock_redis, mock_queue):
        """Test that same issue is not queued multiple times."""
        with patch("agents.task_queue.get_redis_connection", return_value=mock_redis):
            with patch("agents.task_queue.Queue", return_value=mock_queue):
                manager = TaskQueueManager()
                manager._redis = mock_redis
                manager._queue = mock_queue

                mock_job = MagicMock()
                mock_job.id = "issue_moderate:1"
                mock_queue.enqueue.return_value = mock_job

                job1 = manager.enqueue_issue_moderate(1)
                job2 = manager.enqueue_issue_moderate(1)
                job3 = manager.enqueue_issue_moderate(1)

                assert job1 is not None
                assert job2 is None
                assert job3 is None
                assert mock_queue.enqueue.call_count == 1

    def test_different_issues_queued_separately(self, mock_redis, mock_queue):
        """Test that different issues are queued separately."""
        with patch("agents.task_queue.get_redis_connection", return_value=mock_redis):
            with patch("agents.task_queue.Queue", return_value=mock_queue):
                manager = TaskQueueManager()
                manager._redis = mock_redis
                manager._queue = mock_queue

                def create_job(func, issue_num, **kwargs):
                    job = MagicMock()
                    job.id = kwargs.get("job_id", f"job_{issue_num}")
                    return job

                mock_queue.enqueue.side_effect = create_job

                jobs = [manager.enqueue_issue_moderate(i) for i in range(1, 6)]

                assert all(job is not None for job in jobs)
                assert mock_queue.enqueue.call_count == 5
