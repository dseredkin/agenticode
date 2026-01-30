"""Tests for Task Queue with Huey/Redis."""

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
        assert "redis://" in config.redis_url
        assert config.default_timeout == 600
        assert config.result_ttl == 3600
        assert config.max_retries == 3

    def test_config_from_env(self, monkeypatch):
        """Test configuration reads REDIS_URL from environment."""
        monkeypatch.setenv("REDIS_URL", "redis://custom:6379/1")
        config = QueueConfig()
        assert config.redis_url == "redis://custom:6379/1"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = QueueConfig(
            redis_url="redis://localhost:6380/2",
            default_timeout=300,
        )
        assert config.redis_url == "redis://localhost:6380/2"
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

        result = run_issue_moderator.call_local(123)

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

        result = run_code_agent_issue.call_local(10)

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
            decision=MagicMock(status="approve"),
        )
        mock_agent_class.return_value = mock_agent

        result = run_reviewer_agent.call_local(55)

        assert result["success"]
        assert result["pr_number"] == 55
        assert result["decision"] == "approve"
        mock_agent.run.assert_called_once_with(55)


class TestTaskQueueManager:
    """Tests for TaskQueueManager class."""

    @pytest.fixture
    def redis_url(self):
        """Create a test Redis URL."""
        return "redis://localhost:6379/15"

    @pytest.fixture
    def manager(self, redis_url):
        """Create TaskQueueManager with test Redis URL."""
        return TaskQueueManager(redis_url=redis_url)

    def test_get_task_id(self, manager):
        """Test task ID generation."""
        task_id = manager._get_task_id(TaskType.ISSUE_MODERATE, 123)
        assert task_id == "issue_moderate:123"

        task_id = manager._get_task_id(TaskType.PR_REVIEW, 456)
        assert task_id == "pr_review:456"

    def test_mark_processing(self, manager):
        """Test marking task as processing."""
        task_id = "test_task:1"
        assert not manager._is_processing(task_id)

        manager._mark_processing(task_id)
        assert manager._is_processing(task_id)

    def test_unmark_processing(self, manager):
        """Test unmarking task from processing."""
        task_id = "test_task:2"
        manager._mark_processing(task_id)
        assert manager._is_processing(task_id)

        manager._unmark_processing(task_id)
        assert not manager._is_processing(task_id)

    def test_enqueue_issue_moderate(self, manager):
        """Test enqueueing issue moderation."""
        result = manager.enqueue_issue_moderate(123)

        assert result is not None

    def test_enqueue_issue_moderate_deduplicate(self, manager):
        """Test deduplication prevents duplicate tasks."""
        manager._mark_processing("issue_moderate:123")

        result = manager.enqueue_issue_moderate(123, deduplicate=True)

        assert result is None

    def test_enqueue_code_generation(self, manager):
        """Test enqueueing code generation."""
        result = manager.enqueue_code_generation(456)

        assert result is not None

    def test_enqueue_pr_review(self, manager):
        """Test enqueueing PR review."""
        result = manager.enqueue_pr_review(789)

        assert result is not None

    def test_enqueue_pr_iteration(self, manager):
        """Test enqueueing PR iteration."""
        result = manager.enqueue_pr_iteration(101)

        assert result is not None

    def test_get_queue_stats(self, manager):
        """Test getting queue statistics."""
        stats = manager.get_queue_stats()

        assert "queued" in stats
        assert "processing" in stats

    def test_is_healthy(self, manager):
        """Test health check."""
        assert manager.is_healthy()


class TestDeduplication:
    """Tests for task deduplication."""

    @pytest.fixture
    def redis_url(self):
        """Create a test Redis URL."""
        return "redis://localhost:6379/15"

    def test_deduplicate_multiple_same_issues(self, redis_url):
        """Test that same issue is not queued multiple times."""
        manager = TaskQueueManager(redis_url=redis_url)

        result1 = manager.enqueue_issue_moderate(1)
        result2 = manager.enqueue_issue_moderate(1)
        result3 = manager.enqueue_issue_moderate(1)

        assert result1 is not None
        assert result2 is None
        assert result3 is None

    def test_different_issues_queued_separately(self, redis_url):
        """Test that different issues are queued separately."""
        manager = TaskQueueManager(redis_url=redis_url)

        results = [manager.enqueue_issue_moderate(i) for i in range(1, 6)]

        assert all(result is not None for result in results)

    def test_same_issue_different_task_types(self, redis_url):
        """Test same issue can be queued for different task types."""
        manager = TaskQueueManager(redis_url=redis_url)

        result1 = manager.enqueue_issue_moderate(1)
        result2 = manager.enqueue_code_generation(1)

        assert result1 is not None
        assert result2 is not None
