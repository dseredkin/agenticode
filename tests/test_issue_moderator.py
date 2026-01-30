"""Tests for Issue Moderator."""

from unittest.mock import MagicMock

import pytest

from agents.issue_moderator import (
    ClassificationResult,
    IssueModerator,
    ModerationResult,
    RESPONSE_TEMPLATES,
)
from agents.utils.github_client import IssueDetails


class TestClassificationResult:
    """Tests for ClassificationResult dataclass."""

    def test_classification_result_creation(self):
        """Test ClassificationResult creation."""
        result = ClassificationResult(
            issue_type="bug",
            severity="major",
            comment="Test comment",
            labels=["bug", "severity:major"],
        )
        assert result.issue_type == "bug"
        assert result.severity == "major"
        assert result.comment == "Test comment"
        assert "bug" in result.labels


class TestModerationResult:
    """Tests for ModerationResult dataclass."""

    def test_moderation_result_success(self):
        """Test successful moderation result."""
        classification = ClassificationResult(
            issue_type="question",
            severity="none",
            comment="Thanks for your question",
            labels=["question"],
        )
        result = ModerationResult(
            success=True,
            issue_number=123,
            classification=classification,
        )
        assert result.success
        assert result.issue_number == 123
        assert result.classification.issue_type == "question"

    def test_moderation_result_failure(self):
        """Test failed moderation result."""
        result = ModerationResult(
            success=False,
            issue_number=456,
            error="Failed to fetch issue",
        )
        assert not result.success
        assert "fetch issue" in result.error


class TestIssueModerator:
    """Tests for IssueModerator class."""

    @pytest.fixture
    def mock_github(self):
        """Create a mock GitHub client."""
        client = MagicMock()
        client.get_issue.return_value = IssueDetails(
            number=1,
            title="Test Issue",
            body="Test body",
            labels=[],
            state="open",
        )
        return client

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM client."""
        client = MagicMock()
        return client

    @pytest.fixture
    def moderator(self, mock_github, mock_llm):
        """Create an IssueModerator with mocked dependencies."""
        return IssueModerator(
            github_client=mock_github,
            llm_client=mock_llm,
        )

    def test_generate_labels_bug(self, moderator):
        """Test label generation for bugs."""
        labels = moderator._generate_labels("bug", "major")
        assert "bug" in labels
        assert "severity:major" in labels

    def test_generate_labels_bug_critical(self, moderator):
        """Test label generation for critical bugs."""
        labels = moderator._generate_labels("bug", "critical")
        assert "bug" in labels
        assert "severity:critical" in labels

    def test_generate_labels_question(self, moderator):
        """Test label generation for questions."""
        labels = moderator._generate_labels("question", "none")
        assert "question" in labels
        assert len(labels) == 1

    def test_generate_labels_documentation(self, moderator):
        """Test label generation for documentation issues."""
        labels = moderator._generate_labels("documentation", "none")
        assert "documentation" in labels

    def test_generate_labels_unknown(self, moderator):
        """Test label generation for unknown issues."""
        labels = moderator._generate_labels("unknown", "none")
        assert "needs-triage" in labels

    def test_generate_labels_suggestion_empty(self, moderator):
        """Test that suggestions get no labels."""
        labels = moderator._generate_labels("suggestion", "none")
        assert labels == []

    def test_parse_classification_valid_json(self, moderator):
        """Test parsing valid JSON classification."""
        response = '{"type": "bug", "severity": "minor", "comment": "Thanks for reporting!"}'
        result = moderator._parse_classification(response)
        assert result.issue_type == "bug"
        assert result.severity == "minor"
        assert "reporting" in result.comment

    def test_parse_classification_json_with_text(self, moderator):
        """Test parsing JSON wrapped in text."""
        response = '''Here is the classification:
{"type": "question", "severity": "none", "comment": "Good question!"}
Let me know if you need more help.'''
        result = moderator._parse_classification(response)
        assert result.issue_type == "question"
        assert result.severity == "none"

    def test_parse_classification_invalid_type(self, moderator):
        """Test parsing with invalid issue type defaults to unknown."""
        response = '{"type": "invalid_type", "severity": "none", "comment": "Comment"}'
        result = moderator._parse_classification(response)
        assert result.issue_type == "unknown"

    def test_parse_classification_no_comment_uses_template(self, moderator):
        """Test that missing comment uses template."""
        response = '{"type": "bug", "severity": "major", "comment": ""}'
        result = moderator._parse_classification(response)
        assert result.comment == RESPONSE_TEMPLATES["bug"]

    def test_parse_classification_invalid_json(self, moderator):
        """Test parsing invalid JSON returns unknown."""
        response = "This is not valid JSON at all"
        result = moderator._parse_classification(response)
        assert result.issue_type == "unknown"
        assert "needs-triage" in result.labels

    def test_run_success(self, moderator, mock_github, mock_llm):
        """Test successful moderation run."""
        mock_llm.generate_code.return_value = (
            '{"type": "bug", "severity": "minor", "comment": "Thanks!"}'
        )

        result = moderator.run(1)

        assert result.success
        assert result.issue_number == 1
        assert result.classification.issue_type == "bug"
        mock_github.post_comment.assert_called_once()
        mock_github.add_label.assert_called()

    def test_run_suggestion_no_labels(self, moderator, mock_github, mock_llm):
        """Test that suggestions get comment but no labels."""
        mock_llm.generate_code.return_value = (
            '{"type": "suggestion", "severity": "none", "comment": "Thanks for your suggestion!"}'
        )

        result = moderator.run(1)

        assert result.success
        assert result.classification.issue_type == "suggestion"
        assert result.classification.labels == []
        mock_github.post_comment.assert_called_once()
        mock_github.add_label.assert_not_called()

    def test_run_issue_not_found(self, moderator, mock_github):
        """Test handling of missing issue."""
        mock_github.get_issue.side_effect = Exception("Issue not found")

        result = moderator.run(999)

        assert not result.success
        assert "Failed to fetch issue" in result.error

    def test_run_classification_failure(self, moderator, mock_llm):
        """Test handling of classification failure."""
        mock_llm.generate_code.side_effect = Exception("LLM error")

        result = moderator.run(1)

        assert not result.success
        assert "classify issue" in result.error

    def test_run_post_comment_failure(self, moderator, mock_github, mock_llm):
        """Test handling of comment posting failure."""
        mock_llm.generate_code.return_value = (
            '{"type": "bug", "severity": "minor", "comment": "Thanks!"}'
        )
        mock_github.post_comment.side_effect = Exception("Permission denied")

        result = moderator.run(1)

        assert not result.success
        assert result.classification is not None
        assert "post response" in result.error


class TestResponseTemplates:
    """Tests for response templates."""

    def test_bug_template_exists(self):
        """Test bug template exists and has content."""
        assert "bug" in RESPONSE_TEMPLATES
        assert "reproduce" in RESPONSE_TEMPLATES["bug"].lower()

    def test_suggestion_template_needs_discussion(self):
        """Test suggestion template mentions discussion."""
        assert "suggestion" in RESPONSE_TEMPLATES
        assert "discussion" in RESPONSE_TEMPLATES["suggestion"].lower()

    def test_question_template_exists(self):
        """Test question template exists."""
        assert "question" in RESPONSE_TEMPLATES
        assert "confusing" in RESPONSE_TEMPLATES["question"].lower()

    def test_documentation_template_exists(self):
        """Test documentation template exists."""
        assert "documentation" in RESPONSE_TEMPLATES
        assert "section" in RESPONSE_TEMPLATES["documentation"].lower()

    def test_unknown_template_exists(self):
        """Test unknown template exists."""
        assert "unknown" in RESPONSE_TEMPLATES
        assert "category" in RESPONSE_TEMPLATES["unknown"].lower()
