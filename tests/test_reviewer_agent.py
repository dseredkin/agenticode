"""Tests for Reviewer Agent."""

from unittest.mock import MagicMock, patch

import pytest

from agents.reviewer_agent import ReviewDecision, ReviewerAgent, ReviewResult
from agents.utils.github_client import CIStatus, IssueDetails, PRDetails


class TestReviewDecision:
    """Tests for ReviewDecision model."""

    def test_review_decision_approve(self):
        """Test approval decision."""
        decision = ReviewDecision(
            status="APPROVE",
            requirements_met=True,
            ci_passing=True,
            issues=[],
            suggestions=["Consider adding more tests"],
            summary="Code looks good",
        )
        assert decision.status == "APPROVE"
        assert decision.requirements_met

    def test_review_decision_request_changes(self):
        """Test request changes decision."""
        decision = ReviewDecision(
            status="REQUEST_CHANGES",
            requirements_met=False,
            ci_passing=False,
            issues=["Missing type hints", "Tests failing"],
            suggestions=[],
            summary="Several issues need to be fixed",
        )
        assert decision.status == "REQUEST_CHANGES"
        assert len(decision.issues) == 2


class TestReviewerAgent:
    """Tests for ReviewerAgent class."""

    @pytest.fixture
    def mock_github(self):
        """Create a mock GitHub client."""
        client = MagicMock()
        client.get_pr.return_value = PRDetails(
            number=42,
            title="feat: Add hello function",
            body="Closes #1\n\nAdds a hello function",
            state="open",
            head_branch="feature",
            base_branch="main",
            diff="+ def hello(): pass",
            changed_files=["src/hello.py"],
        )
        client.get_issue.return_value = IssueDetails(
            number=1,
            title="Implement hello function",
            body="Create a function that returns a greeting",
            labels=[],
            state="open",
        )
        client.get_ci_status.return_value = CIStatus(
            state="success",
            checks={"test": "success", "lint": "success"},
            failed_checks=[],
        )
        return client

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM client."""
        client = MagicMock()
        client.generate_code.return_value = '''{
    "status": "APPROVE",
    "requirements_met": true,
    "ci_passing": true,
    "issues": [],
    "suggestions": ["Consider adding docstring"],
    "summary": "Implementation looks correct"
}'''
        return client

    @pytest.fixture
    def agent(self, mock_github, mock_llm):
        """Create a ReviewerAgent with mocked dependencies."""
        return ReviewerAgent(
            github_client=mock_github,
            llm_client=mock_llm,
        )

    def test_run_approve(self, agent, mock_github):
        """Test approving a PR."""
        result = agent.run(42)

        assert result.success
        assert result.decision is not None
        assert result.decision.status == "APPROVE"
        mock_github.post_review.assert_called_once()

    def test_run_request_changes(self, agent, mock_github, mock_llm):
        """Test requesting changes on a PR."""
        mock_llm.generate_code.return_value = '''{
    "status": "REQUEST_CHANGES",
    "requirements_met": false,
    "ci_passing": true,
    "issues": ["Missing type hints"],
    "suggestions": [],
    "summary": "Please add type hints"
}'''

        result = agent.run(42)

        assert result.success
        assert result.decision.status == "REQUEST_CHANGES"

    def test_run_pr_not_found(self, agent, mock_github):
        """Test handling of missing PR."""
        mock_github.get_pr.side_effect = Exception("PR not found")

        result = agent.run(999)

        assert not result.success
        assert "Failed to fetch PR" in result.error

    def test_run_ci_failing(self, agent, mock_github, mock_llm):
        """Test handling of failing CI."""
        mock_github.get_ci_status.return_value = CIStatus(
            state="failure",
            checks={"test": "failure"},
            failed_checks=["test"],
        )

        mock_llm.generate_code.return_value = '''{
    "status": "APPROVE",
    "requirements_met": true,
    "ci_passing": false,
    "issues": [],
    "suggestions": [],
    "summary": "Code looks good"
}'''

        result = agent.run(42)

        assert result.success
        assert result.decision.status == "REQUEST_CHANGES"
        assert "CI checks are failing" in result.decision.issues

    def test_get_linked_issue_closes(self, agent, mock_github):
        """Test extracting linked issue with 'Closes' keyword."""
        pr = PRDetails(
            number=42,
            title="Test",
            body="Closes #123",
            state="open",
            head_branch="feature",
            base_branch="main",
            diff="",
            changed_files=[],
        )

        issue = agent._get_linked_issue(pr)

        assert issue is not None
        mock_github.get_issue.assert_called_with(123)

    def test_get_linked_issue_fixes(self, agent, mock_github):
        """Test extracting linked issue with 'Fixes' keyword."""
        pr = PRDetails(
            number=42,
            title="Test",
            body="Fixes #456",
            state="open",
            head_branch="feature",
            base_branch="main",
            diff="",
            changed_files=[],
        )

        issue = agent._get_linked_issue(pr)

        assert issue is not None
        mock_github.get_issue.assert_called_with(456)

    def test_get_linked_issue_not_found(self, agent, mock_github):
        """Test handling of missing linked issue."""
        pr = PRDetails(
            number=42,
            title="Test",
            body="No linked issue here",
            state="open",
            head_branch="feature",
            base_branch="main",
            diff="",
            changed_files=[],
        )

        issue = agent._get_linked_issue(pr)

        assert issue is None

    def test_parse_review_response_valid_json(self, agent):
        """Test parsing valid JSON response."""
        response = '''{
    "status": "APPROVE",
    "requirements_met": true,
    "ci_passing": true,
    "issues": [],
    "suggestions": ["Add tests"],
    "summary": "Good"
}'''
        ci_status = CIStatus(state="success", checks={}, failed_checks=[])

        decision = agent._parse_review_response(response, ci_status)

        assert decision.status == "APPROVE"
        assert decision.requirements_met

    def test_parse_review_response_invalid_json(self, agent):
        """Test parsing invalid JSON response."""
        response = "This is not JSON at all"
        ci_status = CIStatus(state="success", checks={}, failed_checks=[])

        decision = agent._parse_review_response(response, ci_status)

        assert decision.status == "REQUEST_CHANGES"
        assert "Could not parse review response" in decision.issues

    def test_parse_review_response_json_in_text(self, agent):
        """Test parsing JSON embedded in text."""
        response = '''Here is my review:

{
    "status": "APPROVE",
    "requirements_met": true,
    "ci_passing": true,
    "issues": [],
    "suggestions": [],
    "summary": "Looks good"
}

That's my analysis.'''
        ci_status = CIStatus(state="success", checks={}, failed_checks=[])

        decision = agent._parse_review_response(response, ci_status)

        assert decision.status == "APPROVE"

    def test_run_without_waiting_for_ci(self, agent, mock_github):
        """Test running review without waiting for CI."""
        result = agent.run(42, wait_for_ci=False)

        assert result.success
        mock_github.get_ci_status.assert_not_called()
