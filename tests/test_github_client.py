"""Tests for GitHub client."""

from unittest.mock import MagicMock, patch

import pytest

from agents.utils.github_client import (
    CIStatus,
    FileChange,
    GitHubClient,
    IssueDetails,
    PRDetails,
)


@pytest.fixture
def mock_github():
    """Create a mock GitHub client."""
    with patch("agents.utils.github_client.Github") as mock:
        yield mock


@pytest.fixture
def github_client(mock_github):
    """Create a GitHubClient with mocked dependencies."""
    mock_repo = MagicMock()
    mock_github.return_value.get_repo.return_value = mock_repo

    with patch.dict(
        "os.environ",
        {"GITHUB_TOKEN": "test-token", "GITHUB_REPOSITORY": "owner/repo"},
    ):
        client = GitHubClient()
        return client


class TestGitHubClient:
    """Tests for GitHubClient class."""

    def test_init_requires_token(self):
        """Test that initialization requires a token."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="GitHub token is required"):
                GitHubClient()

    def test_init_requires_repository(self):
        """Test that initialization requires a repository."""
        with patch.dict("os.environ", {"GITHUB_TOKEN": "token"}, clear=True):
            with pytest.raises(ValueError, match="Repository is required"):
                GitHubClient()

    def test_get_issue(self, github_client):
        """Test fetching issue details."""
        mock_issue = MagicMock()
        mock_issue.number = 123
        mock_issue.title = "Test Issue"
        mock_issue.body = "Issue body"
        mock_issue.labels = []
        mock_issue.state = "open"

        github_client._repo.get_issue.return_value = mock_issue

        result = github_client.get_issue(123)

        assert isinstance(result, IssueDetails)
        assert result.number == 123
        assert result.title == "Test Issue"
        assert result.body == "Issue body"

    def test_get_file_content(self, github_client):
        """Test reading file content."""
        mock_content = MagicMock()
        mock_content.decoded_content = b"print('hello')"

        github_client._repo.get_contents.return_value = mock_content

        result = github_client.get_file_content("test.py")

        assert result == "print('hello')"

    def test_get_file_content_not_found(self, github_client):
        """Test reading non-existent file."""
        from github import GithubException

        github_client._repo.get_contents.side_effect = GithubException(
            404, {}, None
        )

        result = github_client.get_file_content("missing.py")

        assert result is None

    def test_create_branch(self, github_client):
        """Test creating a new branch."""
        mock_ref = MagicMock()
        mock_ref.object.sha = "abc123"

        github_client._repo.get_git_ref.return_value = mock_ref

        result = github_client.create_branch("feature-branch")

        assert result == "refs/heads/feature-branch"
        github_client._repo.create_git_ref.assert_called_once()

    def test_commit_files_create_new(self, github_client):
        """Test committing new files."""
        from github import GithubException

        github_client._repo.get_contents.side_effect = GithubException(
            404, {}, None
        )

        mock_ref = MagicMock()
        mock_ref.object.sha = "abc123"
        github_client._repo.get_git_ref.return_value = mock_ref

        files = [FileChange(path="new.py", content="print('new')")]
        github_client.commit_files(files, "Add new file", "main")

        github_client._repo.create_file.assert_called_once()

    def test_create_pr(self, github_client):
        """Test creating a pull request."""
        mock_pr = MagicMock()
        mock_pr.number = 42

        github_client._repo.create_pull.return_value = mock_pr

        result = github_client.create_pr(
            title="Test PR",
            body="PR description",
            head_branch="feature",
        )

        assert result == 42


class TestModels:
    """Tests for data models."""

    def test_issue_details(self):
        """Test IssueDetails model."""
        issue = IssueDetails(
            number=1,
            title="Test",
            body="Body",
            labels=["bug"],
            state="open",
        )
        assert issue.number == 1
        assert issue.labels == ["bug"]

    def test_pr_details(self):
        """Test PRDetails model."""
        pr = PRDetails(
            number=1,
            title="Test PR",
            body="Body",
            state="open",
            head_branch="feature",
            base_branch="main",
            diff="diff content",
            changed_files=["file.py"],
        )
        assert pr.number == 1
        assert pr.changed_files == ["file.py"]

    def test_ci_status(self):
        """Test CIStatus model."""
        status = CIStatus(
            state="failure",
            checks={"test": "failure", "lint": "success"},
            failed_checks=["test"],
        )
        assert status.state == "failure"
        assert "test" in status.failed_checks
