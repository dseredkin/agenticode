"""GitHub API client wrapper."""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from github import Auth, Github, GithubException, InputGitTreeElement
from github.PullRequest import PullRequest
from github.Repository import Repository
from pydantic import BaseModel

from agents.utils.github_app import get_installation_token

logger = logging.getLogger(__name__)


class IssueDetails(BaseModel):
    """Issue details model."""

    number: int
    title: str
    body: str
    labels: list[str]
    state: str


class PRDetails(BaseModel):
    """Pull request details model."""

    number: int
    title: str
    body: str
    state: str
    head_branch: str
    base_branch: str
    diff: str
    changed_files: list[str]


class CIStatus(BaseModel):
    """CI status model."""

    state: str  # success, failure, pending
    checks: dict[str, str]  # check_name -> status
    failed_checks: list[str]


class ReviewComment(BaseModel):
    """Review comment model."""

    body: str
    path: str | None = None
    line: int | None = None


@dataclass
class FileChange:
    """Represents a file change to commit."""

    path: str
    content: str


class GitHubClient:
    """GitHub API client for interacting with repositories, issues, and PRs."""

    def __init__(
        self,
        token: str | None = None,
        repository: str | None = None,
        installation_id: int | None = None,
        app_id: str | None = None,
        app_private_key: str | None = None,
    ) -> None:
        """Initialize GitHub client.

        Args:
            token: GitHub personal access token. Defaults to GITHUB_TOKEN env var.
            repository: Repository in owner/repo format. Defaults to GITHUB_REPOSITORY env var.
            installation_id: GitHub App installation ID for installation-based auth.
                If provided with GitHub App credentials, generates installation token.
            app_id: GitHub App ID. Defaults to GITHUB_APP_ID env var.
            app_private_key: GitHub App private key. Defaults to GITHUB_APP_PRIVATE_KEY env var.
        """
        self._installation_id = installation_id
        self._repository = repository or os.environ.get("GITHUB_REPOSITORY", "")
        self._app_id = app_id
        self._app_private_key = app_private_key

        # Try installation-based auth first if installation_id provided
        if installation_id:
            self._token = self._get_installation_token(installation_id)
        else:
            self._token = token or os.environ.get("GITHUB_TOKEN", "")

        if not self._token:
            raise ValueError("GitHub token is required")
        if not self._repository:
            raise ValueError("Repository is required (format: owner/repo)")

        auth = Auth.Token(self._token)
        self._github = Github(auth=auth)
        self._repo: Repository = self._github.get_repo(self._repository)

    def _get_installation_token(self, installation_id: int) -> str:
        """Get installation access token for GitHub App.

        Args:
            installation_id: The installation ID.

        Returns:
            Installation access token.

        Raises:
            ValueError: If GitHub App credentials not configured.
        """
        app_id = self._app_id or os.environ.get("GITHUB_APP_ID")
        private_key_env = self._app_private_key or os.environ.get(
            "GITHUB_APP_PRIVATE_KEY"
        )

        if not app_id or not private_key_env:
            raise ValueError(
                "GitHub App credentials required for installation auth "
                "(GITHUB_APP_ID and GITHUB_APP_PRIVATE_KEY)"
            )

        # Handle key as path or content
        if private_key_env.startswith("-----BEGIN"):
            private_key = private_key_env
        else:
            from pathlib import Path

            private_key = Path(private_key_env).read_text()

        return get_installation_token(app_id, private_key, str(installation_id))

    @classmethod
    def from_installation(
        cls,
        installation_id: int,
        repository: str,
        app_id: str | None = None,
        app_private_key: str | None = None,
    ) -> "GitHubClient":
        """Create a GitHubClient for a specific installation and repository.

        Args:
            installation_id: GitHub App installation ID.
            repository: Repository in owner/repo format.
            app_id: GitHub App ID. Defaults to GITHUB_APP_ID env var.
            app_private_key: GitHub App private key. Defaults to GITHUB_APP_PRIVATE_KEY env var.

        Returns:
            GitHubClient instance configured for the installation.
        """
        return cls(
            installation_id=installation_id,
            repository=repository,
            app_id=app_id,
            app_private_key=app_private_key,
        )

    @property
    def installation_id(self) -> int | None:
        """Get the installation ID if using installation-based auth."""
        return self._installation_id

    @property
    def repository_name(self) -> str:
        """Get the repository name in owner/repo format."""
        return self._repository

    @property
    def repo(self) -> Repository:
        """Get the repository object."""
        return self._repo

    def get_issue(self, issue_number: int) -> IssueDetails:
        """Fetch issue details.

        Args:
            issue_number: The issue number to fetch.

        Returns:
            IssueDetails with issue information.
        """
        issue = self._repo.get_issue(issue_number)
        return IssueDetails(
            number=issue.number,
            title=issue.title,
            body=issue.body or "",
            labels=[label.name for label in issue.labels],
            state=issue.state,
        )

    def get_repo_structure(self, path: str = "") -> list[str]:
        """List files in the repository using Git Tree API (single API call).

        Args:
            path: Optional path prefix to filter files.

        Returns:
            List of file paths in the repository.
        """
        files: list[str] = []
        try:
            # Use Git Tree API with recursive=True for single API call
            default_branch = self._repo.default_branch
            branch = self._repo.get_branch(default_branch)
            tree = self._repo.get_git_tree(branch.commit.sha, recursive=True)

            for item in tree.tree:
                if item.type == "blob":  # Files only, not directories
                    if not path or item.path.startswith(path):
                        files.append(item.path)

            logger.debug(f"Fetched {len(files)} files from repo tree")
        except GithubException as e:
            logger.warning(f"Failed to get repo structure: {e}")
        return files

    def get_file_content(self, path: str, ref: str | None = None) -> str | None:
        """Read file content from the repository.

        Args:
            path: Path to the file.
            ref: Optional branch/commit ref.

        Returns:
            File content as string, or None if file doesn't exist.
        """
        try:
            if ref:
                content = self._repo.get_contents(path, ref=ref)
            else:
                content = self._repo.get_contents(path)
            if isinstance(content, list):
                return None
            return content.decoded_content.decode("utf-8")
        except GithubException:
            return None

    def create_branch(self, branch_name: str, base_branch: str | None = None) -> str:
        """Create a new branch.

        Args:
            branch_name: Name for the new branch.
            base_branch: Base branch to create from. If None, uses repo default.

        Returns:
            The full ref name of the created branch.
        """
        if base_branch is None:
            base_branch = self._repo.default_branch
        base_ref = self._repo.get_git_ref(f"heads/{base_branch}")
        ref_name = f"refs/heads/{branch_name}"

        try:
            self._repo.create_git_ref(ref_name, base_ref.object.sha)
            logger.info(f"Created branch: {branch_name}")
        except GithubException as e:
            if e.status == 422:  # Branch already exists
                logger.info(f"Branch already exists: {branch_name}")
            else:
                raise

        return ref_name

    def commit_files(
        self,
        files: list[FileChange],
        message: str,
        branch: str,
    ) -> str:
        """Commit file changes to a branch as a single atomic commit.

        Args:
            files: List of FileChange objects with path and content.
            message: Commit message.
            branch: Branch to commit to.

        Returns:
            The commit SHA.
        """
        ref = self._repo.get_git_ref(f"heads/{branch}")
        base_sha = ref.object.sha
        base_commit = self._repo.get_git_commit(base_sha)
        base_tree = base_commit.tree

        # Parallel blob creation for all files
        blob_shas: dict[str, str] = {}

        def create_blob(file_change: FileChange) -> tuple[str, str]:
            blob = self._repo.create_git_blob(file_change.content, "utf-8")
            return file_change.path, blob.sha

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_blob, fc) for fc in files]
            for future in as_completed(futures):
                path, sha = future.result()
                blob_shas[path] = sha
                logger.info(f"Staged file: {path}")

        # Build tree elements in original order
        tree_elements = []
        for file_change in files:
            element = InputGitTreeElement(
                path=file_change.path,
                mode="100644",
                type="blob",
                sha=blob_shas[file_change.path],
            )
            tree_elements.append(element)

        new_tree = self._repo.create_git_tree(tree_elements, base_tree)
        new_commit = self._repo.create_git_commit(message, new_tree, [base_commit])
        ref.edit(new_commit.sha)

        logger.info(f"Created commit {new_commit.sha[:7]} with {len(files)} files")
        return new_commit.sha

    def create_pr(
        self,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str | None = None,
    ) -> int:
        """Create a pull request.

        Args:
            title: PR title.
            body: PR description.
            head_branch: Source branch.
            base_branch: Target branch. If None, uses repo default.

        Returns:
            The PR number.
        """
        if base_branch is None:
            base_branch = self._repo.default_branch
        pr = self._repo.create_pull(
            title=title,
            body=body,
            head=head_branch,
            base=base_branch,
        )
        logger.info(f"Created PR #{pr.number}: {title}")
        return pr.number

    def get_pr(self, pr_number: int) -> PRDetails:
        """Get pull request details.

        Args:
            pr_number: The PR number.

        Returns:
            PRDetails with PR information.
        """
        pr = self._repo.get_pull(pr_number)
        diff = self._get_pr_diff(pr)
        changed_files = [f.filename for f in pr.get_files()]

        return PRDetails(
            number=pr.number,
            title=pr.title,
            body=pr.body or "",
            state=pr.state,
            head_branch=pr.head.ref,
            base_branch=pr.base.ref,
            diff=diff,
            changed_files=changed_files,
        )

    def _get_pr_diff(self, pr: PullRequest) -> str:
        """Get the diff for a pull request.

        Args:
            pr: The PullRequest object.

        Returns:
            The diff as a string.
        """
        files = pr.get_files()
        diff_parts: list[str] = []
        for f in files:
            diff_parts.append(f"--- a/{f.filename}")
            diff_parts.append(f"+++ b/{f.filename}")
            if f.patch:
                diff_parts.append(f.patch)
            diff_parts.append("")
        return "\n".join(diff_parts)

    def get_ci_status(
        self,
        pr_number: int,
        timeout: int = 300,
        poll_interval: int = 10,
    ) -> CIStatus:
        """Get CI workflow results for a PR.

        Args:
            pr_number: The PR number.
            timeout: Max time to wait for CI completion in seconds.
            poll_interval: Time between status checks in seconds.

        Returns:
            CIStatus with check results.
        """
        pr = self._repo.get_pull(pr_number)
        head_sha = pr.head.sha

        start_time = time.time()
        while time.time() - start_time < timeout:
            combined_status = self._repo.get_commit(head_sha).get_combined_status()
            check_runs = list(self._repo.get_commit(head_sha).get_check_runs())

            checks: dict[str, str] = {}
            failed_checks: list[str] = []

            for check in check_runs:
                checks[check.name] = check.conclusion or "pending"
                if check.conclusion and check.conclusion not in ("success", "skipped"):
                    failed_checks.append(check.name)

            # No CI configured - return success immediately
            if not check_runs and combined_status.total_count == 0:
                logger.info("No CI checks configured, skipping CI wait")
                return CIStatus(
                    state="success",
                    checks={},
                    failed_checks=[],
                )

            all_complete = (
                all(c.conclusion is not None for c in check_runs)
                if check_runs
                else combined_status.state != "pending"
            )

            if all_complete:
                state = "success" if not failed_checks else "failure"
                return CIStatus(
                    state=state,
                    checks=checks,
                    failed_checks=failed_checks,
                )

            logger.info(f"CI still running, waiting {poll_interval}s...")
            time.sleep(poll_interval)

        return CIStatus(
            state="pending",
            checks=checks if "checks" in dir() else {},
            failed_checks=[],
        )

    def post_review(
        self,
        pr_number: int,
        body: str,
        event: str = "COMMENT",
        comments: list[ReviewComment] | None = None,
    ) -> None:
        """Post a review on a pull request.

        Args:
            pr_number: The PR number.
            body: Review body text.
            event: Review event type (APPROVE, REQUEST_CHANGES, COMMENT).
            comments: Optional list of line comments.
        """
        pr = self._repo.get_pull(pr_number)

        review_comments: list[dict[str, Any]] = []
        if comments:
            for comment in comments:
                if comment.path and comment.line:
                    review_comments.append(
                        {
                            "path": comment.path,
                            "line": comment.line,
                            "body": comment.body,
                        }
                    )

        if review_comments:
            pr.create_review(body=body, event=event, comments=review_comments)  # type: ignore[arg-type]
        else:
            pr.create_review(body=body, event=event)

        logger.info(f"Posted {event} review on PR #{pr_number}")

    def add_label(self, issue_or_pr_number: int, label: str) -> None:
        """Add a label to an issue or PR.

        Args:
            issue_or_pr_number: The issue or PR number.
            label: Label name to add.
        """
        issue = self._repo.get_issue(issue_or_pr_number)
        issue.add_to_labels(label)

    def close_issue(self, issue_number: int) -> None:
        """Close an issue.

        Args:
            issue_number: The issue number to close.
        """
        issue = self._repo.get_issue(issue_number)
        issue.edit(state="closed")

    def link_pr_to_issue(self, pr_number: int, issue_number: int) -> None:
        """Update PR body to link to an issue.

        Args:
            pr_number: The PR number.
            issue_number: The issue number to link.
        """
        pr = self._repo.get_pull(pr_number)
        current_body = pr.body or ""
        if f"Closes #{issue_number}" not in current_body:
            new_body = f"{current_body}\n\nCloses #{issue_number}"
            pr.edit(body=new_body.strip())

    def get_pr_reviews(self, pr_number: int) -> list[dict[str, Any]]:
        """Get all reviews for a pull request.

        Args:
            pr_number: The PR number.

        Returns:
            List of review dictionaries with body, state, and user.
        """
        pr = self._repo.get_pull(pr_number)
        reviews: list[dict[str, Any]] = []
        for review in pr.get_reviews():
            reviews.append(
                {
                    "id": review.id,
                    "body": review.body or "",
                    "state": review.state,
                    "user": review.user.login if review.user else "unknown",
                    "submitted_at": (
                        review.submitted_at.isoformat() if review.submitted_at else None
                    ),
                }
            )
        return reviews

    def get_review_comments(self, pr_number: int) -> list[dict[str, Any]]:
        """Get all review comments for a pull request.

        Args:
            pr_number: The PR number.

        Returns:
            List of comment dictionaries with body, path, line, and user.
        """
        pr = self._repo.get_pull(pr_number)
        comments: list[dict[str, Any]] = []
        for comment in pr.get_review_comments():
            comments.append(
                {
                    "id": comment.id,
                    "body": comment.body,
                    "path": comment.path,
                    "line": comment.line,
                    "user": comment.user.login if comment.user else "unknown",
                }
            )
        return comments

    def get_pr_labels(self, pr_number: int) -> list[str]:
        """Get labels for a pull request.

        Args:
            pr_number: The PR number.

        Returns:
            List of label names.
        """
        issue = self._repo.get_issue(pr_number)
        return [label.name for label in issue.labels]

    def post_comment(self, issue_or_pr_number: int, body: str) -> None:
        """Post a comment on an issue or PR.

        Args:
            issue_or_pr_number: The issue or PR number.
            body: Comment body text.
        """
        issue = self._repo.get_issue(issue_or_pr_number)
        issue.create_comment(body)
        logger.info(f"Posted comment on #{issue_or_pr_number}")

    def get_issue_comments(self, issue_number: int) -> list[dict[str, Any]]:
        """Get all comments on an issue.

        Args:
            issue_number: The issue number.

        Returns:
            List of comment dictionaries with body and user.
        """
        issue = self._repo.get_issue(issue_number)
        comments: list[dict[str, Any]] = []
        for comment in issue.get_comments():
            comments.append(
                {
                    "id": comment.id,
                    "body": comment.body,
                    "user": comment.user.login if comment.user else "unknown",
                }
            )
        return comments

    def remove_label(self, issue_or_pr_number: int, label: str) -> None:
        """Remove a label from an issue or PR.

        Args:
            issue_or_pr_number: The issue or PR number.
            label: Label name to remove.
        """
        issue = self._repo.get_issue(issue_or_pr_number)
        try:
            issue.remove_from_labels(label)
        except GithubException as e:
            if e.status != 404:
                raise
