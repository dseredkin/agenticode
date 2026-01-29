"""GitHub API client wrapper."""

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from github import Auth, Github, GithubException
from github.PullRequest import PullRequest
from github.Repository import Repository
from pydantic import BaseModel

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
    ) -> None:
        """Initialize GitHub client.

        Args:
            token: GitHub personal access token. Defaults to GITHUB_TOKEN env var.
            repository: Repository in owner/repo format. Defaults to GITHUB_REPOSITORY env var.
        """
        self._token = token or os.environ.get("GITHUB_TOKEN", "")
        self._repository = repository or os.environ.get("GITHUB_REPOSITORY", "")

        if not self._token:
            raise ValueError("GitHub token is required")
        if not self._repository:
            raise ValueError("Repository is required (format: owner/repo)")

        auth = Auth.Token(self._token)
        self._github = Github(auth=auth)
        self._repo: Repository = self._github.get_repo(self._repository)

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
        """List files in the repository.

        Args:
            path: Optional path to list files from.

        Returns:
            List of file paths in the repository.
        """
        files: list[str] = []
        try:
            contents = self._repo.get_contents(path)
            if not isinstance(contents, list):
                contents = [contents]

            for content in contents:
                if content.type == "dir":
                    files.extend(self.get_repo_structure(content.path))
                else:
                    files.append(content.path)
        except GithubException as e:
            logger.warning(f"Failed to get repo structure at {path}: {e}")
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
            content = self._repo.get_contents(path, ref=ref)
            if isinstance(content, list):
                return None
            return content.decoded_content.decode("utf-8")
        except GithubException:
            return None

    def create_branch(self, branch_name: str, base_branch: str = "main") -> str:
        """Create a new branch.

        Args:
            branch_name: Name for the new branch.
            base_branch: Base branch to create from.

        Returns:
            The full ref name of the created branch.
        """
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
        """Commit file changes to a branch.

        Args:
            files: List of FileChange objects with path and content.
            message: Commit message.
            branch: Branch to commit to.

        Returns:
            The commit SHA.
        """
        for file_change in files:
            try:
                existing = self._repo.get_contents(file_change.path, ref=branch)
                if isinstance(existing, list):
                    raise ValueError(f"Path is a directory: {file_change.path}")
                self._repo.update_file(
                    file_change.path,
                    message,
                    file_change.content,
                    existing.sha,
                    branch=branch,
                )
                logger.info(f"Updated file: {file_change.path}")
            except GithubException as e:
                if e.status == 404:
                    self._repo.create_file(
                        file_change.path,
                        message,
                        file_change.content,
                        branch=branch,
                    )
                    logger.info(f"Created file: {file_change.path}")
                else:
                    raise

        ref = self._repo.get_git_ref(f"heads/{branch}")
        return ref.object.sha

    def create_pr(
        self,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str = "main",
    ) -> int:
        """Create a pull request.

        Args:
            title: PR title.
            body: PR description.
            head_branch: Source branch.
            base_branch: Target branch.

        Returns:
            The PR number.
        """
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

            all_complete = all(
                c.conclusion is not None for c in check_runs
            ) if check_runs else combined_status.state != "pending"

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
                    review_comments.append({
                        "path": comment.path,
                        "line": comment.line,
                        "body": comment.body,
                    })

        if review_comments:
            pr.create_review(body=body, event=event, comments=review_comments)
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
