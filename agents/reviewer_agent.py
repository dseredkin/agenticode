"""Reviewer Agent - PR review and approval automation."""

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel

from agents.utils.github_app import get_installation_id_for_repo
from agents.utils.github_client import (
    CIStatus,
    GitHubClient,
    IssueDetails,
    PRDetails,
    ReviewComment,
)
from agents.utils.llm_client import LLMClient
from agents.utils.prompts import CODE_REVIEW_SYSTEM_PROMPT, format_code_review_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class LineComment(BaseModel):
    """Line-specific review comment."""

    path: str
    line: int
    body: str


class ReviewDecision(BaseModel):
    """Review decision model."""

    status: str  # APPROVE or REQUEST_CHANGES
    requirements_met: bool
    ci_passing: bool
    issues: list[str]
    suggestions: list[str]
    summary: str
    line_comments: list[LineComment] = []


@dataclass
class ReviewResult:
    """Result of a PR review."""

    success: bool
    decision: ReviewDecision | None
    error: str | None = None


class ReviewerAgent:
    """Agent for reviewing pull requests."""

    def __init__(
        self,
        github_client: GitHubClient | None = None,
        llm_client: LLMClient | None = None,
        installation_id: int | None = None,
        repository: str | None = None,
    ) -> None:
        """Initialize Reviewer Agent.

        Args:
            github_client: GitHub client instance.
            llm_client: LLM client instance.
            installation_id: GitHub App installation ID (for multi-tenant support).
            repository: Repository in owner/repo format.
        """
        # Use separate reviewer app credentials if available
        reviewer_app_id = os.environ.get("GITHUB_APP_REVIEWER_ID")
        reviewer_app_key_env = os.environ.get("GITHUB_APP_REVIEWER_PRIVATE_KEY")

        # Handle private key (can be path or content)
        reviewer_app_key = None
        if reviewer_app_key_env:
            if reviewer_app_key_env.startswith("-----BEGIN"):
                reviewer_app_key = reviewer_app_key_env
            else:
                from agents.utils.github_app import load_private_key
                reviewer_app_key = load_private_key(reviewer_app_key_env)

        if github_client:
            self._github = github_client
        elif reviewer_app_id and reviewer_app_key and repository:
            # Look up the reviewer app's installation ID for this repository
            reviewer_installation_id = get_installation_id_for_repo(
                reviewer_app_id, reviewer_app_key, repository
            )
            self._github = GitHubClient(
                installation_id=reviewer_installation_id,
                repository=repository,
                app_id=reviewer_app_id,
                app_private_key=reviewer_app_key,
            )
        else:
            from agents.utils.github_app import get_app_token_from_env

            # Try GitHub App token first, then PAT
            token = get_app_token_from_env()
            if not token:
                token = os.environ.get("REVIEWER_AGENT_TOKEN")
            if not token:
                token = os.environ.get("GITHUB_TOKEN")
            self._github = GitHubClient(token=token, repository=repository)
        self._llm = llm_client or LLMClient()

    def run(self, pr_number: int, wait_for_ci: bool = True) -> ReviewResult:
        """Run review for a pull request.

        Args:
            pr_number: GitHub PR number.
            wait_for_ci: Whether to wait for CI to complete.

        Returns:
            ReviewResult with the review outcome.
        """
        logger.info(f"Starting review for PR #{pr_number}")

        try:
            pr = self._github.get_pr(pr_number)
            logger.info(f"PR: {pr.title}")
        except Exception as e:
            logger.error(f"Failed to fetch PR: {e}")
            return ReviewResult(
                success=False,
                decision=None,
                error=f"Failed to fetch PR: {e}",
            )

        issue = self._get_linked_issue(pr)

        if wait_for_ci:
            logger.info("Waiting for CI to complete...")
            ci_status = self._github.get_ci_status(pr_number)
        else:
            ci_status = CIStatus(state="unknown", checks={}, failed_checks=[])

        try:
            decision = self._analyze_pr(pr, issue, ci_status)
        except Exception as e:
            logger.error(f"Failed to analyze PR: {e}")
            return ReviewResult(
                success=False,
                decision=None,
                error=f"Failed to analyze PR: {e}",
            )

        try:
            self._post_review(pr_number, decision)
        except Exception as e:
            logger.error(f"Failed to post review: {e}")
            return ReviewResult(
                success=False,
                decision=decision,
                error=f"Failed to post review: {e}",
            )

        return ReviewResult(success=True, decision=decision)

    def _get_linked_issue(self, pr: PRDetails) -> IssueDetails | None:
        """Get the linked issue from PR body.

        Args:
            pr: The PR details.

        Returns:
            IssueDetails if found, None otherwise.
        """
        patterns = [
            r"[Cc]loses?\s+#(\d+)",
            r"[Ff]ixes?\s+#(\d+)",
            r"[Rr]esolves?\s+#(\d+)",
            r"[Ii]ssue\s+#(\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, pr.body)
            if match:
                issue_number = int(match.group(1))
                try:
                    return self._github.get_issue(issue_number)
                except Exception as e:
                    logger.warning(f"Failed to fetch linked issue #{issue_number}: {e}")
                    break

        return None

    def _analyze_pr(
        self,
        pr: PRDetails,
        issue: IssueDetails | None,
        ci_status: CIStatus,
    ) -> ReviewDecision:
        """Analyze PR using LLM.

        Args:
            pr: The PR details.
            issue: The linked issue details.
            ci_status: CI status.

        Returns:
            ReviewDecision with the analysis.
        """
        prompt = format_code_review_prompt(
            issue_title=issue.title if issue else "N/A",
            issue_body=issue.body if issue else "No linked issue",
            pr_title=pr.title,
            pr_body=pr.body,
            pr_diff=pr.diff,
            ci_status=ci_status.state,
            failed_checks=ci_status.failed_checks,
            changed_files=pr.changed_files,
        )

        response = self._llm.generate_code(
            prompt=prompt,
            system_prompt=CODE_REVIEW_SYSTEM_PROMPT,
        )

        decision = self._parse_review_response(response, ci_status)
        return decision

    def _parse_review_response(
        self,
        response: str,
        ci_status: CIStatus,
    ) -> ReviewDecision:
        """Parse LLM response to ReviewDecision.

        Args:
            response: Raw LLM response.
            ci_status: CI status for validation.

        Returns:
            ReviewDecision parsed from response.
        """
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            try:
                data = json.loads(json_match.group())

                # Parse line comments
                line_comments: list[LineComment] = []
                for lc in data.get("line_comments", []):
                    if all(k in lc for k in ["path", "line", "body"]):
                        line_comments.append(
                            LineComment(
                                path=lc["path"],
                                line=lc["line"],
                                body=lc["body"],
                            )
                        )

                ci_passing = ci_status.state == "success"
                status = data.get("status", "REQUEST_CHANGES")

                # Enforce CI rules
                if not ci_passing and status == "APPROVE":
                    status = "REQUEST_CHANGES"

                # Only approve if CI passes AND no line comments
                if status == "APPROVE" and line_comments:
                    status = "REQUEST_CHANGES"

                decision = ReviewDecision(
                    status=status,
                    requirements_met=data.get("requirements_met", False),
                    ci_passing=ci_passing,
                    issues=data.get("issues", []),
                    suggestions=data.get("suggestions", []),
                    summary=data.get("summary", ""),
                    line_comments=line_comments,
                )

                if not decision.ci_passing:
                    if "CI checks are failing" not in decision.issues:
                        decision.issues.append("CI checks are failing")

                return decision
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {e}")

        return ReviewDecision(
            status="REQUEST_CHANGES",
            requirements_met=False,
            ci_passing=ci_status.state == "success",
            issues=["Could not parse review response"],
            suggestions=[],
            summary=response[:500],
            line_comments=[],
        )

    def _post_review(self, pr_number: int, decision: ReviewDecision) -> None:
        """Post review to GitHub.

        Args:
            pr_number: The PR number.
            decision: The review decision.
        """
        event = "APPROVE" if decision.status == "APPROVE" else "REQUEST_CHANGES"

        body_parts = [f"## Review Summary\n{decision.summary}"]

        if decision.issues:
            body_parts.append("\n### Issues Found")
            for issue in decision.issues:
                body_parts.append(f"- {issue}")

        if decision.suggestions:
            body_parts.append("\n### Suggestions")
            for suggestion in decision.suggestions:
                body_parts.append(f"- {suggestion}")

        if decision.line_comments:
            body_parts.append(f"\n### Line Comments: {len(decision.line_comments)}")
            body_parts.append("Please resolve the inline comments below.")

        body_parts.append("\n---")
        req_met = "Yes" if decision.requirements_met else "No"
        ci_pass = "Yes" if decision.ci_passing else "No"
        body_parts.append(f"**Requirements Met:** {req_met}")
        body_parts.append(f"**CI Passing:** {ci_pass}")
        body_parts.append("\n*Generated by Reviewer Agent*")

        body = "\n".join(body_parts)

        # Convert line comments to ReviewComment format
        review_comments: list[ReviewComment] = []
        for lc in decision.line_comments:
            review_comments.append(
                ReviewComment(
                    body=lc.body,
                    path=lc.path,
                    line=lc.line,
                )
            )

        try:
            self._github.post_review(pr_number, body, event, review_comments or None)
        except Exception as e:
            # Can't request changes on own PR - fall back to COMMENT
            if "own pull request" in str(e).lower() and event == "REQUEST_CHANGES":
                logger.warning(
                    f"Cannot request changes on own PR, posting as COMMENT instead"
                )
                self._github.post_review(
                    pr_number, body, "COMMENT", review_comments or None
                )
                event = "COMMENT"
            else:
                raise

        logger.info(f"Posted {event} review on PR #{pr_number}")
        if review_comments:
            logger.info(f"  with {len(review_comments)} line comments")


def main() -> None:
    """Main entry point for the Reviewer Agent CLI."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Review GitHub pull requests",
    )
    parser.add_argument(
        "--pr",
        type=int,
        required=True,
        help="GitHub PR number to review",
    )
    parser.add_argument(
        "--no-wait-ci",
        action="store_true",
        help="Don't wait for CI to complete",
    )
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="Output result as JSON",
    )

    args = parser.parse_args()

    agent = ReviewerAgent()
    result = agent.run(args.pr, wait_for_ci=not args.no_wait_ci)

    if args.output_json:
        output: dict[str, Any] = {
            "success": result.success,
            "error": result.error,
        }
        if result.decision:
            output["decision"] = {
                "status": result.decision.status,
                "requirements_met": result.decision.requirements_met,
                "ci_passing": result.decision.ci_passing,
                "issues": result.decision.issues,
                "suggestions": result.decision.suggestions,
                "summary": result.decision.summary,
            }
        print(json.dumps(output, indent=2))
    else:
        if result.success and result.decision:
            print(f"Review posted: {result.decision.status}")
            print(f"Summary: {result.decision.summary}")
            if result.decision.issues:
                print("Issues:")
                for issue in result.decision.issues:
                    print(f"  - {issue}")
        else:
            print(f"Review failed: {result.error}")
            sys.exit(1)


if __name__ == "__main__":
    main()
