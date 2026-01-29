"""Reviewer Agent - PR review and approval automation."""

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic import BaseModel

from agents.utils.github_client import CIStatus, GitHubClient, IssueDetails, PRDetails
from agents.utils.llm_client import LLMClient
from agents.utils.prompts import CODE_REVIEW_SYSTEM_PROMPT, format_code_review_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ReviewDecision(BaseModel):
    """Review decision model."""

    status: str  # APPROVE or REQUEST_CHANGES
    requirements_met: bool
    ci_passing: bool
    issues: list[str]
    suggestions: list[str]
    summary: str


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
    ) -> None:
        """Initialize Reviewer Agent.

        Args:
            github_client: GitHub client instance.
            llm_client: LLM client instance.
        """
        self._github = github_client or GitHubClient()
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
                decision = ReviewDecision(
                    status=data.get("status", "REQUEST_CHANGES"),
                    requirements_met=data.get("requirements_met", False),
                    ci_passing=ci_status.state == "success",
                    issues=data.get("issues", []),
                    suggestions=data.get("suggestions", []),
                    summary=data.get("summary", ""),
                )

                if not decision.ci_passing and decision.status == "APPROVE":
                    decision.status = "REQUEST_CHANGES"
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

        body_parts.append("\n---")
        body_parts.append(f"**Requirements Met:** {'Yes' if decision.requirements_met else 'No'}")
        body_parts.append(f"**CI Passing:** {'Yes' if decision.ci_passing else 'No'}")
        body_parts.append("\n*Generated by Reviewer Agent*")

        body = "\n".join(body_parts)
        self._github.post_review(pr_number, body, event)

        logger.info(f"Posted {event} review on PR #{pr_number}")


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
        output = {
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
