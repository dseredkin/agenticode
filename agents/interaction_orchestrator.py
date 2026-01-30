"""Interaction Orchestrator - Manages the review-fix cycle between agents."""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass

from dotenv import load_dotenv

from agents.code_agent import CodeAgent
from agents.reviewer_agent import ReviewerAgent
from agents.utils.github_client import GitHubClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class InteractionResult:
    """Result of the interaction orchestration."""

    success: bool
    pr_number: int | None
    review_rounds: int
    final_status: str  # "approved", "gave_up", "error"
    error: str | None = None


class InteractionOrchestrator:
    """Orchestrates the review-fix cycle between CodeAgent and ReviewerAgent."""

    def __init__(
        self,
        code_agent: CodeAgent | None = None,
        reviewer_agent: ReviewerAgent | None = None,
        github_client: GitHubClient | None = None,
        max_review_rounds: int | None = None,
    ) -> None:
        """Initialize Interaction Orchestrator.

        Args:
            code_agent: CodeAgent instance for code generation.
            reviewer_agent: ReviewerAgent instance for PR reviews.
            github_client: GitHub client for posting comments/labels.
            max_review_rounds: Maximum number of review-fix cycles.
        """
        self._code_agent = code_agent or CodeAgent()
        self._reviewer_agent = reviewer_agent or ReviewerAgent()

        if github_client:
            self._github = github_client
        else:
            token = os.environ.get("GITHUB_TOKEN")
            self._github = GitHubClient(token=token)

        self._max_review_rounds = max_review_rounds or int(
            os.environ.get("MAX_REVIEW_ROUNDS", "3")
        )

    def run_from_issue(self, issue_number: int) -> InteractionResult:
        """Run the full cycle: issue -> PR -> review loop.

        Args:
            issue_number: GitHub issue number to process.

        Returns:
            InteractionResult with the orchestration outcome.
        """
        logger.info(f"Starting orchestration from issue #{issue_number}")

        generation_result = self._code_agent.run(issue_number)

        if not generation_result.success or not generation_result.pr_number:
            logger.error(f"Code generation failed: {generation_result.error}")
            return InteractionResult(
                success=False,
                pr_number=None,
                review_rounds=0,
                final_status="error",
                error=generation_result.error or "Code generation failed",
            )

        pr_number = generation_result.pr_number
        logger.info(f"PR #{pr_number} created, starting review loop")

        return self._run_review_loop(pr_number, starting_round=1)

    def run_from_pr(self, pr_number: int) -> InteractionResult:
        """Continue orchestration from an existing PR.

        Args:
            pr_number: GitHub PR number to continue from.

        Returns:
            InteractionResult with the orchestration outcome.
        """
        logger.info(f"Starting orchestration from PR #{pr_number}")

        labels = self._github.get_pr_labels(pr_number)
        starting_round = self._get_review_round_from_labels(labels) + 1

        logger.info(f"Resuming from round {starting_round}")

        return self._run_review_loop(pr_number, starting_round)

    def _run_review_loop(
        self,
        pr_number: int,
        starting_round: int,
    ) -> InteractionResult:
        """Core loop: review -> fix -> review until approved or threshold.

        Args:
            pr_number: The PR number to review.
            starting_round: The round number to start from.

        Returns:
            InteractionResult with the loop outcome.
        """
        current_round = starting_round

        while current_round <= self._max_review_rounds:
            logger.info(
                f"Review round {current_round}/{self._max_review_rounds} "
                f"for PR #{pr_number}"
            )

            review_result = self._reviewer_agent.run(pr_number, wait_for_ci=True)

            if not review_result.success or not review_result.decision:
                logger.error(f"Review failed: {review_result.error}")
                return InteractionResult(
                    success=False,
                    pr_number=pr_number,
                    review_rounds=current_round,
                    final_status="error",
                    error=review_result.error or "Review failed",
                )

            if review_result.decision.status == "APPROVE":
                logger.info(f"PR #{pr_number} approved after {current_round} rounds")
                self._update_review_round_label(pr_number, current_round)
                return InteractionResult(
                    success=True,
                    pr_number=pr_number,
                    review_rounds=current_round,
                    final_status="approved",
                )

            if review_result.decision.status != "REQUEST_CHANGES":
                logger.error(
                    f"Unexpected review status: {review_result.decision.status}. "
                    "Expected APPROVE or REQUEST_CHANGES."
                )
                return InteractionResult(
                    success=False,
                    pr_number=pr_number,
                    review_rounds=current_round,
                    final_status="error",
                    error=f"Unexpected review status: {review_result.decision.status}",
                )

            logger.info(
                f"Reviewer explicitly requested changes "
                f"(status: {review_result.decision.status}), running code iteration"
            )

            iteration_result = self._code_agent.run_pr_iteration(pr_number)

            if not iteration_result.success:
                logger.error(f"Code iteration failed: {iteration_result.error}")
                return InteractionResult(
                    success=False,
                    pr_number=pr_number,
                    review_rounds=current_round,
                    final_status="error",
                    error=iteration_result.error or "Code iteration failed",
                )

            self._update_review_round_label(pr_number, current_round)
            current_round += 1

        logger.warning(
            f"Review threshold exceeded for PR #{pr_number} "
            f"after {self._max_review_rounds} rounds"
        )
        self._post_giving_up_message(pr_number, self._max_review_rounds)

        return InteractionResult(
            success=False,
            pr_number=pr_number,
            review_rounds=self._max_review_rounds,
            final_status="gave_up",
            error=f"Review threshold exceeded after {self._max_review_rounds} rounds",
        )

    def _post_giving_up_message(self, pr_number: int, rounds: int) -> None:
        """Post comment and label when threshold is exceeded.

        Args:
            pr_number: The PR number.
            rounds: Number of review rounds attempted.
        """
        message = (
            f"Review threshold exceeded after {rounds} rounds. "
            "Manual intervention required."
        )

        try:
            self._github.post_comment(pr_number, message)
            self._github.add_label(pr_number, "review-threshold-exceeded")
            logger.info(f"Posted giving up message on PR #{pr_number}")
        except Exception as e:
            logger.error(f"Failed to post giving up message: {e}")

    def _get_review_round_from_labels(self, labels: list[str]) -> int:
        """Get current review round from labels.

        Args:
            labels: List of label names.

        Returns:
            Current review round number (0 if no label found).
        """
        for label in labels:
            if label.startswith("review-round-"):
                try:
                    return int(label.split("-")[2])
                except (IndexError, ValueError):
                    pass
        return 0

    def _update_review_round_label(self, pr_number: int, round_num: int) -> None:
        """Update the review round label on a PR.

        Args:
            pr_number: The PR number.
            round_num: The new round number.
        """
        try:
            labels = self._github.get_pr_labels(pr_number)

            for label in labels:
                if label.startswith("review-round-"):
                    self._github.remove_label(pr_number, label)

            self._github.add_label(pr_number, f"review-round-{round_num}")
        except Exception as e:
            logger.warning(f"Failed to update review round label: {e}")


def main() -> None:
    """Main entry point for the Interaction Orchestrator CLI."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Orchestrate the review-fix cycle between agents",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--issue",
        type=int,
        help="GitHub issue number to start from",
    )
    group.add_argument(
        "--pr",
        type=int,
        help="GitHub PR number to continue from",
    )
    parser.add_argument(
        "--max-review-rounds",
        type=int,
        default=None,
        help="Maximum number of review-fix cycles",
    )
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="Output result as JSON",
    )

    args = parser.parse_args()

    orchestrator = InteractionOrchestrator(
        max_review_rounds=args.max_review_rounds,
    )

    if args.pr:
        result = orchestrator.run_from_pr(args.pr)
    else:
        result = orchestrator.run_from_issue(args.issue)

    if args.output_json:
        output = {
            "success": result.success,
            "pr_number": result.pr_number,
            "review_rounds": result.review_rounds,
            "final_status": result.final_status,
            "error": result.error,
        }
        print(json.dumps(output, indent=2))
    else:
        if result.success:
            print(f"PR #{result.pr_number} {result.final_status}")
            print(f"Review rounds: {result.review_rounds}")
        else:
            print(f"Orchestration failed: {result.final_status}")
            print(f"Error: {result.error}")
            sys.exit(1)


if __name__ == "__main__":
    main()
