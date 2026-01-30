"""Issue Moderator - Classifies and responds to new GitHub issues."""

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass

from dotenv import load_dotenv

from agents.utils.github_client import GitHubClient
from agents.utils.llm_client import LLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ISSUE_CLASSIFICATION_SYSTEM_PROMPT = """You are an assistant responsible for moderating and organizing GitHub issues.

Analyze the issue title and body, then classify it into one of these categories:
- bug: Describes broken or unintended behavior
- suggestion: Proposes a new feature or improvement
- question: Asks for guidance, clarification, or explanation
- documentation: Points out an error or unclear section in documentation
- unknown: Cannot be categorized confidently or lacks context

For bugs, also determine severity:
- critical: System crash, data loss, security vulnerability
- major: Feature broken, significant impact on users
- minor: Small issue, cosmetic, workaround exists

Generate a friendly response message based on the issue type.

IMPORTANT: Respond ONLY with valid JSON in this exact format:
{
  "type": "<bug | suggestion | question | documentation | unknown>",
  "severity": "<minor | major | critical | none>",
  "comment": "<friendly response message>"
}
"""

RESPONSE_TEMPLATES = {
    "bug": """Hello! Thank you for reporting this issue. To help us investigate more efficiently, please provide:

- The version you're using (commit, tag, or environment information)
- Steps to reproduce the problem
- Expected vs. actual behavior
- A minimal code example, if possible

This will help us determine the priority and fix it faster.""",
    "suggestion": """Hello! Thank you for your suggestion. This idea needs discussion before we can proceed.

Please describe your idea in more detail:
- What problem does it solve?
- How do you imagine the desired behavior?
- Are there similar implementations in other projects?

Your input helps us make this project better. A maintainer will review and discuss this with you.""",
    "question": """Hi there! Thanks for your question. Please describe what's confusing or not working as expected. Include a short code example, what you expected to happen, and what actually occurs.

We'll do our best to clarify things, and if it's a common issue, we may improve the documentation too.""",
    "documentation": """Thanks for pointing out an issue in the documentation. Could you please specify where you found the problem (file, section, example)?

We'll fix it to keep the documentation clear and accurate.""",
    "unknown": """Hello! Thank you for opening an issue. It's not completely clear what category this belongs to (bug, suggestion, or question).

Please provide more context or a detailed description so we can classify it correctly.""",
}


@dataclass
class ClassificationResult:
    """Result of issue classification."""

    issue_type: str
    severity: str
    comment: str
    labels: list[str]


@dataclass
class ModerationResult:
    """Result of issue moderation."""

    success: bool
    issue_number: int
    classification: ClassificationResult | None = None
    error: str | None = None


class IssueModerator:
    """Agent for classifying and responding to GitHub issues."""

    def __init__(
        self,
        github_client: GitHubClient | None = None,
        llm_client: LLMClient | None = None,
        installation_id: int | None = None,
        repository: str | None = None,
    ) -> None:
        """Initialize Issue Moderator.

        Args:
            github_client: GitHub client instance.
            llm_client: LLM client instance.
            installation_id: GitHub App installation ID (for multi-tenant support).
            repository: Repository in owner/repo format.
        """
        if github_client:
            self._github = github_client
        elif installation_id:
            self._github = GitHubClient(
                installation_id=installation_id,
                repository=repository,
            )
        else:
            token = os.environ.get("GITHUB_TOKEN")
            self._github = GitHubClient(token=token, repository=repository)
        self._llm = llm_client or LLMClient()

    def run(self, issue_number: int) -> ModerationResult:
        """Moderate a GitHub issue.

        Args:
            issue_number: GitHub issue number.

        Returns:
            ModerationResult with the moderation outcome.
        """
        logger.info(f"Moderating issue #{issue_number}")

        try:
            issue = self._github.get_issue(issue_number)
            logger.info(f"Issue: {issue.title}")
        except Exception as e:
            logger.error(f"Failed to fetch issue: {e}")
            return ModerationResult(
                success=False,
                issue_number=issue_number,
                error=f"Failed to fetch issue: {e}",
            )

        if self._is_already_moderated(issue_number):
            logger.info(f"Issue #{issue_number} already moderated, skipping")
            return ModerationResult(
                success=True,
                issue_number=issue_number,
            )

        try:
            classification = self._classify_issue(issue.title, issue.body)
        except Exception as e:
            logger.error(f"Failed to classify issue: {e}")
            return ModerationResult(
                success=False,
                issue_number=issue_number,
                error=f"Failed to classify issue: {e}",
            )

        try:
            self._post_response(issue_number, classification)
        except Exception as e:
            logger.error(f"Failed to post response: {e}")
            return ModerationResult(
                success=False,
                issue_number=issue_number,
                classification=classification,
                error=f"Failed to post response: {e}",
            )

        return ModerationResult(
            success=True,
            issue_number=issue_number,
            classification=classification,
        )

    def _is_already_moderated(self, issue_number: int) -> bool:
        """Check if issue was already moderated by looking for bot signature.

        Args:
            issue_number: The issue number.

        Returns:
            True if already moderated, False otherwise.
        """
        try:
            comments = self._github.get_issue_comments(issue_number)
            for comment in comments:
                if "*Generated by Issue Moderator*" in comment.get("body", ""):
                    return True
        except Exception as e:
            logger.warning(f"Failed to check existing comments: {e}")
        return False

    def _classify_issue(self, title: str, body: str) -> ClassificationResult:
        """Classify issue using LLM.

        Args:
            title: Issue title.
            body: Issue body.

        Returns:
            ClassificationResult with type, severity, and comment.
        """
        prompt = f"""Analyze this GitHub issue and classify it:

**Title:** {title}

**Body:**
{body[:2000] if body else "(no description provided)"}

Respond with JSON only."""

        response = self._llm.generate_code(
            prompt=prompt,
            system_prompt=ISSUE_CLASSIFICATION_SYSTEM_PROMPT,
        )

        classification = self._parse_classification(response)
        return classification

    def _parse_classification(self, response: str) -> ClassificationResult:
        """Parse LLM response into ClassificationResult.

        Args:
            response: Raw LLM response.

        Returns:
            ClassificationResult parsed from response.
        """
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                issue_type = data.get("type", "unknown")
                severity = data.get("severity", "none")
                comment = data.get("comment", "")

                if issue_type not in [
                    "bug",
                    "suggestion",
                    "question",
                    "documentation",
                    "unknown",
                ]:
                    issue_type = "unknown"

                if not comment:
                    comment = RESPONSE_TEMPLATES.get(
                        issue_type, RESPONSE_TEMPLATES["unknown"]
                    )

                labels = self._generate_labels(issue_type, severity)

                return ClassificationResult(
                    issue_type=issue_type,
                    severity=severity,
                    comment=comment,
                    labels=labels,
                )
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON: {e}")

        return ClassificationResult(
            issue_type="unknown",
            severity="none",
            comment=RESPONSE_TEMPLATES["unknown"],
            labels=["needs-triage"],
        )

    def _generate_labels(self, issue_type: str, severity: str) -> list[str]:
        """Generate labels based on classification.

        Args:
            issue_type: The issue type.
            severity: The severity level.

        Returns:
            List of label names. Empty for suggestions (needs discussion first).
        """
        if issue_type == "suggestion":
            return []

        labels = []

        type_labels = {
            "bug": "bug",
            "question": "question",
            "documentation": "documentation",
            "unknown": "needs-triage",
        }
        labels.append(type_labels.get(issue_type, "needs-triage"))

        if issue_type == "bug" and severity in ["minor", "major", "critical"]:
            labels.append(f"severity:{severity}")

        return labels

    def _post_response(
        self, issue_number: int, classification: ClassificationResult
    ) -> None:
        """Post comment and add labels to issue.

        Args:
            issue_number: The issue number.
            classification: The classification result.
        """
        comment = f"{classification.comment}\n\n---\n*Generated by Issue Moderator*"
        self._github.post_comment(issue_number, comment)
        logger.info(f"Posted comment on issue #{issue_number}")

        for label in classification.labels:
            try:
                self._github.add_label(issue_number, label)
                logger.info(f"Added label '{label}' to issue #{issue_number}")
            except Exception as e:
                logger.warning(f"Failed to add label '{label}': {e}")


def main() -> None:
    """Main entry point for the Issue Moderator CLI."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Classify and respond to GitHub issues",
    )
    parser.add_argument(
        "--issue",
        type=int,
        required=True,
        help="GitHub issue number to moderate",
    )
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="Output result as JSON",
    )

    args = parser.parse_args()

    moderator = IssueModerator()
    result = moderator.run(args.issue)

    if args.output_json:
        output = {
            "success": result.success,
            "issue_number": result.issue_number,
            "error": result.error,
        }
        if result.classification:
            output["classification"] = {
                "type": result.classification.issue_type,
                "severity": result.classification.severity,
                "labels": result.classification.labels,
            }
        print(json.dumps(output, indent=2))
    else:
        if result.success and result.classification:
            print(
                f"Issue #{result.issue_number} classified as: {result.classification.issue_type}"
            )
            print(f"Severity: {result.classification.severity}")
            print(f"Labels: {', '.join(result.classification.labels)}")
        else:
            print(f"Moderation failed: {result.error}")
            sys.exit(1)


if __name__ == "__main__":
    main()
