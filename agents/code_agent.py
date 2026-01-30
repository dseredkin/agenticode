"""Code Agent - CLI tool for generating code from GitHub issues."""

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from dotenv import load_dotenv

from agents.utils.code_formatter import CodeFormatter, ValidationSummary
from agents.utils.github_app import get_installation_id_for_repo, load_private_key
from agents.utils.github_client import FileChange, GitHubClient, IssueDetails, PRDetails
from agents.utils.llm_client import LLMClient
from agents.utils.prompts import (
    CODE_GENERATION_SYSTEM_PROMPT,
    format_code_generation_prompt,
    format_code_iteration_prompt,
    format_pr_review_iteration_prompt,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class GeneratedFile:
    """Represents a generated file."""

    path: str
    content: str


@dataclass
class IterationResult:
    """Result of a single code generation iteration."""

    iteration: int
    files: list[GeneratedFile]
    validation: ValidationSummary | None
    success: bool
    errors: list[str] = field(default_factory=list)


@dataclass
class GenerationResult:
    """Final result of code generation."""

    success: bool
    iterations: list[IterationResult]
    final_files: list[GeneratedFile]
    pr_number: int | None = None
    error: str | None = None


class CodeAgent:
    """Agent for generating code from GitHub issues."""

    def __init__(
        self,
        github_client: GitHubClient | None = None,
        llm_client: LLMClient | None = None,
        code_formatter: CodeFormatter | None = None,
        max_iterations: int | None = None,
        iteration_timeout: int | None = None,
        installation_id: int | None = None,
        repository: str | None = None,
    ) -> None:
        """Initialize Code Agent.

        Args:
            github_client: GitHub client instance.
            llm_client: LLM client instance.
            code_formatter: Code formatter instance.
            max_iterations: Maximum number of generation attempts.
            iteration_timeout: Timeout per iteration in seconds.
            installation_id: GitHub App installation ID (for multi-tenant support).
            repository: Repository in owner/repo format.
        """
        # Use contributor app credentials if available
        contributor_app_id = os.environ.get("GITHUB_APP_CONTRIBUTOR_ID")
        contributor_app_key_env = os.environ.get("GITHUB_APP_CONTRIBUTOR_PRIVATE_KEY")

        # Handle private key (can be path or content)
        contributor_app_key = None
        if contributor_app_key_env:
            if contributor_app_key_env.startswith("-----BEGIN"):
                contributor_app_key = contributor_app_key_env
            else:
                contributor_app_key = load_private_key(contributor_app_key_env)

        if github_client:
            self._github = github_client
        elif contributor_app_id and contributor_app_key and repository:
            # Look up the contributor app's installation ID for this repository
            contributor_installation_id = get_installation_id_for_repo(
                contributor_app_id, contributor_app_key, repository
            )
            self._github = GitHubClient(
                installation_id=contributor_installation_id,
                repository=repository,
                app_id=contributor_app_id,
                app_private_key=contributor_app_key,
            )
        else:
            token = os.environ.get("CODE_AGENT_TOKEN") or os.environ.get("GITHUB_TOKEN")
            self._github = GitHubClient(token=token, repository=repository)
        self._llm = llm_client or LLMClient()
        self._formatter = code_formatter or CodeFormatter()
        self._max_iterations = max_iterations or int(
            os.environ.get("MAX_ITERATIONS", "5")
        )
        self._iteration_timeout = iteration_timeout or int(
            os.environ.get("ITERATION_TIMEOUT", "600")
        )
        self._repo_structure_limit = int(os.environ.get("REPO_STRUCTURE_LIMIT", "100"))
        self._files_to_check_limit = int(os.environ.get("FILES_TO_CHECK_LIMIT", "30"))
        self._relevant_files_limit = int(os.environ.get("RELEVANT_FILES_LIMIT", "10"))

    def run(self, issue_number: int) -> GenerationResult:
        """Run code generation for an issue.

        Args:
            issue_number: GitHub issue number.

        Returns:
            GenerationResult with the generation outcome.
        """
        logger.info(f"Starting code generation for issue #{issue_number}")

        try:
            issue = self._github.get_issue(issue_number)
            logger.info(f"Issue: {issue.title}")
        except Exception as e:
            logger.error(f"Failed to fetch issue: {e}")
            return GenerationResult(
                success=False,
                iterations=[],
                final_files=[],
                error=f"Failed to fetch issue: {e}",
            )

        logger.info("Fetching repository structure...")
        repo_structure = self._github.get_repo_structure()
        logger.info(f"Found {len(repo_structure)} files in repository")

        logger.info("Fetching relevant code for context...")
        existing_code = self._get_relevant_code(repo_structure, issue)
        logger.info(f"Found {len(existing_code)} relevant files")

        iterations: list[IterationResult] = []
        current_files: list[GeneratedFile] = []
        previous_errors: list[str] = []

        for iteration in range(1, self._max_iterations + 1):
            logger.info(f"Iteration {iteration}/{self._max_iterations}")
            start_time = time.time()

            try:
                result = self._run_iteration(
                    issue=issue,
                    repo_structure=repo_structure,
                    existing_code=existing_code,
                    previous_files=current_files,
                    previous_errors=previous_errors,
                    iteration=iteration,
                )
                iterations.append(result)

                if result.success:
                    current_files = result.files
                    logger.info(f"Iteration {iteration} succeeded")
                    break

                current_files = result.files
                previous_errors = result.errors
                logger.warning(
                    f"Iteration {iteration} failed with {len(result.errors)} errors:"
                )
                for err in result.errors:
                    logger.warning(f"  - {err}")

            except TimeoutError:
                logger.error(f"Iteration {iteration} timed out")
                iterations.append(
                    IterationResult(
                        iteration=iteration,
                        files=current_files,
                        validation=None,
                        success=False,
                        errors=["Iteration timed out"],
                    )
                )
                break

            elapsed = time.time() - start_time
            logger.info(f"Iteration {iteration} completed in {elapsed:.1f}s")
            if elapsed > self._iteration_timeout:
                logger.warning(f"Iteration took {elapsed:.1f}s, approaching timeout")

        if not iterations or not iterations[-1].success:
            logger.error("Code generation failed after all iterations")
            return GenerationResult(
                success=False,
                iterations=iterations,
                final_files=current_files,
                error="Max iterations reached without success",
            )

        try:
            logger.info("Creating pull request...")
            pr_number = self._create_pr(issue, current_files)
            logger.info(f"Successfully created PR #{pr_number}")
            return GenerationResult(
                success=True,
                iterations=iterations,
                final_files=current_files,
                pr_number=pr_number,
            )
        except Exception as e:
            logger.error(f"Failed to create PR: {e}", exc_info=True)
            return GenerationResult(
                success=False,
                iterations=iterations,
                final_files=current_files,
                error=f"Failed to create PR: {e}",
            )

    def _run_iteration(
        self,
        issue: IssueDetails,
        repo_structure: list[str],
        existing_code: dict[str, str],
        previous_files: list[GeneratedFile],
        previous_errors: list[str],
        iteration: int,
    ) -> IterationResult:
        """Run a single code generation iteration.

        Args:
            issue: The issue details.
            repo_structure: List of files in the repo.
            existing_code: Relevant existing code.
            previous_files: Files from previous iteration.
            previous_errors: Errors from previous iteration.
            iteration: Current iteration number.

        Returns:
            IterationResult with the iteration outcome.
        """
        if previous_errors and previous_files:
            logger.info("Building iteration prompt with previous errors...")
            previous_code = "\n\n".join(
                f"# {f.path}\n{f.content}" for f in previous_files
            )
            prompt = format_code_iteration_prompt(
                issue_title=issue.title,
                issue_body=issue.body,
                previous_code=previous_code,
                validation_errors=previous_errors,
            )
        else:
            logger.info("Building initial code generation prompt...")
            prompt = format_code_generation_prompt(
                issue_title=issue.title,
                issue_body=issue.body,
                repo_structure=repo_structure,
                existing_code=existing_code,
                repo_structure_limit=self._repo_structure_limit,
            )

        logger.info(f"Prompt length: {len(prompt)} chars")
        logger.info("Calling LLM for code generation...")
        llm_start = time.time()
        response = self._llm.generate_code(
            prompt=prompt,
            system_prompt=CODE_GENERATION_SYSTEM_PROMPT,
        )
        llm_elapsed = time.time() - llm_start
        logger.info(f"LLM response received in {llm_elapsed:.1f}s")

        files = self._parse_code_response(response)
        if not files:
            logger.error(
                f"No code files parsed from LLM response (iteration {iteration})"
            )
            logger.error(f"LLM response (first 1000 chars): {response[:1000]}")
            return IterationResult(
                iteration=iteration,
                files=[],
                validation=None,
                success=False,
                errors=["No code files generated - LLM response could not be parsed"],
            )

        logger.info(f"Generated {len(files)} files: {[f.path for f in files]}")

        # Parallel validation
        logger.info("Validating generated files...")
        validated_files, all_errors = self._validate_files_parallel(files)

        success = len(all_errors) == 0
        if success:
            logger.info("All files passed validation")
        return IterationResult(
            iteration=iteration,
            files=validated_files,
            validation=None,
            success=success,
            errors=all_errors,
        )

    def _parse_code_response(self, response: str) -> list[GeneratedFile]:
        """Parse LLM response to extract code files.

        Args:
            response: Raw LLM response.

        Returns:
            List of GeneratedFile objects.
        """
        files: list[GeneratedFile] = []
        # Match any language identifier (or none) in code blocks
        code_block_pattern = r"```(\w*)\n(.*?)```"
        matches = re.findall(code_block_pattern, response, re.DOTALL)

        # Map language identifiers to file extensions
        lang_to_ext = {
            "python": ".py",
            "py": ".py",
            "kotlin": ".kt",
            "java": ".java",
            "javascript": ".js",
            "js": ".js",
            "typescript": ".ts",
            "ts": ".ts",
            "go": ".go",
            "rust": ".rs",
            "ruby": ".rb",
            "cpp": ".cpp",
            "c": ".c",
            "csharp": ".cs",
            "cs": ".cs",
            "swift": ".swift",
            "scala": ".scala",
            "php": ".php",
            "html": ".html",
            "css": ".css",
            "json": ".json",
            "yaml": ".yaml",
            "yml": ".yml",
            "xml": ".xml",
            "sql": ".sql",
            "sh": ".sh",
            "bash": ".sh",
            "": ".txt",
        }

        for lang, block in matches:
            lines = block.strip().split("\n")
            if not lines:
                continue

            first_line = lines[0].strip()
            # Check if first line is a file path comment (# path or // path)
            path_match = re.match(r"^(?:#|//)\s*(.+\.\w+)$", first_line)
            if path_match:
                path = path_match.group(1).strip()
                content = "\n".join(lines[1:])
            elif "/" in first_line and "." in first_line.split("/")[-1]:
                # First line looks like a path without comment prefix
                path = first_line
                content = "\n".join(lines[1:])
            else:
                # Generate a default path based on language
                ext = lang_to_ext.get(lang.lower(), ".txt")
                path = f"src/generated_{len(files)}{ext}"
                content = block.strip()

            if content.strip():
                files.append(GeneratedFile(path=path, content=content))

        return files

    def _get_relevant_code(
        self,
        repo_structure: list[str],
        issue: IssueDetails,
    ) -> dict[str, str]:
        """Get relevant existing code based on issue context.

        Args:
            repo_structure: List of files in the repo.
            issue: The issue details.

        Returns:
            Dict mapping file paths to content.
        """
        relevant_files: list[str] = []
        keywords = self._extract_keywords(issue.title + " " + issue.body)

        python_files = [f for f in repo_structure if f.endswith(".py")]

        # Parallel fetch file contents for keyword matching
        file_contents: dict[str, str] = {}
        files_to_check = python_files[: self._files_to_check_limit]

        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_path = {
                executor.submit(self._github.get_file_content, path): path
                for path in files_to_check
            }
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    content = future.result()
                    if content:
                        file_contents[path] = content
                except Exception as e:
                    logger.warning(f"Failed to fetch {path}: {e}")

        # Check for keyword matches
        for file_path, content in file_contents.items():
            for keyword in keywords:
                if keyword.lower() in content.lower():
                    relevant_files.append(file_path)
                    break

        # Return relevant files (already fetched, no need to re-fetch)
        result: dict[str, str] = {}
        for file_path in relevant_files[: self._relevant_files_limit]:
            if file_path in file_contents:
                result[file_path] = file_contents[file_path]

        return result

    def _validate_files_parallel(
        self, files: list[GeneratedFile]
    ) -> tuple[list[GeneratedFile], list[str]]:
        """Validate and format files in parallel.

        Args:
            files: List of generated files.

        Returns:
            Tuple of (validated_files, all_errors).
        """
        validated_files: list[GeneratedFile] = []
        all_errors: list[str] = []

        # Separate Python files for validation
        py_files = [(i, f) for i, f in enumerate(files) if f.path.endswith(".py")]
        non_py_files = [
            (i, f) for i, f in enumerate(files) if not f.path.endswith(".py")
        ]

        # Parallel validation for Python files
        validation_results: dict[int, tuple[GeneratedFile, list[str]]] = {}

        def validate_file(
            idx: int, file: GeneratedFile
        ) -> tuple[int, GeneratedFile, list[str]]:
            validation = self._formatter.validate_all(file.content, file.path)
            if validation.black_result.formatted_code:
                file = GeneratedFile(
                    path=file.path,
                    content=validation.black_result.formatted_code,
                )
            if validation.all_errors:
                err_count = len(validation.all_errors)
                logger.info(f"Validation failed for {file.path}: {err_count} errors")
                for err in validation.all_errors:
                    logger.info(f"  [{file.path}] {err}")
            else:
                logger.info(f"Validation passed for {file.path}")
            return idx, file, validation.all_errors

        logger.info(f"Validating {len(py_files)} Python files...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(validate_file, i, f) for i, f in py_files]
            for future in as_completed(futures):
                idx, file, errors = future.result()
                validation_results[idx] = (file, errors)

        # Combine results preserving original order
        all_indexed: list[tuple[int, GeneratedFile]] = []
        for i, f in non_py_files:
            all_indexed.append((i, f))
        for i, (f, errors) in validation_results.items():
            all_indexed.append((i, f))
            all_errors.extend(errors)

        all_indexed.sort(key=lambda x: x[0])
        validated_files = [f for _, f in all_indexed]

        return validated_files, all_errors

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from text for code search.

        Args:
            text: Text to extract keywords from.

        Returns:
            List of keywords.
        """
        words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", text)
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "used",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "and",
            "but",
            "or",
            "nor",
            "so",
            "yet",
            "both",
            "either",
            "neither",
            "not",
            "only",
            "own",
            "same",
            "than",
            "too",
            "very",
            "just",
            "also",
            "now",
            "add",
            "create",
            "implement",
            "fix",
            "update",
            "change",
            "modify",
            "new",
            "feature",
            "that",
            "this",
            "these",
            "those",
            "which",
            "what",
            "who",
        }
        keywords = [w for w in words if w.lower() not in stopwords and len(w) > 2]
        return list(dict.fromkeys(keywords))[:20]

    def _create_pr(
        self,
        issue: IssueDetails,
        files: list[GeneratedFile],
    ) -> int:
        """Create a pull request with the generated code.

        Args:
            issue: The issue details.
            files: Generated files to commit.

        Returns:
            The PR number.
        """
        branch_name = f"auto/issue-{issue.number}"
        self._github.create_branch(branch_name)

        file_changes = [FileChange(path=f.path, content=f.content) for f in files]
        self._github.commit_files(
            files=file_changes,
            message=f"feat: implement issue #{issue.number}\n\n{issue.title}",
            branch=branch_name,
        )

        pr_body = f"""## Summary
Automated implementation for issue #{issue.number}.

## Changes
{chr(10).join(f"- `{f.path}`" for f in files)}

## Original Issue
{issue.body[:500]}{"..." if len(issue.body) > 500 else ""}

---
Generated by Code Agent
"""

        pr_number = self._github.create_pr(
            title=f"feat: {issue.title}",
            body=pr_body,
            head_branch=branch_name,
        )

        self._github.link_pr_to_issue(pr_number, issue.number)
        return pr_number

    def run_pr_iteration(self, pr_number: int) -> GenerationResult:
        """Run code generation iteration based on PR review feedback.

        Args:
            pr_number: GitHub PR number to iterate on.

        Returns:
            GenerationResult with the iteration outcome.
        """
        logger.info(f"Starting PR iteration for PR #{pr_number}")

        try:
            pr = self._github.get_pr(pr_number)
            logger.info(f"PR: {pr.title}")
        except Exception as e:
            logger.error(f"Failed to fetch PR: {e}")
            return GenerationResult(
                success=False,
                iterations=[],
                final_files=[],
                error=f"Failed to fetch PR: {e}",
            )

        issue = self._get_linked_issue_from_pr(pr)
        if not issue:
            logger.error(f"No linked issue found in PR #{pr_number}")
            return GenerationResult(
                success=False,
                iterations=[],
                final_files=[],
                error="No linked issue found in PR",
            )

        labels = self._github.get_pr_labels(pr_number)
        iteration = self._get_iteration_from_labels(labels) + 1

        if iteration > self._max_iterations:
            logger.error(f"Max iterations ({self._max_iterations}) reached")
            return GenerationResult(
                success=False,
                iterations=[],
                final_files=[],
                error=f"Max iterations ({self._max_iterations}) reached",
            )

        review_feedback = self._get_review_feedback(pr_number)
        if not review_feedback:
            logger.error(f"No review feedback found for PR #{pr_number}")
            return GenerationResult(
                success=False,
                iterations=[],
                final_files=[],
                error="No review feedback found",
            )

        current_code = self._get_pr_files(pr)

        prompt = format_pr_review_iteration_prompt(
            issue_title=issue.title,
            issue_body=issue.body,
            current_code=current_code,
            review_feedback=review_feedback,
            iteration=iteration,
            max_iterations=self._max_iterations,
        )

        response = self._llm.generate_code(
            prompt=prompt,
            system_prompt=CODE_GENERATION_SYSTEM_PROMPT,
        )

        files = self._parse_code_response(response)
        if not files:
            logger.error(f"No code files parsed from LLM response for PR #{pr_number}")
            logger.error(f"LLM response (first 1000 chars): {response[:1000]}")
            return GenerationResult(
                success=False,
                iterations=[],
                final_files=[],
                error="No code files generated - LLM response could not be parsed",
            )

        # Parallel validation
        validated_files, all_errors = self._validate_files_parallel(files)

        if all_errors:
            logger.warning(f"Validation errors found: {len(all_errors)}")
            for err in all_errors:
                logger.debug(f"Validation error: {err}")

        file_changes = [
            FileChange(path=f.path, content=f.content) for f in validated_files
        ]
        self._github.commit_files(
            files=file_changes,
            message=f"fix: address review feedback (iteration {iteration})",
            branch=pr.head_branch,
        )

        self._update_iteration_label(pr_number, iteration)

        logger.info(f"PR #{pr_number} updated with iteration {iteration}")
        return GenerationResult(
            success=True,
            iterations=[
                IterationResult(
                    iteration=iteration,
                    files=validated_files,
                    validation=None,
                    success=len(all_errors) == 0,
                    errors=all_errors,
                )
            ],
            final_files=validated_files,
            pr_number=pr_number,
        )

    def _get_linked_issue_from_pr(self, pr: PRDetails) -> IssueDetails | None:
        """Extract linked issue from PR body.

        Args:
            pr: The PRDetails object.

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

    def _get_iteration_from_labels(self, labels: list[str]) -> int:
        """Get current iteration count from labels.

        Args:
            labels: List of label names.

        Returns:
            Current iteration number (0 if no iteration label found).
        """
        for label in labels:
            if label.startswith("iteration-"):
                try:
                    return int(label.split("-")[1])
                except (IndexError, ValueError):
                    pass
        return 0

    def _update_iteration_label(self, pr_number: int, iteration: int) -> None:
        """Update the iteration label on a PR.

        Args:
            pr_number: The PR number.
            iteration: The new iteration number.
        """
        labels = self._github.get_pr_labels(pr_number)
        for label in labels:
            if label.startswith("iteration-"):
                self._github.remove_label(pr_number, label)

        if "needs-revision" in labels:
            self._github.remove_label(pr_number, "needs-revision")

        self._github.add_label(pr_number, f"iteration-{iteration}")

    def _get_review_feedback(self, pr_number: int) -> str:
        """Get review feedback from PR reviews.

        Args:
            pr_number: The PR number.

        Returns:
            Formatted review feedback string.
        """
        reviews = self._github.get_pr_reviews(pr_number)
        comments = self._github.get_review_comments(pr_number)

        feedback_parts: list[str] = []

        request_changes_reviews = [
            r for r in reviews if r["state"] == "CHANGES_REQUESTED"
        ]

        for review in request_changes_reviews:
            if review["body"]:
                feedback_parts.append(f"### Review by {review['user']}")
                feedback_parts.append(review["body"])
                feedback_parts.append("")

        if comments:
            feedback_parts.append("### Line Comments")
            for comment in comments:
                if comment["line"]:
                    location = f"{comment['path']}:{comment['line']}"
                else:
                    location = comment["path"]
                feedback_parts.append(f"- **{location}**: {comment['body']}")

        return "\n".join(feedback_parts) if feedback_parts else ""

    def _get_pr_files(self, pr: PRDetails) -> str:
        """Get current file contents from PR.

        Args:
            pr: The PRDetails object.

        Returns:
            Formatted string of file contents.
        """
        file_contents: dict[str, str] = {}

        # Parallel fetch all changed files
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_path = {
                executor.submit(
                    self._github.get_file_content, path, pr.head_branch
                ): path
                for path in pr.changed_files
            }
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    content = future.result()
                    if content:
                        file_contents[path] = content
                except Exception as e:
                    logger.warning(f"Failed to fetch {path}: {e}")

        # Build result in original order
        result: list[str] = []
        for file_path in pr.changed_files:
            if file_path in file_contents:
                result.append(
                    f"### {file_path}\n```python\n{file_contents[file_path]}\n```"
                )

        return "\n\n".join(result)


def main() -> None:
    """Main entry point for the Code Agent CLI."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Generate code from GitHub issues or iterate on PRs",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--issue",
        type=int,
        help="GitHub issue number to process",
    )
    group.add_argument(
        "--pr",
        type=int,
        help="GitHub PR number to iterate on based on review feedback",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum number of generation attempts",
    )
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="Output result as JSON",
    )

    args = parser.parse_args()

    agent = CodeAgent(max_iterations=args.max_iterations)

    if args.pr:
        result = agent.run_pr_iteration(args.pr)
    else:
        result = agent.run(args.issue)

    if args.output_json:
        output = {
            "success": result.success,
            "pr_number": result.pr_number,
            "iterations": len(result.iterations),
            "files": [f.path for f in result.final_files],
            "error": result.error,
        }
        print(json.dumps(output, indent=2))
    else:
        if result.success:
            if args.pr:
                print(f"Successfully updated PR #{result.pr_number}")
            else:
                print(f"Successfully created PR #{result.pr_number}")
            print(f"Files generated: {len(result.final_files)}")
            for f in result.final_files:
                print(f"  - {f.path}")
        else:
            print(f"Failed to generate code: {result.error}")
            sys.exit(1)


if __name__ == "__main__":
    main()
