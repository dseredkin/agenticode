"""Code Agent - CLI tool for generating code from GitHub issues."""

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from agents.utils.code_formatter import CodeFormatter, ValidationSummary
from agents.utils.github_client import FileChange, GitHubClient, IssueDetails
from agents.utils.llm_client import LLMClient
from agents.utils.prompts import (
    CODE_GENERATION_SYSTEM_PROMPT,
    format_code_generation_prompt,
    format_code_iteration_prompt,
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
    ) -> None:
        """Initialize Code Agent.

        Args:
            github_client: GitHub client instance.
            llm_client: LLM client instance.
            code_formatter: Code formatter instance.
            max_iterations: Maximum number of generation attempts.
            iteration_timeout: Timeout per iteration in seconds.
        """
        self._github = github_client or GitHubClient()
        self._llm = llm_client or LLMClient()
        self._formatter = code_formatter or CodeFormatter()
        self._max_iterations = max_iterations or int(
            os.environ.get("MAX_ITERATIONS", "5")
        )
        self._iteration_timeout = iteration_timeout or int(
            os.environ.get("ITERATION_TIMEOUT", "600")
        )

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

        repo_structure = self._github.get_repo_structure()
        existing_code = self._get_relevant_code(repo_structure, issue)

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
                    f"Iteration {iteration} failed with {len(result.errors)} errors"
                )

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
            if elapsed > self._iteration_timeout:
                logger.warning(f"Iteration took {elapsed:.1f}s, approaching timeout")

        if not iterations or not iterations[-1].success:
            return GenerationResult(
                success=False,
                iterations=iterations,
                final_files=current_files,
                error="Max iterations reached without success",
            )

        try:
            pr_number = self._create_pr(issue, current_files)
            return GenerationResult(
                success=True,
                iterations=iterations,
                final_files=current_files,
                pr_number=pr_number,
            )
        except Exception as e:
            logger.error(f"Failed to create PR: {e}")
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
            prompt = format_code_generation_prompt(
                issue_title=issue.title,
                issue_body=issue.body,
                repo_structure=repo_structure,
                existing_code=existing_code,
            )

        response = self._llm.generate_code(
            prompt=prompt,
            system_prompt=CODE_GENERATION_SYSTEM_PROMPT,
        )

        files = self._parse_code_response(response)
        if not files:
            return IterationResult(
                iteration=iteration,
                files=[],
                validation=None,
                success=False,
                errors=["No code files generated"],
            )

        all_errors: list[str] = []
        validated_files: list[GeneratedFile] = []

        for file in files:
            if file.path.endswith(".py"):
                validation = self._formatter.validate_all(file.content, file.path)
                if validation.black_result.formatted_code:
                    file = GeneratedFile(
                        path=file.path,
                        content=validation.black_result.formatted_code,
                    )
                all_errors.extend(validation.all_errors)
            validated_files.append(file)

        success = len(all_errors) == 0
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
        code_block_pattern = r"```(?:python|py)?\n(.*?)```"
        blocks = re.findall(code_block_pattern, response, re.DOTALL)

        for block in blocks:
            lines = block.strip().split("\n")
            if not lines:
                continue

            first_line = lines[0].strip()
            if first_line.startswith("#") and (
                "/" in first_line or first_line.endswith(".py")
            ):
                path = first_line.lstrip("#").strip()
                content = "\n".join(lines[1:])
            else:
                path = f"src/generated_{len(files)}.py"
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
        relevant_files: dict[str, str] = []
        keywords = self._extract_keywords(issue.title + " " + issue.body)

        python_files = [f for f in repo_structure if f.endswith(".py")]

        for file_path in python_files[:10]:
            content = self._github.get_file_content(file_path)
            if content:
                for keyword in keywords:
                    if keyword.lower() in content.lower():
                        relevant_files.append(file_path)
                        break

        result: dict[str, str] = {}
        for file_path in relevant_files[:5]:
            content = self._github.get_file_content(file_path)
            if content:
                result[file_path] = content

        return result

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from text for code search.

        Args:
            text: Text to extract keywords from.

        Returns:
            List of keywords.
        """
        words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", text)
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once",
            "and", "but", "or", "nor", "so", "yet", "both", "either",
            "neither", "not", "only", "own", "same", "than", "too",
            "very", "just", "also", "now", "add", "create", "implement",
            "fix", "update", "change", "modify", "new", "feature",
            "that", "this", "these", "those", "which", "what", "who",
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

        file_changes = [
            FileChange(path=f.path, content=f.content) for f in files
        ]
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


def main() -> None:
    """Main entry point for the Code Agent CLI."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Generate code from GitHub issues",
    )
    parser.add_argument(
        "--issue",
        type=int,
        required=True,
        help="GitHub issue number to process",
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
            print(f"Successfully created PR #{result.pr_number}")
            print(f"Files generated: {len(result.final_files)}")
            for f in result.final_files:
                print(f"  - {f.path}")
        else:
            print(f"Failed to generate code: {result.error}")
            sys.exit(1)


if __name__ == "__main__":
    main()
