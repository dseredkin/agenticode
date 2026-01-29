"""LLM prompt templates for code generation and review."""

CODE_GENERATION_SYSTEM_PROMPT = """You are an expert Python developer. Your task is to generate high-quality, production-ready Python code based on the given requirements.

Follow these guidelines:
1. Write clean, readable, and well-structured code
2. Use type hints for all function parameters and return values
3. Follow PEP 8 style guidelines
4. Include docstrings for modules, classes, and functions
5. Handle errors appropriately
6. Never include secrets, API keys, or sensitive data in code
7. Use meaningful variable and function names
8. Keep functions focused and small

Output format:
- Return ONLY the code, no explanations
- Use markdown code blocks with the filename as a comment on the first line
- If multiple files are needed, separate them with a blank line

Example output format:
```python
# src/example.py
def hello() -> str:
    \"\"\"Return a greeting.\"\"\"
    return "Hello, World!"
```
"""

CODE_GENERATION_PROMPT = """## Issue Requirements
Title: {issue_title}
Description:
{issue_body}

## Repository Structure
{repo_structure}

## Existing Code Context
{existing_code}

## Task
Generate Python code that fulfills the requirements described in the issue.

Requirements:
1. Follow the existing code patterns and style
2. Place new code in appropriate locations
3. Include any necessary imports
4. Add tests if the issue mentions testing
5. Ensure type safety with proper type hints

Generate the complete implementation:
"""

CODE_ITERATION_PROMPT = """## Previous Attempt
The previous code generation attempt had the following issues:

### Validation Errors
{validation_errors}

### Previous Code
{previous_code}

## Original Requirements
Title: {issue_title}
Description:
{issue_body}

## Task
Fix the issues mentioned above and generate corrected code.
Focus specifically on resolving the validation errors while maintaining the original functionality.

Generate the corrected implementation:
"""

CODE_REVIEW_SYSTEM_PROMPT = """You are an expert code reviewer. Your task is to review pull requests and provide constructive feedback.

Review criteria:
1. Code correctness - Does the code work as intended?
2. Code quality - Is the code clean, readable, and maintainable?
3. Security - Are there any security vulnerabilities?
4. Performance - Are there any obvious performance issues?
5. Testing - Is the code adequately tested?
6. Requirements - Does the code fulfill the original requirements?

Output format:
Return a JSON object with the following structure:
{
    "status": "APPROVE" or "REQUEST_CHANGES",
    "requirements_met": true/false,
    "ci_passing": true/false,
    "issues": ["list of problems found"],
    "suggestions": ["list of improvement suggestions"],
    "summary": "brief review summary"
}
"""

CODE_REVIEW_PROMPT = """## Original Issue
Title: {issue_title}
Description:
{issue_body}

## Pull Request
Title: {pr_title}
Description:
{pr_body}

## Code Changes (Diff)
{pr_diff}

## CI Results
Status: {ci_status}
Failed Checks: {failed_checks}

## Changed Files
{changed_files}

## Task
Review this pull request and determine if it should be approved or if changes are needed.

Consider:
1. Does the code fulfill the requirements from the original issue?
2. Are there any bugs, security issues, or code quality problems?
3. Are the CI checks passing?
4. Is the code properly tested?

Provide your review as a JSON object:
"""

PARSE_CODE_BLOCKS_PROMPT = """Extract all code blocks from the following LLM response.

For each code block, identify:
1. The filename (from the comment on the first line, or infer from context)
2. The code content

Response:
{response}

Return a JSON array with objects containing "filename" and "content" fields:
"""


def format_code_generation_prompt(
    issue_title: str,
    issue_body: str,
    repo_structure: list[str],
    existing_code: dict[str, str] | None = None,
) -> str:
    """Format the code generation prompt.

    Args:
        issue_title: The issue title.
        issue_body: The issue description.
        repo_structure: List of files in the repository.
        existing_code: Dict mapping file paths to their content.

    Returns:
        Formatted prompt string.
    """
    structure_str = "\n".join(f"- {f}" for f in repo_structure[:50])
    if len(repo_structure) > 50:
        structure_str += f"\n... and {len(repo_structure) - 50} more files"

    code_context = ""
    if existing_code:
        for path, content in existing_code.items():
            code_context += f"\n### {path}\n```python\n{content}\n```\n"

    return CODE_GENERATION_PROMPT.format(
        issue_title=issue_title,
        issue_body=issue_body,
        repo_structure=structure_str,
        existing_code=code_context or "No existing code context provided.",
    )


def format_code_iteration_prompt(
    issue_title: str,
    issue_body: str,
    previous_code: str,
    validation_errors: list[str],
) -> str:
    """Format the code iteration prompt for fixing errors.

    Args:
        issue_title: The issue title.
        issue_body: The issue description.
        previous_code: The code from the previous attempt.
        validation_errors: List of validation errors to fix.

    Returns:
        Formatted prompt string.
    """
    errors_str = "\n".join(f"- {e}" for e in validation_errors)

    return CODE_ITERATION_PROMPT.format(
        issue_title=issue_title,
        issue_body=issue_body,
        previous_code=previous_code,
        validation_errors=errors_str,
    )


def format_code_review_prompt(
    issue_title: str,
    issue_body: str,
    pr_title: str,
    pr_body: str,
    pr_diff: str,
    ci_status: str,
    failed_checks: list[str],
    changed_files: list[str],
) -> str:
    """Format the code review prompt.

    Args:
        issue_title: The original issue title.
        issue_body: The original issue description.
        pr_title: The PR title.
        pr_body: The PR description.
        pr_diff: The PR diff.
        ci_status: CI status (success/failure/pending).
        failed_checks: List of failed CI check names.
        changed_files: List of changed file paths.

    Returns:
        Formatted prompt string.
    """
    return CODE_REVIEW_PROMPT.format(
        issue_title=issue_title,
        issue_body=issue_body,
        pr_title=pr_title,
        pr_body=pr_body,
        pr_diff=pr_diff,
        ci_status=ci_status,
        failed_checks=", ".join(failed_checks) if failed_checks else "None",
        changed_files="\n".join(f"- {f}" for f in changed_files),
    )
