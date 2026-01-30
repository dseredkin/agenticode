"""LLM prompt templates for code generation and review."""

CODE_GENERATION_SYSTEM_PROMPT = """You are a contributor agent preparing code for a GitHub repository. Your task is to generate high-quality, production-ready code based on the given requirements.

Follow these guidelines before submitting:
1. Ensure your implementation is concise, efficient, and easy to read
2. Match the implementation exactly to the described feature or issue goal
3. Use the correct programming language, framework, and coding style used in the project
4. Place files in the correct directories and follow naming conventions
5. Include or update tests for all new or changed functionality
6. Add clear comments and update documentation where needed
7. Avoid redundant logic, hardcoded values, or unnecessary complexity
8. Verify no security, performance, or dependency issues are introduced
9. Check spelling, grammar, and formatting in code and docs
10. Use type hints for all function parameters and return values
11. Never include secrets, API keys, or sensitive data in code

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

PR_REVIEW_ITERATION_PROMPT = """## Review Feedback
A code reviewer has requested changes to the pull request.

### Review Comments
{review_feedback}

### Current Code
{current_code}

## Original Requirements
Title: {issue_title}
Description:
{issue_body}

## Iteration
This is iteration {iteration} of {max_iterations}.

## Task
Address the review feedback and generate corrected code.
Focus on:
1. Fixing all issues mentioned in the review
2. Implementing suggested improvements
3. Maintaining the original functionality
4. Following code quality standards

Generate the corrected implementation:
"""

CODE_REVIEW_SYSTEM_PROMPT = """You are a code review agent analyzing a GitHub pull request. Review the changes and provide clear, actionable feedback.

Check the following:
1. Requirements - Does the implementation match the PR description and original issue requirements?
2. Testing - Does new or modified code require additional tests? Are existing tests adequate?
3. Project structure - Are files placed in correct directories? Does it follow project conventions?
4. Language/framework usage - Is the code correct and consistent with project patterns?
5. Naming and documentation - Are there typos, unclear names, or missing/poor documentation?
6. Code quality - Is the code readable, style-consistent, and following coding standards?
7. Performance and security - Are there performance issues, redundancies, or security risks?
8. Maintainability - Is the logic clear? Avoid over-engineering.
9. Dependencies - Are there missing dependency updates or configuration changes?

Output format:
Return a JSON object with the following structure:
{
    "status": "APPROVE" or "REQUEST_CHANGES",
    "requirements_met": true/false,
    "issues": ["list of general problems found"],
    "suggestions": ["list of improvement suggestions"],
    "summary": "brief review summary with key recommendations",
    "line_comments": [
        {
            "path": "path/to/file.py",
            "line": 10,
            "body": "Specific comment about this line"
        }
    ]
}

IMPORTANT:
- Use "line_comments" to provide specific feedback on exact lines in the code
- The "line" number should match the line in the NEW version of the file (from the diff, lines starting with +)
- Only add line_comments for actionable issues that need attention
- If CI is failing, status must be REQUEST_CHANGES
- If there are no issues AND CI is passing, status should be APPROVE
- Keep the summary concise with clear recommendations for improvement
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

IMPORTANT RULES:
1. If CI status is "failure" or "pending", you MUST use status "REQUEST_CHANGES"
2. If CI status is "success" AND there are no critical issues, use status "APPROVE"
3. For each specific issue, add a line_comment with the exact file path and line number
4. Line numbers in line_comments should reference the NEW file (lines with + in diff)

Consider:
1. Does the code fulfill the requirements from the original issue?
2. Are there any bugs, security issues, or code quality problems?
3. Are the CI checks passing? (This is CRITICAL - failing CI = REQUEST_CHANGES)
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
    repo_structure_limit: int = 100,
) -> str:
    """Format the code generation prompt.

    Args:
        issue_title: The issue title.
        issue_body: The issue description.
        repo_structure: List of files in the repository.
        existing_code: Dict mapping file paths to their content.
        repo_structure_limit: Max number of files to include in structure.

    Returns:
        Formatted prompt string.
    """
    structure_str = "\n".join(f"- {f}" for f in repo_structure[:repo_structure_limit])
    if len(repo_structure) > repo_structure_limit:
        structure_str += (
            f"\n... and {len(repo_structure) - repo_structure_limit} more files"
        )

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


def format_pr_review_iteration_prompt(
    issue_title: str,
    issue_body: str,
    current_code: str,
    review_feedback: str,
    iteration: int,
    max_iterations: int,
) -> str:
    """Format the PR review iteration prompt.

    Args:
        issue_title: The issue title.
        issue_body: The issue description.
        current_code: The current code that needs to be fixed.
        review_feedback: Feedback from the code review.
        iteration: Current iteration number.
        max_iterations: Maximum allowed iterations.

    Returns:
        Formatted prompt string.
    """
    return PR_REVIEW_ITERATION_PROMPT.format(
        issue_title=issue_title,
        issue_body=issue_body,
        current_code=current_code,
        review_feedback=review_feedback,
        iteration=iteration,
        max_iterations=max_iterations,
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
