"""Tests for Code Agent."""

from unittest.mock import MagicMock, patch

import pytest

from agents.code_agent import (
    CodeAgent,
    GeneratedFile,
    GenerationResult,
    IterationResult,
)
from agents.utils.github_client import IssueDetails


class TestGeneratedFile:
    """Tests for GeneratedFile model."""

    def test_generated_file(self):
        """Test GeneratedFile creation."""
        file = GeneratedFile(path="src/test.py", content="print('hello')")
        assert file.path == "src/test.py"
        assert file.content == "print('hello')"


class TestIterationResult:
    """Tests for IterationResult model."""

    def test_iteration_result_success(self):
        """Test successful iteration result."""
        result = IterationResult(
            iteration=1,
            files=[GeneratedFile(path="test.py", content="code")],
            validation=None,
            success=True,
        )
        assert result.success
        assert result.iteration == 1

    def test_iteration_result_failure(self):
        """Test failed iteration result."""
        result = IterationResult(
            iteration=2,
            files=[],
            validation=None,
            success=False,
            errors=["Syntax error"],
        )
        assert not result.success
        assert "Syntax error" in result.errors


class TestCodeAgent:
    """Tests for CodeAgent class."""

    @pytest.fixture
    def mock_github(self):
        """Create a mock GitHub client."""
        client = MagicMock()
        client.get_issue.return_value = IssueDetails(
            number=1,
            title="Test Issue",
            body="Implement a hello function",
            labels=[],
            state="open",
        )
        client.get_repo_structure.return_value = ["src/app.py"]
        client.get_file_content.return_value = None
        client.create_branch.return_value = "refs/heads/auto/issue-1"
        client.create_pr.return_value = 42
        return client

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM client."""
        client = MagicMock()
        client.generate_code.return_value = '''```python
# src/hello.py
def hello() -> str:
    """Return a greeting."""
    return "Hello, World!"
```'''
        return client

    @pytest.fixture
    def mock_formatter(self):
        """Create a mock code formatter."""
        formatter = MagicMock()
        formatter.validate_all.return_value = MagicMock(
            success=True,
            all_errors=[],
            black_result=MagicMock(
                formatted_code="def hello() -> str:\n    return 'Hello'"
            ),
        )
        return formatter

    @pytest.fixture
    def agent(self, mock_github, mock_llm, mock_formatter):
        """Create a CodeAgent with mocked dependencies."""
        return CodeAgent(
            github_client=mock_github,
            llm_client=mock_llm,
            code_formatter=mock_formatter,
            max_iterations=3,
        )

    def test_run_success(self, agent, mock_github):
        """Test successful code generation."""
        result = agent.run(1)

        assert result.success
        assert result.pr_number == 42
        assert len(result.final_files) > 0
        mock_github.create_pr.assert_called_once()

    def test_run_issue_not_found(self, agent, mock_github):
        """Test handling of missing issue."""
        mock_github.get_issue.side_effect = Exception("Issue not found")

        result = agent.run(999)

        assert not result.success
        assert "Failed to fetch issue" in result.error

    def test_run_max_iterations(self, agent, mock_llm, mock_formatter):
        """Test max iterations limit."""
        mock_formatter.validate_all.return_value = MagicMock(
            success=False,
            all_errors=["Type error"],
            black_result=MagicMock(formatted_code="invalid code"),
        )

        result = agent.run(1)

        assert not result.success
        assert len(result.iterations) == 3

    def test_parse_code_response(self, agent):
        """Test parsing LLM response."""
        response = """Here is the code:

```python
# src/module.py
def greet(name: str) -> str:
    return f"Hello, {name}"
```

And here is another file:

```python
# tests/test_module.py
def test_greet():
    assert greet("World") == "Hello, World"
```
"""
        files = agent._parse_code_response(response)

        assert len(files) == 2
        assert files[0].path == "src/module.py"
        assert files[1].path == "tests/test_module.py"

    def test_parse_code_response_no_filename(self, agent):
        """Test parsing response without filename comments."""
        response = """```python
def hello():
    pass
```"""
        files = agent._parse_code_response(response)

        assert len(files) == 1
        assert files[0].path.startswith("src/generated_")

    def test_extract_keywords(self, agent):
        """Test keyword extraction."""
        text = "Implement a UserService class that handles authentication"
        keywords = agent._extract_keywords(text)

        assert "UserService" in keywords
        assert "authentication" in keywords
        assert "a" not in keywords
        assert "that" not in keywords

    def test_run_pr_creation_failure(self, agent, mock_github, mock_llm):
        """Test handling of PR creation failure."""
        mock_github.create_pr.side_effect = Exception("Permission denied")

        result = agent.run(1)

        assert not result.success
        assert "Failed to create PR" in result.error


class TestCodeAgentIntegration:
    """Integration-style tests for CodeAgent."""

    @patch("agents.code_agent.GitHubClient")
    @patch("agents.code_agent.LLMClient")
    @patch("agents.code_agent.CodeFormatter")
    def test_full_flow(self, mock_formatter_cls, mock_llm_cls, mock_github_cls):
        """Test full code generation flow."""
        mock_github = MagicMock()
        mock_github.get_issue.return_value = IssueDetails(
            number=5,
            title="Add calculator",
            body="Create add and subtract functions",
            labels=[],
            state="open",
        )
        mock_github.get_repo_structure.return_value = []
        mock_github.get_file_content.return_value = None
        mock_github.create_branch.return_value = "refs/heads/auto/issue-5"
        mock_github.create_pr.return_value = 10
        mock_github_cls.return_value = mock_github

        mock_llm = MagicMock()
        mock_llm.generate_code.return_value = """```python
# src/calculator.py
def add(a: int, b: int) -> int:
    return a + b

def subtract(a: int, b: int) -> int:
    return a - b
```"""
        mock_llm_cls.return_value = mock_llm

        mock_formatter = MagicMock()
        mock_formatter.validate_all.return_value = MagicMock(
            success=True,
            all_errors=[],
            black_result=MagicMock(formatted_code="formatted code"),
        )
        mock_formatter_cls.return_value = mock_formatter

        agent = CodeAgent()
        result = agent.run(5)

        assert result.success
        assert result.pr_number == 10
