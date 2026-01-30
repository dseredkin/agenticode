"""Tests for code formatter."""

from unittest.mock import MagicMock, patch

import pytest

from agents.utils.code_formatter import (
    CodeFormatter,
    ValidationResult,
    ValidationSummary,
)


class TestValidationResult:
    """Tests for ValidationResult model."""

    def test_validation_result_success(self):
        """Test successful validation result."""
        result = ValidationResult(success=True)
        assert result.success
        assert result.errors == []
        assert result.warnings == []

    def test_validation_result_failure(self):
        """Test failed validation result."""
        result = ValidationResult(
            success=False,
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"],
        )
        assert not result.success
        assert len(result.errors) == 2
        assert len(result.warnings) == 1


class TestValidationSummary:
    """Tests for ValidationSummary model."""

    def test_all_errors(self):
        """Test aggregating errors from all checks."""
        summary = ValidationSummary(
            success=False,
            black_result=ValidationResult(success=False, errors=["black error"]),
            ruff_result=ValidationResult(success=False, errors=["ruff error"]),
            mypy_result=ValidationResult(success=False, errors=["mypy error"]),
        )

        errors = summary.all_errors
        assert len(errors) == 3
        assert any("black" in e for e in errors)
        assert any("ruff" in e for e in errors)
        assert any("mypy" in e for e in errors)

    def test_all_warnings(self):
        """Test aggregating warnings from all checks."""
        summary = ValidationSummary(
            success=True,
            black_result=ValidationResult(success=True, warnings=["black warn"]),
            ruff_result=ValidationResult(success=True, warnings=["ruff warn"]),
            mypy_result=ValidationResult(success=True, warnings=["mypy warn"]),
        )

        warnings = summary.all_warnings
        assert len(warnings) == 3


class TestCodeFormatter:
    """Tests for CodeFormatter class."""

    @pytest.fixture
    def formatter(self):
        """Create a CodeFormatter instance."""
        return CodeFormatter()

    @patch("subprocess.run")
    def test_format_code_success(self, mock_run, formatter, tmp_path):
        """Test successful code formatting."""
        mock_run.return_value = MagicMock(returncode=0)

        code = "def foo():pass"
        result = formatter.format_code(code)

        assert result.success
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_format_code_failure(self, mock_run, formatter):
        """Test failed code formatting."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr="Syntax error",
        )

        code = "def foo(:"
        result = formatter.format_code(code)

        assert not result.success
        assert len(result.errors) > 0

    @patch("subprocess.run")
    def test_lint_code_with_fix(self, mock_run, formatter):
        """Test linting with auto-fix."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")

        code = "import os\nimport sys\n"
        result = formatter.lint_code(code, fix=True)

        assert result.success
        assert mock_run.call_count >= 1

    @patch("subprocess.run")
    def test_lint_code_errors(self, mock_run, formatter):
        """Test linting with errors."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="test.py:1:1: F401 'os' imported but unused",
        )

        code = "import os\n"
        result = formatter.lint_code(code, fix=False)

        assert not result.success
        assert len(result.errors) > 0

    @patch("subprocess.run")
    def test_type_check(self, mock_run, formatter, tmp_path):
        """Test type checking."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Success: no issues found",
        )

        test_file = tmp_path / "test.py"
        test_file.write_text("def foo() -> str: return 'hello'")

        result = formatter.type_check([test_file])

        assert result.success

    @patch("subprocess.run")
    def test_type_check_errors(self, mock_run, formatter, tmp_path):
        """Test type checking with errors."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="test.py:1: error: Missing return type",
        )

        test_file = tmp_path / "test.py"
        test_file.write_text("def foo(): pass")

        result = formatter.type_check([test_file])

        assert not result.success
        assert len(result.errors) > 0

    @patch("subprocess.run")
    def test_validate_all(self, mock_run, formatter):
        """Test running all validation checks."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        code = "def foo() -> str:\n    return 'hello'\n"
        result = formatter.validate_all(code)

        assert isinstance(result, ValidationSummary)

    @patch("subprocess.run")
    def test_validate_files(self, mock_run, formatter, tmp_path):
        """Test validating files."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        test_file = tmp_path / "test.py"
        test_file.write_text("def foo() -> str:\n    return 'hello'\n")

        result = formatter.validate_files([test_file])

        assert isinstance(result, ValidationSummary)
