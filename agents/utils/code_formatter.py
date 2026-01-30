"""Code formatting, linting, and type checking utilities."""

import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of code validation."""

    success: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    formatted_code: str | None = None


@dataclass
class ValidationSummary:
    """Summary of all validation checks."""

    success: bool
    black_result: ValidationResult
    ruff_result: ValidationResult
    mypy_result: ValidationResult

    @property
    def all_errors(self) -> list[str]:
        """Get all errors from all checks."""
        errors: list[str] = []
        if not self.black_result.success:
            errors.extend([f"[black] {e}" for e in self.black_result.errors])
        if not self.ruff_result.success:
            errors.extend([f"[ruff] {e}" for e in self.ruff_result.errors])
        if not self.mypy_result.success:
            errors.extend([f"[mypy] {e}" for e in self.mypy_result.errors])
        return errors

    @property
    def all_warnings(self) -> list[str]:
        """Get all warnings from all checks."""
        warnings: list[str] = []
        warnings.extend([f"[black] {w}" for w in self.black_result.warnings])
        warnings.extend([f"[ruff] {w}" for w in self.ruff_result.warnings])
        warnings.extend([f"[mypy] {w}" for w in self.mypy_result.warnings])
        return warnings


class CodeFormatter:
    """Code formatting, linting, and type checking integration."""

    def __init__(self, project_root: Path | None = None) -> None:
        """Initialize code formatter.

        Args:
            project_root: Root directory of the project for config discovery.
        """
        self._project_root = project_root or Path.cwd()

    def format_code(self, code: str, filename: str = "code.py") -> ValidationResult:
        """Format code using Black.

        Args:
            code: Python code to format.
            filename: Filename hint for Black.

        Returns:
            ValidationResult with formatted code.
        """
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
        ) as f:
            f.write(code)
            temp_path = Path(f.name)

        try:
            result = subprocess.run(
                ["black", "--quiet", str(temp_path)],
                capture_output=True,
                text=True,
                cwd=self._project_root,
            )

            if result.returncode == 0:
                formatted_code = temp_path.read_text()
                return ValidationResult(
                    success=True,
                    formatted_code=formatted_code,
                )
            else:
                errors = result.stderr.strip().split("\n") if result.stderr else []
                return ValidationResult(
                    success=False,
                    errors=[e for e in errors if e],
                    formatted_code=code,
                )
        finally:
            temp_path.unlink(missing_ok=True)

    def lint_code(
        self,
        code: str,
        filename: str = "code.py",
        fix: bool = True,
    ) -> ValidationResult:
        """Lint code using Ruff.

        Args:
            code: Python code to lint.
            filename: Filename hint for Ruff.
            fix: Whether to auto-fix issues.

        Returns:
            ValidationResult with linted code and any remaining issues.
        """
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
        ) as f:
            f.write(code)
            temp_path = Path(f.name)

        try:
            if fix:
                subprocess.run(
                    ["ruff", "check", "--fix", "--quiet", str(temp_path)],
                    capture_output=True,
                    text=True,
                    cwd=self._project_root,
                )
                code = temp_path.read_text()

            result = subprocess.run(
                ["ruff", "check", "--output-format=text", str(temp_path)],
                capture_output=True,
                text=True,
                cwd=self._project_root,
            )

            errors: list[str] = []
            warnings: list[str] = []

            if result.stdout:
                for line in result.stdout.strip().split("\n"):
                    if line:
                        if ": error" in line.lower():
                            errors.append(line)
                        else:
                            errors.append(line)

            return ValidationResult(
                success=result.returncode == 0,
                errors=errors,
                warnings=warnings,
                formatted_code=code,
            )
        finally:
            temp_path.unlink(missing_ok=True)

    def type_check(self, files: list[Path]) -> ValidationResult:
        """Type check files using Mypy.

        Args:
            files: List of file paths to check.

        Returns:
            ValidationResult with type check results.
        """
        if not files:
            return ValidationResult(success=True)

        result = subprocess.run(
            ["mypy", "--ignore-missing-imports", *[str(f) for f in files]],
            capture_output=True,
            text=True,
            cwd=self._project_root,
        )

        errors: list[str] = []
        warnings: list[str] = []

        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                if line and not line.startswith("Success"):
                    if ": error:" in line:
                        errors.append(line)
                    elif ": warning:" in line or ": note:" in line:
                        warnings.append(line)

        return ValidationResult(
            success=result.returncode == 0,
            errors=errors,
            warnings=warnings,
        )

    def type_check_code(self, code: str, filename: str = "code.py") -> ValidationResult:
        """Type check code string using Mypy.

        Args:
            code: Python code to check.
            filename: Filename hint for error messages.

        Returns:
            ValidationResult with type check results.
        """
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
        ) as f:
            f.write(code)
            temp_path = Path(f.name)

        try:
            return self.type_check([temp_path])
        finally:
            temp_path.unlink(missing_ok=True)

    def validate_all(
        self,
        code: str,
        filename: str = "code.py",
        fix: bool = True,
    ) -> ValidationSummary:
        """Run all validation checks on code.

        Args:
            code: Python code to validate.
            filename: Filename hint.
            fix: Whether to auto-fix formatting/linting issues.

        Returns:
            ValidationSummary with all check results.
        """
        black_result = self.format_code(code, filename)
        current_code = black_result.formatted_code or code

        ruff_result = self.lint_code(current_code, filename, fix=fix)
        current_code = ruff_result.formatted_code or current_code

        mypy_result = self.type_check_code(current_code, filename)

        success = black_result.success and ruff_result.success and mypy_result.success

        return ValidationSummary(
            success=success,
            black_result=ValidationResult(
                success=black_result.success,
                errors=black_result.errors,
                warnings=black_result.warnings,
                formatted_code=current_code,
            ),
            ruff_result=ruff_result,
            mypy_result=mypy_result,
        )

    def validate_files(self, files: list[Path]) -> ValidationSummary:
        """Run all validation checks on files.

        Args:
            files: List of file paths to validate.

        Returns:
            ValidationSummary with all check results.
        """
        black_result = self._run_black_on_files(files)
        ruff_result = self._run_ruff_on_files(files)
        mypy_result = self.type_check(files)

        success = black_result.success and ruff_result.success and mypy_result.success

        return ValidationSummary(
            success=success,
            black_result=black_result,
            ruff_result=ruff_result,
            mypy_result=mypy_result,
        )

    def _run_black_on_files(self, files: list[Path]) -> ValidationResult:
        """Run Black check on files without modifying them.

        Args:
            files: List of file paths to check.

        Returns:
            ValidationResult with check results.
        """
        if not files:
            return ValidationResult(success=True)

        result = subprocess.run(
            ["black", "--check", "--quiet", *[str(f) for f in files]],
            capture_output=True,
            text=True,
            cwd=self._project_root,
        )

        errors: list[str] = []
        if result.returncode != 0:
            if result.stderr:
                errors = [line for line in result.stderr.strip().split("\n") if line]
            else:
                errors = ["Files need formatting"]

        return ValidationResult(
            success=result.returncode == 0,
            errors=errors,
        )

    def _run_ruff_on_files(self, files: list[Path]) -> ValidationResult:
        """Run Ruff check on files.

        Args:
            files: List of file paths to check.

        Returns:
            ValidationResult with check results.
        """
        if not files:
            return ValidationResult(success=True)

        result = subprocess.run(
            ["ruff", "check", "--output-format=text", *[str(f) for f in files]],
            capture_output=True,
            text=True,
            cwd=self._project_root,
        )

        errors: list[str] = []
        if result.stdout:
            errors = [line for line in result.stdout.strip().split("\n") if line]

        return ValidationResult(
            success=result.returncode == 0,
            errors=errors,
        )
