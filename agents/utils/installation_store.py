"""Installation storage for multi-tenant GitHub App support."""

import logging
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.environ.get(
    "INSTALLATIONS_DB_PATH",
    str(Path(__file__).parent.parent.parent / "data" / "installations.db"),
)


@dataclass
class Installation:
    """GitHub App installation record."""

    installation_id: int
    account_login: str
    account_type: str  # "User" or "Organization"
    created_at: datetime
    suspended_at: datetime | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Installation":
        """Create Installation from database row."""
        return cls(
            installation_id=row["installation_id"],
            account_login=row["account_login"],
            account_type=row["account_type"],
            created_at=datetime.fromisoformat(row["created_at"]),
            suspended_at=(
                datetime.fromisoformat(row["suspended_at"])
                if row["suspended_at"]
                else None
            ),
        )


@dataclass
class InstallationRepository:
    """Repository associated with an installation."""

    installation_id: int
    repo_full_name: str  # owner/repo format
    added_at: datetime

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "InstallationRepository":
        """Create InstallationRepository from database row."""
        return cls(
            installation_id=row["installation_id"],
            repo_full_name=row["repo_full_name"],
            added_at=datetime.fromisoformat(row["added_at"]),
        )


class InstallationStore:
    """SQLite-backed storage for GitHub App installations."""

    def __init__(self, db_path: str | None = None) -> None:
        """Initialize installation store.

        Args:
            db_path: Path to SQLite database file.
        """
        self._db_path = db_path or DEFAULT_DB_PATH
        db_dir = Path(self._db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS installations (
                    installation_id INTEGER PRIMARY KEY,
                    account_login TEXT NOT NULL,
                    account_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    suspended_at TEXT
                );

                CREATE TABLE IF NOT EXISTS installation_repositories (
                    installation_id INTEGER NOT NULL,
                    repo_full_name TEXT NOT NULL,
                    added_at TEXT NOT NULL,
                    PRIMARY KEY (installation_id, repo_full_name),
                    FOREIGN KEY (installation_id) REFERENCES installations(installation_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_repo_full_name
                    ON installation_repositories(repo_full_name);
            """)

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Get database connection with row factory."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def add_installation(
        self,
        installation_id: int,
        account_login: str,
        account_type: str,
        repositories: list[str] | None = None,
    ) -> Installation:
        """Add or update an installation.

        Args:
            installation_id: GitHub installation ID.
            account_login: Account name (user or org).
            account_type: "User" or "Organization".
            repositories: List of repo full names (owner/repo).

        Returns:
            The created/updated Installation.
        """
        now = datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO installations (installation_id, account_login, account_type, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(installation_id) DO UPDATE SET
                    account_login = excluded.account_login,
                    account_type = excluded.account_type,
                    suspended_at = NULL
                """,
                (installation_id, account_login, account_type, now),
            )

            if repositories:
                for repo in repositories:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO installation_repositories
                            (installation_id, repo_full_name, added_at)
                        VALUES (?, ?, ?)
                        """,
                        (installation_id, repo, now),
                    )

        logger.info(
            f"Added installation {installation_id} for {account_login} "
            f"with {len(repositories or [])} repositories"
        )

        return Installation(
            installation_id=installation_id,
            account_login=account_login,
            account_type=account_type,
            created_at=datetime.fromisoformat(now),
        )

    def remove_installation(self, installation_id: int) -> bool:
        """Remove an installation and its repositories.

        Args:
            installation_id: GitHub installation ID.

        Returns:
            True if installation was removed, False if not found.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM installations WHERE installation_id = ?",
                (installation_id,),
            )

        removed = cursor.rowcount > 0
        if removed:
            logger.info(f"Removed installation {installation_id}")
        return removed

    def suspend_installation(self, installation_id: int) -> bool:
        """Mark an installation as suspended.

        Args:
            installation_id: GitHub installation ID.

        Returns:
            True if installation was suspended, False if not found.
        """
        now = datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            cursor = conn.execute(
                "UPDATE installations SET suspended_at = ? WHERE installation_id = ?",
                (now, installation_id),
            )

        suspended = cursor.rowcount > 0
        if suspended:
            logger.info(f"Suspended installation {installation_id}")
        return suspended

    def unsuspend_installation(self, installation_id: int) -> bool:
        """Remove suspension from an installation.

        Args:
            installation_id: GitHub installation ID.

        Returns:
            True if installation was unsuspended, False if not found.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "UPDATE installations SET suspended_at = NULL WHERE installation_id = ?",
                (installation_id,),
            )

        unsuspended = cursor.rowcount > 0
        if unsuspended:
            logger.info(f"Unsuspended installation {installation_id}")
        return unsuspended

    def get_installation(self, installation_id: int) -> Installation | None:
        """Get an installation by ID.

        Args:
            installation_id: GitHub installation ID.

        Returns:
            Installation if found, None otherwise.
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM installations WHERE installation_id = ?",
                (installation_id,),
            ).fetchone()

        return Installation.from_row(row) if row else None

    def get_installation_for_repo(self, repo_full_name: str) -> Installation | None:
        """Get the installation for a specific repository.

        Args:
            repo_full_name: Repository in owner/repo format.

        Returns:
            Installation if found, None otherwise.
        """
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT i.* FROM installations i
                JOIN installation_repositories r ON i.installation_id = r.installation_id
                WHERE r.repo_full_name = ?
                """,
                (repo_full_name,),
            ).fetchone()

        return Installation.from_row(row) if row else None

    def add_repositories(
        self,
        installation_id: int,
        repositories: list[str],
    ) -> int:
        """Add repositories to an installation.

        Args:
            installation_id: GitHub installation ID.
            repositories: List of repo full names.

        Returns:
            Number of repositories added.
        """
        now = datetime.utcnow().isoformat()
        added = 0

        with self._get_connection() as conn:
            for repo in repositories:
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO installation_repositories
                        (installation_id, repo_full_name, added_at)
                    VALUES (?, ?, ?)
                    """,
                    (installation_id, repo, now),
                )
                added += cursor.rowcount

        logger.info(f"Added {added} repositories to installation {installation_id}")
        return added

    def remove_repositories(
        self,
        installation_id: int,
        repositories: list[str],
    ) -> int:
        """Remove repositories from an installation.

        Args:
            installation_id: GitHub installation ID.
            repositories: List of repo full names.

        Returns:
            Number of repositories removed.
        """
        removed = 0

        with self._get_connection() as conn:
            for repo in repositories:
                cursor = conn.execute(
                    """
                    DELETE FROM installation_repositories
                    WHERE installation_id = ? AND repo_full_name = ?
                    """,
                    (installation_id, repo),
                )
                removed += cursor.rowcount

        logger.info(
            f"Removed {removed} repositories from installation {installation_id}"
        )
        return removed

    def get_repositories(self, installation_id: int) -> list[InstallationRepository]:
        """Get all repositories for an installation.

        Args:
            installation_id: GitHub installation ID.

        Returns:
            List of InstallationRepository objects.
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM installation_repositories WHERE installation_id = ?",
                (installation_id,),
            ).fetchall()

        return [InstallationRepository.from_row(row) for row in rows]

    def list_installations(self, include_suspended: bool = False) -> list[Installation]:
        """List all installations.

        Args:
            include_suspended: Whether to include suspended installations.

        Returns:
            List of Installation objects.
        """
        with self._get_connection() as conn:
            if include_suspended:
                rows = conn.execute("SELECT * FROM installations").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM installations WHERE suspended_at IS NULL"
                ).fetchall()

        return [Installation.from_row(row) for row in rows]

    def is_repo_registered(self, repo_full_name: str) -> bool:
        """Check if a repository is registered with any installation.

        Args:
            repo_full_name: Repository in owner/repo format.

        Returns:
            True if repository is registered.
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM installation_repositories WHERE repo_full_name = ?",
                (repo_full_name,),
            ).fetchone()

        return row is not None


# Module-level singleton
_store: InstallationStore | None = None


def get_installation_store() -> InstallationStore:
    """Get or create the installation store singleton."""
    global _store
    if _store is None:
        _store = InstallationStore()
    return _store
