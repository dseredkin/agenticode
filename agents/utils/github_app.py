"""GitHub App authentication utilities."""

import logging
import time
from pathlib import Path

import httpx
import jwt

logger = logging.getLogger(__name__)


def generate_jwt(app_id: str, private_key: str) -> str:
    """Generate a JWT for GitHub App authentication.

    Args:
        app_id: The GitHub App ID.
        private_key: The private key content (PEM format).

    Returns:
        JWT token string.
    """
    now = int(time.time())
    payload = {
        "iat": now - 60,  # Issued 60 seconds ago (clock drift)
        "exp": now + 600,  # Expires in 10 minutes
        "iss": app_id,
    }
    return jwt.encode(payload, private_key, algorithm="RS256")


def get_installation_token(
    app_id: str,
    private_key: str,
    installation_id: str,
) -> str:
    """Get an installation access token for a GitHub App.

    Args:
        app_id: The GitHub App ID.
        private_key: The private key content (PEM format).
        installation_id: The installation ID.

    Returns:
        Installation access token.
    """
    jwt_token = generate_jwt(app_id, private_key)

    url = f"https://api.github.com/app/installations/{installation_id}/access_tokens"
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    response = httpx.post(url, headers=headers)
    response.raise_for_status()

    data = response.json()
    token: str = data["token"]
    expires_at = data.get("expires_at", "unknown")
    logger.info(f"Generated installation token (expires: {expires_at})")

    return token


def load_private_key(path: str) -> str:
    """Load private key from file.

    Args:
        path: Path to the .pem file.

    Returns:
        Private key content.
    """
    return Path(path).read_text()


def get_app_token_from_env() -> str | None:
    """Get GitHub App installation token from environment variables.

    Required env vars:
        GITHUB_APP_ID: The App ID
        GITHUB_APP_PRIVATE_KEY: Path to .pem file OR the key content
        GITHUB_APP_INSTALLATION_ID: The installation ID

    Returns:
        Installation access token, or None if env vars not set.
    """
    import os

    app_id = os.environ.get("GITHUB_APP_ID")
    private_key_env = os.environ.get("GITHUB_APP_PRIVATE_KEY")
    installation_id = os.environ.get("GITHUB_APP_INSTALLATION_ID")

    if not app_id or not private_key_env or not installation_id:
        return None

    # Check if it's a path or the key content
    if private_key_env.startswith("-----BEGIN"):
        private_key = private_key_env
    else:
        private_key = load_private_key(private_key_env)

    return get_installation_token(app_id, private_key, installation_id)
