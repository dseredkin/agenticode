"""Load tests for the webhook server using Locust.

Run with:
    locust -f tests/load/locustfile.py --host=http://localhost:8000

Or headless:
    locust -f tests/load/locustfile.py --host=http://localhost:8000 \
        --headless -u 100 -r 10 -t 60s
"""

import hashlib
import hmac
import json
import os
import random

from locust import HttpUser, between, task


def generate_signature(payload: bytes, secret: str) -> str:
    """Generate GitHub webhook signature."""
    signature = hmac.new(
        secret.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()
    return f"sha256={signature}"


class WebhookUser(HttpUser):
    """Simulates GitHub webhook traffic."""

    wait_time = between(0.1, 0.5)
    webhook_secret = os.environ.get("WEBHOOK_SECRET", "test-secret")

    def _make_webhook_request(self, event_type: str, payload: dict) -> None:
        """Make a webhook request with proper headers."""
        body = json.dumps(payload).encode("utf-8")
        signature = generate_signature(body, self.webhook_secret)

        self.client.post(
            "/webhook",
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-GitHub-Event": event_type,
                "X-Hub-Signature-256": signature,
                "X-GitHub-Delivery": f"test-{random.randint(1000, 9999)}",
            },
            name=f"/webhook [{event_type}]",
        )

    @task(10)
    def health_check(self) -> None:
        """Test health endpoint - high frequency."""
        self.client.get("/health")

    @task(5)
    def budget_status(self) -> None:
        """Test budget endpoint."""
        self.client.get("/budget")

    @task(3)
    def queue_stats(self) -> None:
        """Test queue stats endpoint."""
        self.client.get("/queue/stats")

    @task(2)
    def webhook_ping(self) -> None:
        """Test webhook with ping event (ignored)."""
        self._make_webhook_request("ping", {"zen": "Load test ping"})

    @task(2)
    def webhook_push(self) -> None:
        """Test webhook with push event (ignored)."""
        self._make_webhook_request(
            "push",
            {
                "ref": "refs/heads/main",
                "repository": {"full_name": "test/repo"},
            },
        )

    @task(1)
    def webhook_issue_opened(self) -> None:
        """Test webhook with issue opened event."""
        self._make_webhook_request(
            "issues",
            {
                "action": "opened",
                "issue": {
                    "number": random.randint(1, 1000),
                    "title": "Load test issue",
                    "body": "This is a load test issue",
                },
                "repository": {
                    "full_name": "test/repo",
                    "default_branch": "main",
                },
                "installation": {"id": 12345},
            },
        )

    @task(1)
    def webhook_pr_opened(self) -> None:
        """Test webhook with PR opened event."""
        self._make_webhook_request(
            "pull_request",
            {
                "action": "opened",
                "pull_request": {
                    "number": random.randint(1, 1000),
                    "title": "feat: Load test PR",
                    "body": "Load test PR body",
                    "head": {"ref": "feature-branch"},
                    "base": {"ref": "main"},
                },
                "repository": {
                    "full_name": "test/repo",
                    "default_branch": "main",
                },
                "installation": {"id": 12345},
            },
        )

    @task(1)
    def webhook_issue_comment(self) -> None:
        """Test webhook with issue comment event (ignored)."""
        self._make_webhook_request(
            "issue_comment",
            {
                "action": "created",
                "issue": {"number": 1},
                "comment": {"body": "Test comment"},
                "repository": {"full_name": "test/repo"},
            },
        )


class HealthCheckUser(HttpUser):
    """Lightweight user that only hits health endpoints."""

    wait_time = between(0.05, 0.1)
    weight = 3

    @task
    def health_check(self) -> None:
        """Rapid health checks."""
        self.client.get("/health")


class BurstUser(HttpUser):
    """Simulates burst traffic patterns."""

    wait_time = between(0, 0.01)
    weight = 1

    @task(5)
    def health_burst(self) -> None:
        """Burst health checks."""
        self.client.get("/health")

    @task(3)
    def stats_burst(self) -> None:
        """Burst stats checks."""
        self.client.get("/queue/stats")

    @task(2)
    def budget_burst(self) -> None:
        """Burst budget checks."""
        self.client.get("/budget")
