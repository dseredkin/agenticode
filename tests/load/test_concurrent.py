"""Concurrent load tests using pytest and threading.

Run with:
    pytest tests/load/test_concurrent.py -v -s
"""

import hashlib
import hmac
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import pytest

# Skip if server is not running
pytestmark = pytest.mark.skipif(
    os.environ.get("LOAD_TEST_URL") is None,
    reason="Set LOAD_TEST_URL to run load tests (e.g., http://localhost:8000)",
)


@dataclass
class LoadTestResult:
    """Results from a load test run."""

    total_requests: int
    successful: int
    failed: int
    total_time: float
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    requests_per_second: float

    def __str__(self) -> str:
        return (
            f"Requests: {self.total_requests} "
            f"(success: {self.successful}, failed: {self.failed})\n"
            f"Time: {self.total_time:.2f}s\n"
            f"Response time: avg={self.avg_response_time*1000:.1f}ms, "
            f"min={self.min_response_time*1000:.1f}ms, "
            f"max={self.max_response_time*1000:.1f}ms\n"
            f"Throughput: {self.requests_per_second:.1f} req/s"
        )


class LoadTester:
    """Simple load tester using threads."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.webhook_secret = os.environ.get("WEBHOOK_SECRET", "test-secret")

    def _make_request(
        self,
        method: str,
        path: str,
        headers: dict | None = None,
        data: bytes | None = None,
    ) -> tuple[bool, float]:
        """Make a request and return (success, response_time)."""
        import httpx

        url = f"{self.base_url}{path}"
        start = time.perf_counter()
        try:
            with httpx.Client(timeout=10.0) as client:
                if method == "GET":
                    response = client.get(url, headers=headers)
                else:
                    response = client.post(url, headers=headers, content=data)
                elapsed = time.perf_counter() - start
                return response.status_code < 500, elapsed
        except Exception:
            elapsed = time.perf_counter() - start
            return False, elapsed

    def _generate_signature(self, payload: bytes) -> str:
        """Generate GitHub webhook signature."""
        signature = hmac.new(
            self.webhook_secret.encode("utf-8"),
            payload,
            hashlib.sha256,
        ).hexdigest()
        return f"sha256={signature}"

    def run_load_test(
        self,
        request_func,
        num_requests: int = 100,
        concurrency: int = 10,
    ) -> LoadTestResult:
        """Run load test with given request function."""
        results: list[tuple[bool, float]] = []
        lock = threading.Lock()

        def worker():
            success, elapsed = request_func()
            with lock:
                results.append((success, elapsed))

        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(worker) for _ in range(num_requests)]
            for future in as_completed(futures):
                future.result()

        total_time = time.perf_counter() - start_time

        successful = sum(1 for success, _ in results if success)
        failed = len(results) - successful
        response_times = [elapsed for _, elapsed in results]

        return LoadTestResult(
            total_requests=len(results),
            successful=successful,
            failed=failed,
            total_time=total_time,
            avg_response_time=sum(response_times) / len(response_times),
            min_response_time=min(response_times),
            max_response_time=max(response_times),
            requests_per_second=len(results) / total_time,
        )


@pytest.fixture
def load_tester():
    """Create a load tester instance."""
    base_url = os.environ.get("LOAD_TEST_URL", "http://localhost:8000")
    return LoadTester(base_url)


class TestHealthEndpoint:
    """Load tests for health endpoint."""

    def test_health_100_requests(self, load_tester):
        """Test health endpoint with 100 concurrent requests."""

        def make_request():
            return load_tester._make_request("GET", "/health")

        result = load_tester.run_load_test(
            make_request, num_requests=100, concurrency=10
        )
        print(f"\n{result}")

        assert result.failed == 0, f"Had {result.failed} failed requests"
        assert result.avg_response_time < 1.0, "Average response time > 1s"

    def test_health_500_requests_high_concurrency(self, load_tester):
        """Test health endpoint under high concurrency."""

        def make_request():
            return load_tester._make_request("GET", "/health")

        result = load_tester.run_load_test(
            make_request, num_requests=500, concurrency=50
        )
        print(f"\n{result}")

        assert result.successful / result.total_requests > 0.95, "Success rate < 95%"


class TestQueueStatsEndpoint:
    """Load tests for queue stats endpoint."""

    def test_queue_stats_100_requests(self, load_tester):
        """Test queue stats endpoint with 100 requests."""

        def make_request():
            return load_tester._make_request("GET", "/queue/stats")

        result = load_tester.run_load_test(
            make_request, num_requests=100, concurrency=10
        )
        print(f"\n{result}")

        assert result.failed == 0, f"Had {result.failed} failed requests"


class TestWebhookEndpoint:
    """Load tests for webhook endpoint."""

    def test_webhook_ping_100_requests(self, load_tester):
        """Test webhook with ping events."""
        payload = json.dumps({"zen": "Load test"}).encode()
        signature = load_tester._generate_signature(payload)

        def make_request():
            return load_tester._make_request(
                "POST",
                "/webhook",
                headers={
                    "Content-Type": "application/json",
                    "X-GitHub-Event": "ping",
                    "X-Hub-Signature-256": signature,
                    "X-GitHub-Delivery": "test-delivery",
                },
                data=payload,
            )

        result = load_tester.run_load_test(
            make_request, num_requests=100, concurrency=10
        )
        print(f"\n{result}")

        assert result.failed == 0, f"Had {result.failed} failed requests"

    def test_webhook_mixed_events(self, load_tester):
        """Test webhook with mixed event types."""
        import random

        events = [
            ("ping", {"zen": "test"}),
            ("push", {"ref": "refs/heads/main"}),
            ("issue_comment", {"action": "created", "issue": {"number": 1}}),
        ]

        def make_request():
            event_type, payload_dict = random.choice(events)
            payload = json.dumps(payload_dict).encode()
            signature = load_tester._generate_signature(payload)
            return load_tester._make_request(
                "POST",
                "/webhook",
                headers={
                    "Content-Type": "application/json",
                    "X-GitHub-Event": event_type,
                    "X-Hub-Signature-256": signature,
                    "X-GitHub-Delivery": f"test-{random.randint(1000, 9999)}",
                },
                data=payload,
            )

        result = load_tester.run_load_test(
            make_request, num_requests=200, concurrency=20
        )
        print(f"\n{result}")

        assert result.successful / result.total_requests > 0.95, "Success rate < 95%"


class TestMixedLoad:
    """Mixed load tests simulating real traffic patterns."""

    def test_realistic_traffic_pattern(self, load_tester):
        """Test with realistic mix of endpoints."""
        import random

        def make_request():
            # Weighted random choice: health 50%, stats 30%, webhook 20%
            r = random.random()
            if r < 0.5:
                return load_tester._make_request("GET", "/health")
            elif r < 0.8:
                return load_tester._make_request("GET", "/queue/stats")
            else:
                payload = json.dumps({"zen": "test"}).encode()
                signature = load_tester._generate_signature(payload)
                return load_tester._make_request(
                    "POST",
                    "/webhook",
                    headers={
                        "Content-Type": "application/json",
                        "X-GitHub-Event": "ping",
                        "X-Hub-Signature-256": signature,
                        "X-GitHub-Delivery": f"test-{random.randint(1000, 9999)}",
                    },
                    data=payload,
                )

        result = load_tester.run_load_test(
            make_request, num_requests=500, concurrency=30
        )
        print(f"\n{result}")

        assert result.successful / result.total_requests > 0.95, "Success rate < 95%"
        assert result.requests_per_second > 50, "Throughput < 50 req/s"
