"""
RiskLens — Load Testing with Locust
Run: locust -f tests/load/locustfile.py --host=http://localhost:8000
"""

import random
import uuid
from datetime import datetime, timezone
from locust import HttpUser, task, between, events
import json


class FraudTrapUser(HttpUser):
    """Simulates a client sending transactions for fraud scoring."""

    wait_time = between(0.05, 0.2)  # 5-20 req/sec per user

    def on_start(self):
        """Initialize user with tenant and account pools."""
        self.tenants = ["bank_ng_gtb", "bank_ke_equity", "fintech_za_yoco"]
        self.account_pool = [f"acct_{i}" for i in range(1, 10001)]
        self.device_pool = [f"dev_{i}" for i in range(1, 5001)]
        self.merchant_pool = [f"merch_{i}" for i in range(1, 2001)]

    def _random_transaction(self):
        """Generate a realistic transaction payload."""
        tenant = random.choice(self.tenants)
        return {
            "tenant_id": random.choice(self.tenants),
            "account_id": random.choice(self.account_pool),
            "amount": round(
                random.lognormvariate(9, 1.5), 2
            ),  # Realistic amount distribution
            "currency": "NGN",
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "transaction_type": random.choice(
                ["PAYMENT", "TRANSFER", "WITHDRAWAL", "TOP_UP", "REFUND"]
            ),
            "channel": random.choice(["MOBILE", "WEB", "API", "POS", "ATM", "USSD"]),
            "device_id": random.choice(self.device_pool),
            "merchant_id": (
                random.choice(self.merchant_pool) if random.random() > 0.2 else None
            ),
            "country_code": "NG" if random.random() > 0.1 else "US",
            "ip_address_hash": (
                f"ip_{random.randint(1, 10000)}" if random.random() > 0.3 else None
            ),
            "latitude": (
                round(random.uniform(6.0, 7.0), 4) if random.random() > 0.5 else None
            ),
            "longitude": (
                round(random.uniform(3.0, 4.0), 4) if random.random() > 0.5 else None
            ),
        }

    @task(10)
    def score_normal(self):
        """Normal transaction scoring."""
        payload = self._random_transaction()
        with self.client.post(
            "/v1/score", json=payload, catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "decision" in data and "risk_score" in data:
                        response.success()
                    else:
                        response.failure(f"Invalid response: {response.text}")
                except Exception:
                    response.failure(f"Invalid JSON: {response.text}")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")

    @task(3)
    def score_high_risk(self):
        """High-risk transaction (should trigger REVIEW/BLOCK)."""
        payload = {
            "tenant_id": "bank_ng_gtb",
            "account_id": "acct_high_risk",
            "amount": 500000.0,
            "currency": "NGN",
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "transaction_type": "TRANSFER",
            "channel": "API",
            "device_id": "new_device",
            "merchant_id": "new_merchant",
            "country_code": "US",
            "latitude": 40.7128,
            "longitude": -74.0060,
        }
        with self.client.post(
            "/v1/score", json=payload, catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data.get("decision") in ("REVIEW", "BLOCK"):
                        response.success()
                    else:
                        response.failure(
                            f"Expected REVIEW/BLOCK, got {data.get('decision')}"
                        )
                except Exception:
                    response.failure(f"Invalid JSON: {response.text}")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def batch_score(self):
        """Batch scoring endpoint."""
        batch = [self._random_transaction() for _ in range(50)]
        with self.client.post(
            "/v1/score/batch", json=batch, catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if isinstance(data, list) and len(data) == 50:
                        response.success()
                    else:
                        response.failure(
                            f"Invalid batch response: {len(data) if isinstance(data, list) else 'not list'}"
                        )
                except Exception:
                    response.failure(f"Invalid JSON: {response.text}")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def health_check(self):
        """Health endpoint."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def recent_scores(self):
        """Recent scores endpoint."""
        with self.client.get("/v1/recent?limit=100", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "items" in data:
                        response.success()
                    else:
                        response.failure("Missing items in response")
                except Exception:
                    response.failure("Invalid JSON")
            else:
                response.failure(f"HTTP {response.status_code}")


class HighVolumeUser(FraudTrapUser):
    """High-volume user for spike testing."""

    wait_time = between(0.01, 0.05)  # 20-100 req/sec per user
    weight = 3


class ReadOnlyUser(HttpUser):
    """Read-only user for dashboard/monitoring traffic."""

    wait_time = between(1, 5)
    weight = 1

    @task(5)
    def dashboard_overview(self):
        self.client.get("/health")

    @task(3)
    def recent(self):
        self.client.get("/v1/recent?limit=100")

    @task(1)
    def drift(self):
        self.client.get("/v1/drift/bank_ng_gtb")


# Event hooks for custom metrics
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("Load test started")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("Load test stopped")
    stats = environment.stats
    print(f"Total requests: {stats.total.num_requests}")
    print(f"Failures: {stats.total.num_failures}")
    print(f"Avg response time: {stats.total.avg_response_time:.2f}ms")
    print(f"P95 response time: {stats.total.get_response_time_percentile(0.95):.2f}ms")
    print(f"RPS: {stats.total.total_rps:.2f}")


# Custom shape for load testing
class LoadShape:
    """Defines user count over time for different scenarios."""

    @staticmethod
    def baseline():
        """Steady 100 TPS for 10 minutes."""
        return [
            (0, 50),  # Ramp up to 50 users
            (60, 100),  # 100 users for 8 min
            (540, 50),  # Ramp down
            (600, 0),  # Stop
        ]

    @staticmethod
    def peak():
        """Ramp to 500 TPS for 5 minutes."""
        return [
            (0, 50),
            (30, 150),
            (60, 250),
            (90, 400),
            (120, 500),  # Peak
            (420, 500),
            (450, 250),
            (480, 100),
            (500, 0),
        ]

    @staticmethod
    def spike():
        """Instant spike to 1000 TPS for 2 minutes."""
        return [
            (0, 100),
            (10, 500),
            (15, 1000),  # Spike!
            (135, 1000),
            (140, 500),
            (150, 0),
        ]

    @staticmethod
    def soak():
        """Long-running soak test at 200 TPS for 1 hour."""
        return [
            (0, 50),
            (60, 150),
            (120, 200),
            (3600, 200),  # 1 hour
            (3660, 100),
            (3720, 0),
        ]


if __name__ == "__main__":
    # Allow running directly for debugging
    import os

    os.environ.setdefault("LOCUST_HOST", "http://localhost:8000")

    # Quick test
    import requests

    try:
        resp = requests.get("http://localhost:8000/health", timeout=5)
        print(f"Health check: {resp.status_code}")
    except Exception as e:
        print(f"Health check failed: {e}")
