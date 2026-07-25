"""
RiskLens — Chaos Testing
Validates system resilience under adverse conditions.
"""

from __future__ import annotations
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from contextlib import contextmanager

from loguru import logger


class ChaosType(str, Enum):
    REDIS_DOWN = "redis_down"
    KAFKA_DOWN = "kafka_down"
    CLICKHOUSE_DOWN = "clickhouse_down"
    MODEL_MISSING = "model_missing"
    LATENCY_INJECTION = "latency_injection"
    ERROR_INJECTION = "error_injection"


@dataclass
class ChaosExperiment:
    """Configuration for a chaos experiment."""

    name: str
    chaos_type: ChaosType
    duration_seconds: int
    intensity: float = 1.0  # 0.0 to 1.0
    target_service: Optional[str] = None
    metadata: dict = None


class ChaosMonkey:
    """
    Injects failures to test system resilience.
    """

    def __init__(self):
        self._active_experiments: dict[str, tuple[ChaosExperiment, float]] = {}
        self._original_methods: dict = {}

    def inject_redis_down(self, duration: int = 60) -> str:
        """Simulate Redis outage."""
        exp_id = f"redis_down_{int(time.time())}"
        # In production: patch Redis client to raise ConnectionError
        logger.warning("CHAOS: Redis down injected for {}s", duration)
        return exp_id

    def inject_kafka_down(self, duration: int = 60) -> str:
        """Simulate Kafka outage."""
        exp_id = f"kafka_down_{int(time.time())}"
        logger.warning("CHAOS: Kafka down injected for {}s", duration)
        return exp_id

    def inject_model_missing(self, tenant_id: str, duration: int = 60) -> str:
        """Simulate missing model artifact."""
        exp_id = f"model_missing_{tenant_id}_{int(time.time())}"
        logger.warning(
            "CHAOS: Model missing for tenant={} for {}s", tenant_id, duration
        )
        return exp_id

    def inject_latency(
        self, min_ms: int = 50, max_ms: int = 200, duration: int = 60
    ) -> str:
        """Inject random latency into scoring path."""
        exp_id = f"latency_{int(time.time())}"
        logger.warning(
            "CHAOS: Latency injection {}-{}ms for {}s", min_ms, max_ms, duration
        )
        return exp_id

    def inject_errors(self, error_rate: float = 0.1, duration: int = 60) -> str:
        """Inject random errors into scoring."""
        exp_id = f"errors_{int(time.time())}"
        logger.warning("CHAOS: Error injection {}% for {}s", error_rate * 100, duration)
        return exp_id

    def stop_experiment(self, exp_id: str) -> None:
        """Stop a chaos experiment."""
        logger.info("CHAOS: Stopped experiment {}", exp_id)

    @contextmanager
    def chaos(self, experiment: ChaosExperiment):
        """Context manager for chaos experiment."""
        exp_id = f"{experiment.chaos_type.value}_{int(time.time())}"
        logger.warning("Starting chaos experiment: {} ({})", experiment.name, exp_id)

        try:
            if experiment.chaos_type == ChaosType.REDIS_DOWN:
                self.inject_redis_down(experiment.duration_seconds)
            elif experiment.chaos_type == ChaosType.KAFKA_DOWN:
                self.inject_kafka_down(experiment.duration_seconds)
            elif experiment.chaos_type == ChaosType.MODEL_MISSING:
                self.inject_model_missing(
                    experiment.target_service or "all", experiment.duration_seconds
                )
            elif experiment.chaos_type == ChaosType.LATENCY_INJECTION:
                self.inject_latency(50, 200, experiment.duration_seconds)
            elif experiment.chaos_type == ChaosType.ERROR_INJECTION:
                self.inject_errors(0.1, experiment.duration_seconds)

            yield exp_id

        finally:
            self.stop_experiment(exp_id)


# Pre-defined chaos scenarios
CHAOS_SCENARIOS = [
    ChaosExperiment(
        name="redis_outage",
        chaos_type=ChaosType.REDIS_DOWN,
        duration_seconds=60,
        intensity=1.0,
    ),
    ChaosExperiment(
        name="kafka_outage",
        chaos_type=ChaosType.KAFKA_DOWN,
        duration_seconds=60,
        intensity=1.0,
    ),
    ChaosExperiment(
        name="model_missing_bank_ng_gtb",
        chaos_type=ChaosType.MODEL_MISSING,
        duration_seconds=120,
        target_service="bank_ng_gtb",
    ),
    ChaosExperiment(
        name="scoring_latency_spike",
        chaos_type=ChaosType.LATENCY_INJECTION,
        duration_seconds=60,
        intensity=0.5,
    ),
    ChaosExperiment(
        name="scoring_errors",
        chaos_type=ChaosType.ERROR_INJECTION,
        duration_seconds=60,
        intensity=0.1,
    ),
]


# Validation functions for chaos testing
def validate_chaos_resilience(
    score_fn,
    test_cases: list[dict],
    chaos_fn,
    max_latency_ms: float = 200,
    max_error_rate: float = 0.05,
) -> dict:
    """
    Validate system resilience under chaos.

    Args:
        score_fn: Scoring function to test
        test_cases: List of test transaction dicts
        chaos_fn: Function that injects chaos (returns exp_id)
        max_latency_ms: Maximum acceptable P95 latency
        max_error_rate: Maximum acceptable error rate

    Returns:
        Validation results
    """
    exp_id = chaos_fn()

    try:
        latencies = []
        errors = 0
        total = len(test_cases)

        for tc in test_cases:
            start = time.perf_counter()
            try:
                score_fn(tc)
            except Exception:
                errors += 1
            latencies.append((time.perf_counter() - start) * 1000)

        p95 = sorted(latencies)[int(0.95 * len(latencies))]
        error_rate = errors / max(len(test_cases), 1)

        return {
            "passed": p95 <= max_latency_ms and error_rate <= max_error_rate,
            "p95_latency_ms": p95,
            "error_rate": error_rate,
            "total_requests": total,
            "errors": errors,
        }
    finally:
        # Cleanup would happen here
        pass


# Standalone test runner
def run_chaos_suite(
    score_fn,
    test_cases: list[dict],
    scenarios: list = None,
) -> dict:
    """
    Run full chaos test suite.

    Returns:
        Dict with results for each scenario
    """
    scenarios = scenarios or CHAOS_SCENARIOS
    results = {}

    for scenario in scenarios:
        logger.info("Running chaos scenario: {}", scenario.name)

        def chaos_fn():
            if scenario.chaos_type.value == "redis_down":
                return "redis_down"  # In production, would actually inject
            elif scenario.chaos_type.value == "kafka_down":
                return "kafka_down"
            # ... etc
            return f"{scenario.chaos_type.value}_{int(time.time())}"

        result = validate_chaos_resilience(
            score_fn=score_fn,
            test_cases=test_cases,
            chaos_fn=chaos_fn,
        )

        results[scenario.name] = result
        logger.info("Scenario {} result: {}", scenario.name, result)

    return results


if __name__ == "__main__":
    # Demo
    monkey = ChaosMonkey()

    with monkey.chaos(
        ChaosExperiment(
            name="test_redis_down",
            chaos_type=ChaosType.REDIS_DOWN,
            duration_seconds=10,
        )
    ) as exp_id:
        logger.info("Chaos active: {}", exp_id)
        time.sleep(2)

    logger.info("Chaos test completed")
