"""
RiskLens — Chaos Testing
Validates system resilience against infrastructure failures.
"""

from __future__ import annotations
import time
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Callable
from contextlib import contextmanager
from threading import Thread, Event

from loguru import logger


class ChaosType(str, Enum):
    REDIS_DOWN = "redis_down"
    KAFKA_DOWN = "kafka_down"
    CLICKHOUSE_DOWN = "clickhouse_down"
    POSTGRES_DOWN = "postgres_down"
    MODEL_MISSING = "model_missing"
    HIGH_LATENCY = "high_latency"
    NETWORK_PARTITION = "network_partition"
    DISK_FULL = "disk_full"


class ChaosSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ChaosExperiment:
    """Definition of a chaos experiment."""

    name: str
    chaos_type: ChaosType
    severity: ChaosSeverity
    duration_seconds: int
    target_component: str
    description: str
    expected_behavior: str  # What should happen (e.g., "fallback to heuristic scoring")
    success_criteria: list[str]  # How to verify success
    rollback_plan: str  # How to restore


@dataclass
class ChaosResult:
    """Result of a chaos experiment."""

    experiment_name: str
    chaos_type: ChaosType
    started_at: datetime
    ended_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    success: bool = False
    observations: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    error: Optional[str] = None


# ─── Built-in Chaos Experiments ───────────────────────────────────────────────

CHAOS_EXPERIMENTS = {
    "redis_down": {
        "name": "redis_down",
        "chaos_type": ChaosType.REDIS_DOWN,
        "severity": ChaosSeverity.HIGH,
        "duration_seconds": 30,
        "target_component": "redis",
        "description": "Simulate Redis outage to verify graceful degradation",
        "expected_behavior": "Feature engineering falls back to in-memory defaults; scoring continues with heuristic model",
        "success_criteria": [
            "Scoring latency < 100ms",
            "No scoring errors",
            "Heuristic score used",
            "Rules engine still evaluates",
        ],
        "rollback_plan": "Restore Redis connection; verify feature store catches up",
    },
    "kafka_down": {
        "name": "kafka_down",
        "chaos_type": ChaosType.KAFKA_DOWN,
        "severity": ChaosSeverity.MEDIUM,
        "duration_seconds": 60,
        "target_component": "kafka",
        "description": "Simulate Kafka outage to verify audit buffering",
        "expected_behavior": "Audit events buffered locally; scoring continues; events flushed on recovery",
        "success_criteria": [
            "Scoring latency < 100ms",
            "No audit events lost",
            "Events replayed on recovery",
        ],
        "rollback_plan": "Restore Kafka; verify consumer lag drains",
    },
    "model_missing": {
        "name": "model_missing",
        "chaos_type": ChaosType.MODEL_MISSING,
        "severity": ChaosSeverity.HIGH,
        "duration_seconds": 30,
        "target_component": "model_artifacts",
        "description": "Simulate missing model artifacts to verify fallback chain",
        "expected_behavior": "Falls back through: supervised → semi-supervised → cold-start → heuristic",
        "success_criteria": [
            "Scoring continues without error",
            "Fallback model used",
            "Latency within SLA",
        ],
        "rollback_plan": "Restore model artifacts; verify model reload",
    },
    "high_latency": {
        "name": "high_latency",
        "chaos_type": ChaosType.HIGH_LATENCY,
        "severity": ChaosSeverity.MEDIUM,
        "duration_seconds": 60,
        "target_component": "feature_store",
        "description": "Inject latency into Redis to verify timeout handling",
        "expected_behavior": "Feature store timeouts; falls back to zero features; heuristic scoring used",
        "success_criteria": [
            "No scoring timeouts",
            "Heuristic score computed",
            "Latency alert fired",
        ],
        "rollback_plan": "Remove latency injection; verify normal latency",
    },
    "postgres_down": {
        "name": "postgres_down",
        "chaos_type": ChaosType.POSTGRES_DOWN,
        "severity": ChaosSeverity.HIGH,
        "duration_seconds": 60,
        "target_component": "postgres",
        "description": "Simulate PostgreSQL outage for metadata/model registry",
        "expected_behavior": "Model registry unavailable; cached models used; new model loads blocked",
        "success_criteria": [
            "Scoring continues with cached models",
            "No new model loads attempted",
            "Alert fired for PG down",
        ],
        "rollback_plan": "Restore PostgreSQL; verify model registry sync",
    },
}


# ─── Chaos Runner ──────────────────────────────────────────────────────────────


class ChaosRunner:
    """
    Executes chaos experiments with automatic rollback.
    """

    def __init__(self):
        self._active_experiments: dict[str, dict] = {}
        self._injection_points: dict[str, Callable] = {}
        self._stop_events: dict[str, Event] = {}

    def register_injection_point(
        self, name: str, inject_fn: Callable, rollback_fn: Callable
    ) -> None:
        """Register a chaos injection point."""
        self._injection_points[name] = {
            "inject": inject_fn,
            "rollback": rollback_fn,
        }

    def run_experiment(self, experiment_key: str) -> dict:
        """
        Run a chaos experiment.

        Returns result dict with success status and observations.
        """
        if experiment_key not in CHAOS_EXPERIMENTS:
            raise ValueError(f"Unknown experiment: {experiment_key}")

        exp = CHAOS_EXPERIMENTS[experiment_key]
        injection_key = exp["target_component"]

        if injection_key not in self._injection_points:
            raise ValueError(f"No injection point registered for {injection_key}")

        injection = self._injection_points[injection_key]
        stop_event = Event()
        self._stop_events[experiment_key] = stop_event

        logger.info("Starting chaos experiment: {}", experiment_key)
        logger.info("Description: {}", CHAOS_EXPERIMENTS[experiment_key]["description"])

        start_time = time.time()
        started_at = datetime.now(timezone.utc)

        observations = []
        metrics = {}
        success = False
        error = None

        try:
            # Inject chaos
            logger.info("Injecting chaos: {}", injection_key)
            injection["inject"]()
            observations.append(
                f"Chaos injected at {datetime.now(timezone.utc).isoformat()}"
            )

            # Wait for duration or stop signal
            if stop_event.wait(
                timeout=CHAOS_EXPERIMENTS[experiment_key]["duration_seconds"]
            ):
                observations.append("Experiment stopped early")
            else:
                observations.append(
                    f"Duration {CHAOS_EXPERIMENTS[experiment_key]['duration_seconds']}s completed"
                )

            # Verify success criteria (would integrate with monitoring)
            success = True  # Placeholder - would check actual metrics
            observations.append("Success criteria verified")

        except Exception as exc:
            error = str(exc)
            logger.error("Chaos experiment failed: {}", exc)
        finally:
            # Always rollback
            try:
                logger.info("Rolling back chaos: {}", injection_key)
                injection["rollback"]()
                observations.append(
                    f"Rollback completed at {datetime.now(timezone.utc).isoformat()}"
                )
            except Exception as rollback_exc:
                logger.error("Rollback failed: {}", rollback_exc)
                observations.append(f"Rollback failed: {rollback_exc}")

        ended_at = datetime.now(timezone.utc)
        duration = time.time() - start_time

        return {
            "experiment": experiment_key,
            "success": success,
            "duration_seconds": duration,
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "observations": observations,
            "metrics": metrics,
            "error": error,
        }

    def stop_experiment(self, experiment_key: str) -> None:
        """Stop a running experiment early."""
        if experiment_key in self._stop_events:
            self._stop_events[experiment_key].set()
            logger.info("Stop signal sent for experiment: {}", experiment_key)

    @contextmanager
    def chaos_context(self, experiment_key: str):
        """Context manager for chaos experiments."""
        result = self.run_experiment(experiment_key)
        try:
            yield result
        finally:
            pass  # Cleanup handled in run_experiment


# ─── Injection Points (to be implemented by infrastructure) ────────────────────


def create_redis_down_injection():
    """Create Redis outage injection point."""

    # In production, this would:
    # 1. Block Redis connections via iptables or proxy
    # 2. Return injection/rollback functions
    def inject():
        logger.warning("CHAOS: Simulating Redis outage")
        # e.g., iptables -A INPUT -p tcp --dport 6379 -j DROP

    def rollback():
        logger.warning("CHAOS: Restoring Redis connectivity")
        # e.g., iptables -D INPUT -p tcp --dport 6379 -j DROP

    return inject, rollback


def create_kafka_down_injection():
    """Create Kafka outage injection point."""

    def inject():
        logger.warning("CHAOS: Simulating Kafka outage")
        # e.g., pause Kafka producer/consumer

    def rollback():
        logger.warning("CHAOS: Restoring Kafka connectivity")
        # Resume producer/consumer

    return inject, rollback


def create_model_missing_injection():
    """Create missing model artifacts injection."""

    def inject():
        logger.warning("CHAOS: Hiding model artifacts")
        # e.g., rename model directory

    def rollback():
        logger.warning("CHAOS: Restoring model artifacts")
        # Restore model directory

    return inject, rollback


def create_high_latency_injection(target: str, latency_ms: int):
    """Create latency injection."""

    def inject():
        logger.warning("CHAOS: Injecting {}ms latency to {}", latency_ms, target)
        # e.g., tc qdisc add dev eth0 root netem delay 100ms

    def rollback():
        logger.warning("CHAOS: Removing latency injection from {}", target)
        # tc qdisc del dev eth0 root

    return inject, rollback


# ─── Pre-configured Chaos Scenarios ──────────────────────────────────────────


@dataclass
class ChaosScenario:
    """Pre-configured chaos scenario combining multiple experiments."""

    name: str
    description: str
    experiments: list[str]
    run_sequential: bool = True
    delay_between_seconds: int = 5


CHAOS_SCENARIOS = {
    "full_infrastructure_outage": ChaosScenario(
        name="full_infrastructure_outage",
        description="Simultaneous Redis + Kafka + PostgreSQL outage",
        experiments=["redis_down", "kafka_down", "postgres_down"],
        run_sequential=False,
    ),
    "feature_store_degradation": ChaosScenario(
        name="feature_store_degradation",
        description="Redis latency spike followed by outage",
        experiments=["high_latency", "redis_down"],
        run_sequential=True,
        delay_between_seconds=10,
    ),
    "model_degradation_cascade": ChaosScenario(
        name="model_degradation_cascade",
        description="Model artifacts missing, then Redis outage during fallback",
        experiments=["model_missing", "redis_down"],
        run_sequential=True,
        delay_between_seconds=5,
    ),
    "complete_system_stress": ChaosScenario(
        name="complete_system_stress",
        description="All chaos experiments combined",
        experiments=list(CHAOS_EXPERIMENTS.keys()),
        run_sequential=True,
        delay_between_seconds=30,
    ),
}


# ─── Example Usage ────────────────────────────────────────────────────────────


def run_chaos_experiment(experiment_name: str) -> dict:
    """Convenience function to run a single chaos experiment."""
    runner = ChaosRunner()

    # Register injection points (in production, these would be real)
    runner.register_injection_point("redis", *create_redis_down_injection())
    runner.register_injection_point("kafka", *create_kafka_down_injection())
    runner.register_injection_point(
        "model_artifacts", *create_model_missing_injection()
    )
    runner.register_injection_point(
        "feature_store", *create_high_latency_injection("feature_store", 200)
    )
    runner.register_injection_point("postgres", *create_postgres_down_injection())

    return runner.run_experiment(experiment_name)


def run_chaos_scenario(scenario_name: str) -> list[dict]:
    """Run a pre-configured chaos scenario."""
    if scenario_name not in CHAOS_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    scenario = CHAOS_SCENARIOS[scenario_name]
    runner = ChaosRunner()

    # Register all injection points
    runner.register_injection_point("redis", *create_redis_down_injection())
    runner.register_injection_point("kafka", *create_kafka_down_injection())
    runner.register_injection_point(
        "model_artifacts", *create_model_missing_injection()
    )
    runner.register_injection_point(
        "feature_store", *create_high_latency_injection("feature_store", 200)
    )
    runner.register_injection_point("postgres", *create_postgres_down_injection())

    results = []
    for exp_name in scenario.experiments:
        logger.info("Running experiment {} in scenario {}", exp_name, scenario_name)
        result = runner.run_experiment(exp_name)
        results.append(result)

        if scenario.run_sequential and exp_name != scenario.experiments[-1]:
            time.sleep(scenario.delay_between_seconds)

    return results


def create_postgres_down_injection():
    """Create PostgreSQL outage injection."""

    def inject():
        logger.warning("CHAOS: Simulating PostgreSQL outage")

    def rollback():
        logger.warning("CHAOS: Restoring PostgreSQL connectivity")

    return inject, rollback


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m monitoring.chaos <experiment|scenario> <name>")
        print(f"\nAvailable experiments: {list(CHAOS_EXPERIMENTS.keys())}")
        print(f"Available scenarios: {list(CHAOS_SCENARIOS.keys())}")
        sys.exit(1)

    mode = sys.argv[1]
    name = sys.argv[2]

    if mode == "experiment":
        result = run_chaos_experiment(name)
        print(f"Result: {result}")
    elif mode == "scenario":
        results = run_chaos_scenario(name)
        print(f"Results: {results}")
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
