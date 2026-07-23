"""
FraudTrap — A/B Testing Framework
Champion/Challenger model routing with statistical significance testing.
"""

from __future__ import annotations
import hashlib
import time
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
from enum import Enum
from threading import Lock

from loguru import logger


class ExperimentStatus(str, Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"


class VariantType(str, Enum):
    CHAMPION = "champion"
    CHALLENGER = "challenger"


@dataclass
class ExperimentVariant:
    """A single variant in an experiment."""

    name: str
    variant_type: VariantType
    model_id: str  # e.g., "tenant:model_type:version"
    traffic_pct: float  # 0-100
    is_control: bool = False


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    name: str
    tenant_id: str
    description: str
    status: ExperimentStatus = ExperimentStatus.DRAFT

    variants: list = field(default_factory=list)

    # Traffic allocation
    total_traffic_pct: float = 100.0

    # Success metrics
    primary_metric: str = "pr_auc"  # pr_auc, recall, precision, fraud_capture_rate
    minimum_detectable_effect: float = 0.05  # 5% relative lift
    significance_level: float = 0.05  # 95% confidence
    statistical_power: float = 0.80

    # Sample size
    min_sample_per_variant: int = 1000
    max_duration_days: int = 30

    # Scheduling
    start_at: Optional[datetime] = None
    end_at: Optional[datetime] = None

    # Safety
    guardrail_metrics: list[str] = field(
        default_factory=lambda: ["error_rate", "latency_p95"]
    )
    guardrail_thresholds: dict[str, float] = field(
        default_factory=lambda: {"error_rate": 0.01, "latency_p95": 200}
    )

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class ExperimentResult:
    """Running experiment results."""

    experiment_name: str
    variant_name: str
    variant_type: VariantType

    # Sample counts
    samples: int = 0
    conversions: int = 0

    # Metrics
    metric_value: float = 0.0
    metric_std: float = 0.0

    # Statistical test
    p_value: float = 1.0
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    is_significant: bool = False
    lift_pct: float = 0.0

    # Guardrails
    guardrail_passed: bool = True
    guardrail_violations: list[str] = field(default_factory=list)

    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ExperimentRouter:
    """
    Routes traffic to experiment variants using consistent hashing.
    """

    def __init__(self, experiment: ExperimentConfig):
        self.experiment = experiment
        self._variant_weights = self._compute_weights()

    def _compute_weights(self) -> dict[str, float]:
        """Compute cumulative weights for routing."""
        weights = {}
        cumulative = 0.0
        for variant in self.experiment.variants:
            cumulative += variant.traffic_pct / 100.0
            weights[variant.name] = cumulative
        return weights

    def route(self, traffic_key: str) -> str:
        """
        Route traffic to a variant using consistent hashing.

        Args:
            traffic_key: Unique key for consistent assignment (e.g., transaction_id or tenant:user)

        Returns:
            Variant name
        """
        # Consistent hash
        hash_input = f"{self.experiment.name}:{traffic_key}".encode()
        hash_val = int(hashlib.sha256(hash_input).hexdigest(), 16)
        bucket = (hash_val % 10000) / 100.0  # 0-100

        for variant_name, weight in self._variant_weights.items():
            if bucket <= weight:
                return variant_name

        # Fallback to champion
        champion = next(
            v
            for v in self.experiment.variants
            if v.variant_type == VariantType.CHAMPION
        )
        return champion.name

    def get_variant_model_id(self, variant_name: str) -> Optional[str]:
        """Get model ID for a variant."""
        variant = next(
            (v for v in self.experiment.variants if v.name == variant_name), None
        )
        return variant.model_id if variant else None


class ExperimentManager:
    """
    Manages experiment lifecycle, routing, and results.
    """

    def __init__(self):
        self._experiments: dict[str, ExperimentConfig] = {}
        self._routers: dict[str, ExperimentRouter] = {}
        self._results: dict[str, dict[str, ExperimentResult]] = {}
        self._lock = Lock()

    def create_experiment(self, config: ExperimentConfig) -> ExperimentConfig:
        """Create a new experiment."""
        with self._lock:
            if config.name in self._experiments:
                raise ValueError(f"Experiment {config.name} already exists")

            # Validate
            self._validate_experiment(config)

            self._experiments[config.name] = config
            self._routers[config.name] = ExperimentRouter(config)
            self._results[config.name] = {}

            logger.info("Created experiment: {}", config.name)
            return config

    def _validate_experiment(self, config: ExperimentConfig) -> None:
        """Validate experiment configuration."""
        if not config.variants:
            raise ValueError("At least one variant required")

        # Must have exactly one champion
        champions = [
            v for v in config.variants if v.variant_type == VariantType.CHAMPION
        ]
        if len(champions) != 1:
            raise ValueError("Exactly one champion variant required")

        # Traffic must sum to 100%
        total_traffic = sum(v.traffic_pct for v in config.variants)
        if abs(total_traffic - 100.0) > 0.01:
            raise ValueError(
                f"Traffic percentages must sum to 100%, got {total_traffic}"
            )

        # Check variant names unique
        names = [v.name for v in config.variants]
        if len(set(names)) != len(names):
            raise ValueError("Variant names must be unique")

        # Validate traffic percentages
        for v in config.variants:
            if v.traffic_pct <= 0 or v.traffic_pct > 100:
                raise ValueError(f"Variant {v.name} traffic_pct must be 1-100")

    def get_experiment(self, name: str) -> Optional[ExperimentConfig]:
        with self._lock:
            return self._experiments.get(name)

    def start_experiment(self, name: str) -> bool:
        """Start an experiment."""
        with self._lock:
            exp = self._experiments.get(name)
            if not exp:
                return False
            if exp.status != ExperimentStatus.DRAFT:
                logger.warning(
                    "Cannot start experiment {} in status {}", name, exp.status
                )
                return False

            exp.status = ExperimentStatus.RUNNING
            exp.start_at = exp.start_at or datetime.now(timezone.utc)
            if not exp.end_at:
                exp.end_at = exp.start_at + timedelta(days=exp.max_duration_days)

            logger.info("Started experiment: {}", name)
            return True

    def pause_experiment(self, name: str) -> bool:
        with self._lock:
            exp = self._experiments.get(name)
            if not exp or exp.status != ExperimentStatus.RUNNING:
                return False
            exp.status = ExperimentStatus.PAUSED
            logger.info("Paused experiment: {}", name)
            return True

    def stop_experiment(self, name: str) -> bool:
        with self._lock:
            exp = self._experiments.get(name)
            if not exp:
                return False
            exp.status = ExperimentStatus.STOPPED
            exp.end_at = datetime.now(timezone.utc)
            logger.info("Stopped experiment: {}", name)
            return True

    def route(self, experiment_name: str, traffic_key: str) -> Optional[str]:
        """Route traffic to variant."""
        with self._lock:
            router = self._routers.get(name)
            if not router:
                return None
            return router.route(traffic_key)

    def get_variant_model(
        self, experiment_name: str, traffic_key: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Get model ID and variant name for a request."""
        with self._lock:
            router = self._routers.get(experiment_name)
            if not router:
                return None, None
            variant_name = router.route(traffic_key)
            model_id = router.get_variant_model_id(variant_name)
            return model_id, variant_name

    def record_outcome(
        self,
        experiment_name: str,
        variant_name: str,
        metric_value: float,
        converted: bool = False,
        guardrails: dict[str, float] = None,
    ) -> None:
        """Record an outcome for statistical analysis."""
        with self._lock:
            if experiment_name not in self._results:
                self._results[experiment_name] = {}

            if variant_name not in self._results[experiment_name]:
                self._results[experiment_name][variant_name] = ExperimentResult(
                    experiment_name=experiment_name,
                    variant_name=variant_name,
                    variant_type=VariantType.CHALLENGER,  # Will be updated
                )

            result = self._results[experiment_name][variant_name]
            result.samples += 1
            if converted:
                result.conversions += 1

            # Update running metric (simplified - use running average)
            if result.samples == 1:
                result.metric_value = metric_value
            else:
                # Running mean
                result.metric_value = (
                    (result.samples - 1) * result.metric_value + metric_value
                ) / result.samples

            # Check guardrails
            if guardrails:
                violations = []
                for metric, threshold in self._experiments[
                    self.experiment_name
                ].guardrail_thresholds.items():
                    if metric in guardrails and guardrails[metric] > threshold:
                        violations.append(
                            f"{metric}: {guardrails[metric]} > {threshold}"
                        )

                result.guardrail_violations = violations
                result.guardrail_passed = len(violations) == 0

            result.updated_at = datetime.now(timezone.utc)

    def compute_statistics(self, experiment_name: str) -> dict[str, ExperimentResult]:
        """Compute statistical significance for experiment."""
        from scipy import stats
        import numpy as np

        with self._lock:
            exp = self._experiments.get(experiment_name)
            if not exp:
                return {}

            results = self._results.get(experiment_name, {})

            # Find champion
            champion_name = next(
                v.name for v in exp.variants if v.variant_type == VariantType.CHAMPION
            )
            champion_result = results.get(champion_name)

            if not champion_result or champion_result.samples < 30:
                return results

            for variant_name, result in results.items():
                if variant_name == champion_name:
                    result.variant_type = VariantType.CHAMPION
                    continue

                result.variant_type = VariantType.CHALLENGER

                if result.samples < 30:
                    continue

                # Two-proportion z-test (for conversion rates)
                # Or t-test for continuous metrics
                try:
                    # Simplified: use conversion rate for binary, metric_value for continuous
                    if exp.primary_metric in ("conversion_rate", "fraud_capture_rate"):
                        # Two-proportion z-test
                        p1 = champion_result.conversions / max(
                            champion_result.samples, 1
                        )
                        p2 = result.conversions / max(result.samples, 1)
                        pooled_p = (
                            champion_result.conversions + result.conversions
                        ) / (champion_result.samples + result.samples)
                        se = np.sqrt(
                            pooled_p
                            * (1 - pooled_p)
                            * (1 / champion_result.samples + 1 / result.samples)
                        )
                        if se > 0:
                            z = (p2 - p1) / se
                            p_val = 2 * (1 - stats.norm.cdf(abs(z)))
                            result.p_value = p_val

                            # Lift
                            if p1 > 0:
                                result.lift_pct = (p2 - p1) / p1 * 100

                    # Significance
                    result.is_significant = result.p_value < exp.significance_level

                    # Confidence interval (simplified)
                    if result.samples > 0:
                        se = np.sqrt(
                            result.metric_value
                            * (1 - result.metric_value)
                            / result.samples
                        )
                        result.confidence_interval = (
                            result.metric_value - 1.96 * se,
                            result.metric_value + 1.96 * se,
                        )

                except Exception as exc:
                    logger.warning(
                        "Statistical test failed for {}: {}", experiment_name, exc
                    )

            return results

    def check_guardrails(self, experiment_name: str) -> tuple[bool, list[str]]:
        """Check if any guardrails are violated."""
        results = self._results.get(experiment_name, {})
        violations = []

        for variant_name, result in results.items():
            for violation in result.guardrail_violations:
                violations.append(f"{variant_name}: {violation}")

        return len(violations) == 0, violations

    def should_stop_early(self, experiment_name: str) -> tuple[bool, str]:
        """Check if experiment should stop early (significance or guardrail)."""
        exp = self._experiments.get(experiment_name)
        if not exp or exp.status != ExperimentStatus.RUNNING:
            return False, ""

        # Check guardrails
        guardrail_ok, violations = self.check_guardrails(experiment_name)
        if not guardrail_ok:
            return True, f"Guardrail violations: {', '.join(violations)}"

        # Check statistical significance
        stats = self.compute_statistics(experiment_name)
        champion = next(
            (r for r in stats.values() if r.variant_type == VariantType.CHAMPION), None
        )
        challengers = [
            r for r in stats.values() if r.variant_type == VariantType.CHALLENGER
        ]

        for challenger in challengers:
            if challenger.is_significant:
                lift = challenger.lift_pct
                if lift > 0:
                    return (
                        True,
                        f"Challenger {challenger.variant_name} significantly better (lift: {lift:.1f}%)",
                    )
                else:
                    return (
                        True,
                        f"Challenger {challenger.variant_name} significantly worse",
                    )

        # Check duration
        if exp.end_at and datetime.now(timezone.utc) >= exp.end_at:
            return True, "Max duration reached"

        return False, ""


# Global instance
_experiment_manager: Optional[ExperimentManager] = None


def get_experiment_manager() -> ExperimentManager:
    global _experiment_manager
    if _experiment_manager is None:
        _experiment_manager = ExperimentManager()
    return _experiment_manager


# Convenience functions
def create_experiment(
    name: str,
    tenant_id: str,
    champion_model: str,
    challenger_model: str,
    challenger_traffic_pct: float = 10.0,
    primary_metric: str = "pr_auc",
    max_duration_days: int = 30,
) -> ExperimentConfig:
    """Create a simple champion/challenger experiment."""
    config = ExperimentConfig(
        name=name,
        tenant_id=tenant_id,
        description=f"A/B test: {challenger_model} vs {champion_model}",
        variants=[
            ExperimentVariant(
                name="champion",
                variant_type=VariantType.CHAMPION,
                model_id=champion_model,
                traffic_pct=100.0 - challenger_traffic_pct,
                is_control=True,
            ),
            ExperimentVariant(
                name="challenger",
                variant_type=VariantType.CHALLENGER,
                model_id=challenger_model,
                traffic_pct=challenger_traffic_pct,
            ),
        ],
        primary_metric=primary_metric,
        max_duration_days=max_duration_days,
    )
    return get_experiment_manager().create_experiment(config)


def route_experiment(experiment_name: str, traffic_key: str) -> Optional[str]:
    """Route traffic to variant."""
    return get_experiment_manager().route(experiment_name, traffic_key)


def record_experiment_outcome(
    experiment_name: str,
    variant_name: str,
    metric_value: float,
    converted: bool = False,
    guardrails: dict[str, float] = None,
) -> None:
    """Record experiment outcome."""
    get_experiment_manager().record_outcome(
        experiment_name, variant_name, metric_value, converted, guardrails
    )
