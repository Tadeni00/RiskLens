"""
FraudTrap — Explainability Monitoring
Tracks latency, usage, explanation quality, and operational health.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from collections import defaultdict
import numpy as np
from loguru import logger


@dataclass
class ExplainabilityMetrics:
    """Snapshot of explainability monitoring metrics."""

    timestamp: str = ""
    total_explanations: int = 0
    shap_latency_p50_ms: float = 0.0
    shap_latency_p95_ms: float = 0.0
    shap_latency_p99_ms: float = 0.0
    counterfactual_latency_p50_ms: float = 0.0
    counterfactual_latency_p95_ms: float = 0.0
    total_latency_p50_ms: float = 0.0
    total_latency_p95_ms: float = 0.0
    cache_hit_rate: float = 0.0
    counterfactual_success_rate: float = 0.0
    top_fraud_drivers: list = field(default_factory=list)
    most_common_counterfactual_features: list = field(default_factory=list)
    explanation_errors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_explanations": self.total_explanations,
            "shap_latency_p50_ms": self.shap_latency_p50_ms,
            "shap_latency_p95_ms": self.shap_latency_p95_ms,
            "total_latency_p50_ms": self.total_latency_p50_ms,
            "total_latency_p95_ms": self.total_latency_p95_ms,
            "cache_hit_rate": self.cache_hit_rate,
            "counterfactual_success_rate": self.counterfactual_success_rate,
            "top_fraud_drivers": self.top_fraud_drivers,
            "explanation_errors": self.explanation_errors,
        }


class ExplainabilityMonitor:
    """
    Real-time monitoring for the explainability layer.

    Tracks:
    - Explanation latency (SHAP, counterfactual, total)
    - Cache hit rate
    - Counterfactual success rate
    - Most common fraud drivers
    - Most common counterfactual features
    - Explanation errors
    """

    def __init__(self, window_size: int = 10_000):
        self.window_size = window_size

        self._shap_latencies: List[float] = []
        self._cf_latencies: List[float] = []
        self._total_latencies: List[float] = []
        self._cache_hits: List[bool] = []
        self._cf_successes: List[bool] = []
        self._fraud_driver_counts: Dict[str, int] = defaultdict(int)
        self._cf_feature_counts: Dict[str, int] = defaultdict(int)
        self._errors: int = 0
        self._total: int = 0

    def record_explanation(
        self,
        shap_latency_ms: float,
        total_latency_ms: float,
        counterfactual_latency_ms: Optional[float] = None,
        cache_hit: bool = False,
        counterfactual_success: bool = False,
        fraud_drivers: Optional[List[str]] = None,
        counterfactual_features: Optional[List[str]] = None,
    ) -> None:
        """Record a single explanation event."""
        self._total += 1
        self._shap_latencies.append(shap_latency_ms)
        self._total_latencies.append(total_latency_ms)
        self._cache_hits.append(cache_hit)
        self._cf_successes.append(counterfactual_success)

        if counterfactual_latency_ms is not None:
            self._cf_latencies.append(counterfactual_latency_ms)

        if fraud_drivers:
            for driver in fraud_drivers:
                # Extract feature name from driver string
                feature = driver.split(" is ")[0] if " is " in driver else driver
                self._fraud_driver_counts[feature] += 1

        if counterfactual_features:
            for feat in counterfactual_features:
                self._cf_feature_counts[feat] += 1

        # Keep window
        for attr in (
            "_shap_latencies",
            "_cf_latencies",
            "_total_latencies",
            "_cache_hits",
            "_cf_successes",
        ):
            lst = getattr(self, attr)
            if len(lst) > self.window_size:
                setattr(self, attr, lst[-self.window_size :])

    def record_error(self) -> None:
        """Record an explanation error."""
        self._errors += 1

    def get_metrics(self) -> ExplainabilityMetrics:
        """Compute current monitoring metrics."""
        if not self._total_latencies:
            return ExplainabilityMetrics(
                timestamp=datetime.now(timezone.utc).isoformat()
            )

        total_lats = np.array(self._total_latencies)
        shap_lats = (
            np.array(self._shap_latencies) if self._shap_latencies else np.array([0.0])
        )
        cf_lats = (
            np.array(self._cf_latencies) if self._cf_latencies else np.array([0.0])
        )

        cache_hit_rate = np.mean(self._cache_hits) if self._cache_hits else 0.0
        cf_success_rate = np.mean(self._cf_successes) if self._cf_successes else 0.0

        # Top fraud drivers
        sorted_drivers = sorted(
            self._fraud_driver_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        # Most common counterfactual features
        sorted_cf = sorted(
            self._cf_feature_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        return ExplainabilityMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_explanations=self._total,
            shap_latency_p50_ms=float(np.percentile(shap_lats, 50)),
            shap_latency_p95_ms=float(np.percentile(shap_lats, 95)),
            shap_latency_p99_ms=float(np.percentile(shap_lats, 99)),
            counterfactual_latency_p50_ms=float(np.percentile(cf_lats, 50)),
            counterfactual_latency_p95_ms=float(np.percentile(cf_lats, 95)),
            total_latency_p50_ms=float(np.percentile(total_lats, 50)),
            total_latency_p95_ms=float(np.percentile(total_lats, 95)),
            cache_hit_rate=float(cache_hit_rate),
            counterfactual_success_rate=float(cf_success_rate),
            top_fraud_drivers=[{"feature": f, "count": c} for f, c in sorted_drivers],
            most_common_counterfactual_features=[
                {"feature": f, "count": c} for f, c in sorted_cf
            ],
            explanation_errors=self._errors,
        )

    def detect_issues(
        self,
        max_total_latency_ms: float = 40.0,
        max_shap_latency_ms: float = 20.0,
        min_cache_hit_rate: float = 0.3,
    ) -> Dict[str, Any]:
        """Detect operational issues with the explainability layer."""
        if self._total < 100:
            return {"has_issues": False, "issues": [], "n_samples": self._total}

        issues = []

        total_lats = np.array(self._total_latencies[-1000:])
        shap_lats = np.array(self._shap_latencies[-1000:])

        p95_total = float(np.percentile(total_lats, 95))
        if p95_total > max_total_latency_ms:
            issues.append(f"Total explanation P95 latency too high: {p95_total:.1f}ms")

        p95_shap = float(np.percentile(shap_lats, 95))
        if p95_shap > max_shap_latency_ms:
            issues.append(f"SHAP P95 latency too high: {p95_shap:.1f}ms")

        if len(self._cache_hits) >= 100:
            recent_hit_rate = np.mean(self._cache_hits[-1000:])
            if recent_hit_rate < min_cache_hit_rate:
                issues.append(f"Cache hit rate too low: {recent_hit_rate:.1%}")

        return {
            "has_issues": len(issues) > 0,
            "issues": issues,
            "n_samples": self._total,
        }

    def reset(self) -> None:
        """Clear all monitoring state."""
        self._shap_latencies.clear()
        self._cf_latencies.clear()
        self._total_latencies.clear()
        self._cache_hits.clear()
        self._cf_successes.clear()
        self._fraud_driver_counts.clear()
        self._cf_feature_counts.clear()
        self._errors = 0
        self._total = 0
