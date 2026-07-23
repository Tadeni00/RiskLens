"""
FraudTrap — Phase 3: Supervised Monitoring
Tracks CatBoost confidence distribution, FT-Transformer invocation rate,
fusion model performance, latency impact, and calibration drift.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import numpy as np
from loguru import logger


@dataclass
class SupervisedMetrics:
    """Snapshot of Phase 3 monitoring metrics."""
    timestamp: str = ""
    catboost_confidence_mean: float = 0.0
    catboost_confidence_p50: float = 0.0
    catboost_confidence_p95: float = 0.0
    ft_invocation_rate: float = 0.0
    fusion_probability_mean: float = 0.0
    fusion_probability_std: float = 0.0
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    calibration_error: float = 0.0
    n_predictions: int = 0
    n_ft_invocations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "catboost_confidence_mean": self.catboost_confidence_mean,
            "catboost_confidence_p50": self.catboost_confidence_p50,
            "catboost_confidence_p95": self.catboost_confidence_p95,
            "ft_invocation_rate": self.ft_invocation_rate,
            "fusion_probability_mean": self.fusion_probability_mean,
            "fusion_probability_std": self.fusion_probability_std,
            "avg_latency_ms": self.avg_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "calibration_error": self.calibration_error,
            "n_predictions": self.n_predictions,
            "n_ft_invocations": self.n_ft_invocations,
        }


class SupervisedMonitor:
    """
    Monitoring for the Phase 3 confidence-aware supervised layer.
    
    Tracks:
    - CatBoost confidence distribution (should remain high)
    - Percentage of transactions routed to FT-Transformer
    - Fusion model performance
    - Latency impact of FT-Transformer consultation
    - Calibration drift
    - Feature drift (PSI)
    """

    def __init__(self, window_size: int = 10000):
        self.window_size = window_size
        self._confidences: List[float] = []
        self._ft_invoked: List[bool] = []
        self._fusion_probs: List[float] = []
        self._latencies: List[float] = []
        self._calibration_errors: List[float] = []
        self._predictions: List[float] = []

    def record_prediction(
        self,
        confidence: float,
        ft_invoked: bool,
        probability: float,
        latency_ms: float,
        fusion_output: Optional[float] = None,
    ) -> None:
        """Record a single prediction for monitoring."""
        self._confidences.append(confidence)
        self._ft_invoked.append(ft_invoked)
        self._predictions.append(probability)
        self._latencies.append(latency_ms)

        if fusion_output is not None:
            self._fusion_probs.append(fusion_output)

        # Keep window
        for attr in (
            "_confidences", "_ft_invoked", "_predictions",
            "_latencies", "_fusion_probs",
        ):
            lst = getattr(self, attr)
            if len(lst) > self.window_size:
                setattr(self, attr, lst[-self.window_size:])

    def record_calibration_error(self, ece: float) -> None:
        """Record calibration error measurement."""
        self._calibration_errors.append(ece)
        if len(self._calibration_errors) > self.window_size:
            self._calibration_errors = self._calibration_errors[-self.window_size:]

    def get_metrics(self) -> SupervisedMetrics:
        """Compute current monitoring metrics."""
        if not self._confidences:
            return SupervisedMetrics(
                timestamp=datetime.now(timezone.utc).isoformat()
            )

        confs = np.array(self._confidences)
        lats = np.array(self._latencies)
        preds = np.array(self._predictions)
        ft_flags = np.array(self._ft_invoked)

        fusion_mean = 0.0
        fusion_std = 0.0
        if self._fusion_probs:
            fp = np.array(self._fusion_probs)
            fusion_mean = float(fp.mean())
            fusion_std = float(fp.std())

        return SupervisedMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            catboost_confidence_mean=float(confs.mean()),
            catboost_confidence_p50=float(np.percentile(confs, 50)),
            catboost_confidence_p95=float(np.percentile(confs, 95)),
            ft_invocation_rate=float(ft_flags.mean()),
            fusion_probability_mean=fusion_mean,
            fusion_probability_std=fusion_std,
            avg_latency_ms=float(lats.mean()),
            p99_latency_ms=float(np.percentile(lats, 99)),
            calibration_error=(
                float(np.mean(self._calibration_errors))
                if self._calibration_errors
                else 0.0
            ),
            n_predictions=len(preds),
            n_ft_invocations=int(ft_flags.sum()),
        )

    def detect_issues(
        self,
        max_ft_rate: float = 0.20,
        max_confidence_drop: float = 0.10,
        max_latency_ms: float = 90.0,
    ) -> Dict[str, Any]:
        """
        Detect operational issues.
        
        Returns dict with issue status and details.
        """
        if len(self._confidences) < 100:
            return {"has_issues": False, "issues": [], "n_samples": len(self._confidences)}

        issues = []

        # Check FT-Transformer invocation rate
        ft_rate = np.mean(self._ft_invoked[-1000:])
        if ft_rate > max_ft_rate:
            issues.append(
                f"FT-Transformer invocation rate too high: {ft_rate:.1%} (max: {max_ft_rate:.1%})"
            )

        # Check confidence degradation
        if len(self._confidences) >= 1000:
            early = np.mean(self._confidences[:500])
            late = np.mean(self._confidences[-500:])
            if early - late > max_confidence_drop:
                issues.append(
                    f"CatBoost confidence dropping: {early:.3f} → {late:.3f}"
                )

        # Check latency
        recent_lats = np.array(self._latencies[-1000:])
        p99 = np.percentile(recent_lats, 99)
        if p99 > max_latency_ms:
            issues.append(
                f"P99 latency exceeded: {p99:.1f}ms (max: {max_latency_ms}ms)"
            )

        return {
            "has_issues": len(issues) > 0,
            "issues": issues,
            "n_samples": len(self._confidences),
        }

    def reset(self) -> None:
        """Clear all monitoring state."""
        self._confidences.clear()
        self._ft_invoked.clear()
        self._fusion_probs.clear()
        self._latencies.clear()
        self._calibration_errors.clear()
        self._predictions.clear()
