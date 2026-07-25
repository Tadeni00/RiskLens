"""
RiskLens — Adaptive Learning Monitoring

Tracks uncertainty distribution, pseudo-label acceptance rate,
calibration drift, and prediction confidence for the Adaptive Learning Layer.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import numpy as np
from loguru import logger


@dataclass
class AdaptiveMetrics:
    """Snapshot of Adaptive Learning Layer monitoring metrics."""

    timestamp: str = ""
    uncertainty_mean: float = 0.0
    uncertainty_p50: float = 0.0
    uncertainty_p95: float = 0.0
    confidence_mean: float = 0.0
    confidence_p50: float = 0.0
    pseudo_label_acceptance_rate: float = 0.0
    calibration_error: float = 0.0
    prediction_mean: float = 0.0
    prediction_std: float = 0.0
    n_predictions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "uncertainty_mean": self.uncertainty_mean,
            "uncertainty_p50": self.uncertainty_p50,
            "uncertainty_p95": self.uncertainty_p95,
            "confidence_mean": self.confidence_mean,
            "confidence_p50": self.confidence_p50,
            "pseudo_label_acceptance_rate": self.pseudo_label_acceptance_rate,
            "calibration_error": self.calibration_error,
            "prediction_mean": self.prediction_mean,
            "prediction_std": self.prediction_std,
            "n_predictions": self.n_predictions,
        }


class AdaptiveMonitor:
    """
    Monitoring for the Adaptive Learning Layer.

    Tracks:
    - Uncertainty distribution (should remain stable or decrease over time)
    - Pseudo-label acceptance rate (should stabilise as model improves)
    - Calibration drift (ECE should remain low)
    - Prediction confidence distribution
    """

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._predictions: List[float] = []
        self._uncertainties: List[float] = []
        self._confidences: List[float] = []
        self._pseudo_acceptance_history: List[float] = []
        self._calibration_errors: List[float] = []

    def record_prediction(
        self,
        probability: float,
        confidence: float,
        uncertainty: float,
    ) -> None:
        """Record a single prediction for monitoring."""
        self._predictions.append(probability)
        self._confidences.append(confidence)
        self._uncertainties.append(uncertainty)

        if len(self._predictions) > self.window_size:
            self._predictions = self._predictions[-self.window_size :]
            self._confidences = self._confidences[-self.window_size :]
            self._uncertainties = self._uncertainties[-self.window_size :]

    def record_pseudo_label_batch(
        self,
        n_accepted: int,
        n_total: int,
    ) -> None:
        """Record pseudo-label acceptance for a batch."""
        rate = n_accepted / max(n_total, 1)
        self._pseudo_acceptance_history.append(rate)
        if len(self._pseudo_acceptance_history) > self.window_size:
            self._pseudo_acceptance_history = self._pseudo_acceptance_history[-self.window_size :]

    def record_calibration_error(self, ece: float) -> None:
        """Record calibration error measurement."""
        self._calibration_errors.append(ece)
        if len(self._calibration_errors) > self.window_size:
            self._calibration_errors = self._calibration_errors[-self.window_size :]

    def get_metrics(self) -> AdaptiveMetrics:
        """Compute current monitoring metrics."""
        if not self._predictions:
            return AdaptiveMetrics(timestamp=datetime.now(timezone.utc).isoformat())

        preds = np.array(self._predictions)
        uncs = np.array(self._uncertainties)
        confs = np.array(self._confidences)

        return AdaptiveMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            uncertainty_mean=float(uncs.mean()),
            uncertainty_p50=float(np.percentile(uncs, 50)),
            uncertainty_p95=float(np.percentile(uncs, 95)),
            confidence_mean=float(confs.mean()),
            confidence_p50=float(np.percentile(confs, 50)),
            pseudo_label_acceptance_rate=(
                float(np.mean(self._pseudo_acceptance_history))
                if self._pseudo_acceptance_history
                else 0.0
            ),
            calibration_error=(
                float(np.mean(self._calibration_errors)) if self._calibration_errors else 0.0
            ),
            prediction_mean=float(preds.mean()),
            prediction_std=float(preds.std()),
            n_predictions=len(preds),
        )

    def detect_drift(
        self,
        uncertainty_threshold: float = 0.1,
        confidence_threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Detect if monitoring metrics indicate drift.

        Returns:
            Dict with drift status and reasons.
        """
        if len(self._uncertainties) < 100:
            return {
                "drifted": False,
                "reasons": [],
                "n_samples": len(self._uncertainties),
            }

        uncs = np.array(self._uncertainties[-200:])
        confs = np.array(self._confidences[-200:])

        reasons = []

        if len(uncs) >= 100:
            first_half = uncs[: len(uncs) // 2].mean()
            second_half = uncs[len(uncs) // 2 :].mean()
            if second_half - first_half > uncertainty_threshold:
                reasons.append(f"Uncertainty increasing: {first_half:.3f} -> {second_half:.3f}")

        if len(confs) >= 100:
            first_half = confs[: len(confs) // 2].mean()
            second_half = confs[len(confs) // 2 :].mean()
            if first_half - second_half > confidence_threshold:
                reasons.append(f"Confidence decreasing: {first_half:.3f} -> {second_half:.3f}")

        return {
            "drifted": len(reasons) > 0,
            "reasons": reasons,
            "n_samples": len(uncs),
        }

    def reset(self) -> None:
        """Clear all monitoring state."""
        self._predictions.clear()
        self._uncertainties.clear()
        self._confidences.clear()
        self._pseudo_acceptance_history.clear()
        self._calibration_errors.clear()
