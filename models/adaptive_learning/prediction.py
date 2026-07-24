"""
FraudTrap — Adaptive Learning Prediction Types

Strongly typed output for Adaptive Learning Layer predictions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass(frozen=True)
class AdaptivePrediction:
    """
    Strongly typed output from the Adaptive Learning Layer.

    Attributes:
        probability: Calibrated fraud probability in [0, 1].
        confidence: Model confidence in the prediction in [0, 1].
        uncertainty: Estimated prediction uncertainty in [0, 1].
        ft_invoked: Always False for Layer 2 (no specialist consultation).
        fusion_output: None for Layer 2 (no meta-fusion).
        latency_ms: Inference latency in milliseconds.
        model_version: Version string of the model used.
    """

    probability: float
    confidence: float
    uncertainty: float
    ft_invoked: bool = False
    fusion_output: Optional[float] = None
    latency_ms: float = 0.0
    model_version: str = ""

    def __post_init__(self) -> None:
        for name in ("probability", "confidence", "uncertainty"):
            val = getattr(self, name)
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {val}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probability": self.probability,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "ft_invoked": self.ft_invoked,
            "fusion_output": self.fusion_output,
            "latency_ms": self.latency_ms,
            "model_version": self.model_version,
        }


@dataclass(frozen=True)
class PseudoLabelResult:
    """
    Result of pseudo-label generation.

    Attributes:
        X_pseudo: Feature matrix of pseudo-labelled samples.
        y_pseudo: Pseudo labels (0 or 1).
        review_ids: Transaction IDs sent to human review queue.
        high_conf_count: Number of high-confidence pseudo-labels.
        low_conf_count: Number of low-confidence pseudo-labels.
        pseudo_high_threshold: Threshold used for high-confidence.
        pseudo_low_threshold: Threshold used for low-confidence.
    """

    X_pseudo: Any  # np.ndarray
    y_pseudo: Any  # np.ndarray
    review_ids: List[Any] = field(default_factory=list)
    high_conf_count: int = 0
    low_conf_count: int = 0
    pseudo_high_threshold: float = 0.0
    pseudo_low_threshold: float = 0.0
