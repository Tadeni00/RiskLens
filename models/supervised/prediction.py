"""
FraudTrap — Phase 3: Supervised Prediction Types
Strongly typed output for the confidence-aware CatBoost + FT-Transformer architecture.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class SupervisedPrediction:
    """
    Strongly typed output from the Phase 3 supervised layer.

    Attributes:
        probability: Final calibrated fraud probability in [0, 1].
        confidence: CatBoost confidence in [0, 1].
        ft_invoked: Whether FT-Transformer was consulted.
        fusion_output: Meta-fusion result (None if FT was not invoked).
        latency_ms: Total inference latency in milliseconds.
        model_version: Version string of the CatBoost model used.
        catboost_version: Version of CatBoost model.
        ft_transformer_version: Version of FT-Transformer if invoked.
    """

    probability: float
    confidence: float
    ft_invoked: bool = False
    fusion_output: Optional[float] = None
    latency_ms: float = 0.0
    model_version: str = ""
    catboost_version: str = ""
    ft_transformer_version: str = ""

    def __post_init__(self) -> None:
        for name in ("probability", "confidence"):
            val = getattr(self, name)
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {val}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probability": self.probability,
            "confidence": self.confidence,
            "ft_invoked": self.ft_invoked,
            "fusion_output": self.fusion_output,
            "latency_ms": self.latency_ms,
            "model_version": self.model_version,
            "catboost_version": self.catboost_version,
            "ft_transformer_version": self.ft_transformer_version,
        }
