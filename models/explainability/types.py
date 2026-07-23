"""
FraudTrap — Explainability Types
Strongly typed dataclasses for the explainability framework.
Every output is immutable, validated, and tenant-scoped.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any


@dataclass(frozen=True)
class FeatureAttribution:
    """Single feature's contribution to the prediction."""
    feature: str
    value: float
    impact: float
    direction: str  # "increase" or "decrease"
    method: str  # "shap", "prototype_distance", "rule_weight"


@dataclass(frozen=True)
class SHAPExplanation:
    """SHAP-based feature attribution output."""
    fraud_probability: float
    base_value: float
    top_features: tuple[FeatureAttribution, ...]
    all_shap_values: Optional[tuple[float, ...]] = None
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fraud_probability": self.fraud_probability,
            "base_value": self.base_value,
            "top_features": [
                {"feature": f.feature, "value": f.value, "impact": f.impact, "direction": f.direction}
                for f in self.top_features
            ],
            "latency_ms": self.latency_ms,
        }


@dataclass(frozen=True)
class CounterfactualChange:
    """A single feature change in a counterfactual explanation."""
    feature: str
    current_value: float
    counterfactual_value: float
    realistic: bool


@dataclass(frozen=True)
class NearestNeighbor:
    """Reference to the nearest legitimate transaction."""
    transaction_id: str
    distance: float
    features: Dict[str, float]


@dataclass(frozen=True)
class CounterfactualExplanation:
    """Counterfactual explanation: what minimal changes would flip the decision."""
    prediction_delta: float
    changes: tuple[CounterfactualChange, ...]
    source: str  # "nearest_neighbor" or "dice"
    nearest_neighbor: Optional[NearestNeighbor] = None
    dice_distance: Optional[float] = None
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_delta": self.prediction_delta,
            "changes": [
                {"feature": c.feature, "current": c.current_value,
                 "counterfactual": c.counterfactual_value, "realistic": c.realistic}
                for c in self.changes
            ],
            "source": self.source,
            "latency_ms": self.latency_ms,
        }


@dataclass(frozen=True)
class ConfidenceInfo:
    """Model confidence metadata."""
    expert_used: str  # "CatBoost", "FTTransformer", "NetPFN", etc.
    confidence: float = 0.0
    ft_invoked: bool = False
    fusion_output: Optional[float] = None


@dataclass(frozen=True)
class FormattedReport:
    """Analyst-friendly explanation report."""
    fraud_probability: float
    confidence: ConfidenceInfo
    risk_drivers: tuple[str, ...]
    counterfactual_summary: Optional[str]
    nearest_legitimate: Optional[str]
    minimal_changes: tuple[str, ...]
    raw_shap: Optional[SHAPExplanation] = None
    raw_counterfactual: Optional[CounterfactualExplanation] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fraud_probability": self.fraud_probability,
            "confidence": {
                "expert_used": self.confidence.expert_used,
                "confidence": self.confidence.confidence,
                "ft_invoked": self.confidence.ft_invoked,
            },
            "risk_drivers": list(self.risk_drivers),
            "counterfactual_summary": self.counterfactual_summary,
            "nearest_legitimate": self.nearest_legitimate,
            "minimal_changes": list(self.minimal_changes),
        }


@dataclass
class FullExplanation:
    """Complete explanation output from the ExplainabilityEngine."""
    transaction_id: str
    tenant_id: str
    fraud_probability: float
    shap: Optional[SHAPExplanation] = None
    counterfactual: Optional[CounterfactualExplanation] = None
    formatted: Optional[FormattedReport] = None
    confidence: Optional[ConfidenceInfo] = None
    total_latency_ms: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "tenant_id": self.tenant_id,
            "fraud_probability": self.fraud_probability,
            "shap": self.shap.to_dict() if self.shap else None,
            "counterfactual": self.counterfactual.to_dict() if self.counterfactual else None,
            "formatted": self.formatted.to_dict() if self.formatted else None,
            "confidence": {
                "expert_used": self.confidence.expert_used,
                "confidence": self.confidence.confidence,
                "ft_invoked": self.confidence.ft_invoked,
            } if self.confidence else None,
            "total_latency_ms": self.total_latency_ms,
            "timestamp": self.timestamp,
        }
