"""
FraudTrap — Explainability Framework
Production-grade feature attribution and counterfactual explanations.
"""
from models.explainability.types import (
    FeatureAttribution,
    SHAPExplanation,
    CounterfactualChange,
    NearestNeighbor,
    CounterfactualExplanation,
    ConfidenceInfo,
    FormattedReport,
    FullExplanation,
)
from models.explainability.shap_explainer import SHAPExplainer
from models.explainability.nn_counterfactual import (
    NearestNeighborCounterfactual,
    NearestNeighborIndex,
    WeightedDistanceMetric,
)
from models.explainability.dice_counterfactual import DiCECounterfactual
from models.explainability.formatter import ExplanationFormatter
from models.explainability.cache import ExplanationCache, SHAPCache
from models.explainability.monitoring import ExplainabilityMonitor, ExplainabilityMetrics
from models.explainability.engine import ExplainabilityEngine, ExplainabilityConfig

__all__ = [
    "FeatureAttribution",
    "SHAPExplanation",
    "CounterfactualChange",
    "NearestNeighbor",
    "CounterfactualExplanation",
    "ConfidenceInfo",
    "FormattedReport",
    "FullExplanation",
    "SHAPExplainer",
    "NearestNeighborCounterfactual",
    "NearestNeighborIndex",
    "WeightedDistanceMetric",
    "DiCECounterfactual",
    "ExplanationFormatter",
    "ExplanationCache",
    "SHAPCache",
    "ExplainabilityMonitor",
    "ExplainabilityMetrics",
    "ExplainabilityEngine",
    "ExplainabilityConfig",
]
