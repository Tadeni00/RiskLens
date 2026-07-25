"""
RiskLens — Adaptive Learning Layer

The Adaptive Learning Layer (Layer 2) combines weak supervision, analyst feedback,
behavioral intelligence, confidence estimation, and pseudo-label generation to
continuously improve training data quality during the transition from unsupervised
detection to mature supervised learning.

TabPFN is the default scarce-label learner, abstracted behind the AdaptiveLearner
interface so it can be replaced with NetPFN or other learners via configuration.

Reference:
  Hollmann et al., "Accurate predictions on small data with a tabular
  foundation model", Nature (2025).
"""

from models.adaptive_learning.learner import AdaptiveLearner
from models.adaptive_learning.tabpfn_learner import TabPFNAdaptiveLearner
from models.adaptive_learning.prediction import AdaptivePrediction, PseudoLabelResult

__all__ = [
    "AdaptiveLearner",
    "TabPFNAdaptiveLearner",
    "AdaptivePrediction",
    "PseudoLabelResult",
]
