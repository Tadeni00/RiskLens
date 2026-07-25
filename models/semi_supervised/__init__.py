"""
RiskLens — Semi-Supervised Learning (Legacy Re-exports)

This package is maintained for backwards compatibility only.
All new code should import from ``models.adaptive_learning`` instead.
"""

from models.adaptive_learning.prediction import AdaptivePrediction as SemiSupervisedPrediction
from models.adaptive_learning.prediction import PseudoLabelResult
from models.adaptive_learning.tabpfn_learner import TabPFNAdaptiveLearner as TabPFNModel
from models.adaptive_learning.trainer import AdaptiveTrainer as SemiSupervisedTrainer
from models.adaptive_learning.trainer import AdaptiveConfig as SemiSupervisedConfig
from models.adaptive_learning.trainer import AdaptiveTrainingResult as SemiSupervisedTrainingResult
from models.adaptive_learning.monitoring import AdaptiveMonitor as SemiSupervisedMonitor
from models.adaptive_learning.monitoring import AdaptiveMetrics as SemiSupervisedMetrics

__all__ = [
    "SemiSupervisedPrediction",
    "TabPFNModel",
    "SemiSupervisedTrainer",
    "SemiSupervisedConfig",
    "SemiSupervisedTrainingResult",
    "SemiSupervisedMonitor",
    "SemiSupervisedMetrics",
    "PseudoLabelResult",
]
