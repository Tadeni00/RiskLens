"""
FraudTrap — Phase 2: Semi-Supervised Learning (TabPFN)

Uses Prior Labs TabPFN — a pretrained tabular foundation model — as the
core of the semi-supervised learning layer.  TabPFN performs in-context
prediction on small-to-medium labelled datasets without gradient training.

Reference:
  Hollmann et al., "Accurate predictions on small data with a tabular
  foundation model", Nature (2025).
"""

from models.semi_supervised.prediction import SemiSupervisedPrediction
from models.semi_supervised.tabpfn import TabPFNModel
from models.semi_supervised.trainer import SemiSupervisedTrainer

__all__ = [
    "SemiSupervisedPrediction",
    "TabPFNModel",
    "SemiSupervisedTrainer",
]
