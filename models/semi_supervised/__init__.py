"""
FraudTrap — Phase 2: Semi-Supervised Learning (NetPFN)
Replaces the XGBoost-based SemiSupervisedBridge with a NetPFN architecture
designed for few-label, weak-label, and pseudo-label scenarios.
"""
from models.semi_supervised.prediction import SemiSupervisedPrediction
from models.semi_supervised.netpfn import NetPFNModel
from models.semi_supervised.trainer import SemiSupervisedTrainer

__all__ = [
    "SemiSupervisedPrediction",
    "NetPFNModel",
    "SemiSupervisedTrainer",
]
