"""
FraudTrap — AdaptiveLearner Interface

Abstract base class for the Adaptive Learning Layer (Layer 2).
All scarce-label learners must conform to this interface so the surrounding
pipeline remains learner-agnostic.

Supported implementations:
  - TabPFNAdaptiveLearner (default, commercial license)
  - Future: NetPFNAdaptiveLearner (fully permissive alternative)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class AdaptiveLearner(ABC):
    """
    Abstract interface for the Adaptive Learning Layer's scarce-label learner.

    Responsibilities:
      - Learning from very small labelled datasets
      - Producing calibrated fraud probabilities
      - Generating high-confidence pseudo-labels
      - Estimating prediction confidence
      - Prioritising uncertain samples for analyst review (active learning)
      - Accelerating transition toward supervised learning

    This learner is NOT a production fraud classifier.
    Its objective is to continuously improve the quality of labelled data
    before mature supervised learning takes over.
    """

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> "AdaptiveLearner":
        """
        Fit the learner on the labelled training data.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Labels (n_samples,), binary 0/1.
            sample_weights: Optional per-sample weights.

        Returns:
            self
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return hard class predictions (0 or 1).

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Array of predictions (n_samples,).
        """
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return calibrated fraud probabilities.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            1-D array of fraud probabilities in [0, 1].
        """
        ...

    @abstractmethod
    def generate_pseudo_labels(
        self,
        X_unlabelled: np.ndarray,
        high_threshold: float = 0.95,
        low_threshold: float = 0.10,
    ) -> Dict[str, Any]:
        """
        Score unlabelled data and assign pseudo-labels.

        Args:
            X_unlabelled: Feature matrix of unlabelled samples.
            high_threshold: Probability above which samples are labelled fraud.
            low_threshold: Probability below which samples are labelled legit.

        Returns:
            Dict with keys:
              - X_pseudo: Feature matrix of pseudo-labelled samples.
              - y_pseudo: Pseudo labels (0 or 1).
              - high_conf_count: Number of high-confidence fraud pseudo-labels.
              - low_conf_count: Number of low-confidence legit pseudo-labels.
        """
        ...

    @abstractmethod
    def confidence(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate prediction confidence for each sample.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Array of confidence values in [0, 1] (1 = certain).
        """
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Persist the learner to disk.

        Args:
            path: Directory to save model artifacts.
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "AdaptiveLearner":
        """
        Load a previously saved learner from disk.

        Args:
            path: Directory containing saved model artifacts.

        Returns:
            Loaded AdaptiveLearner instance.
        """
        ...

    @abstractmethod
    def explain(self, X: np.ndarray, top_n: int = 8) -> List[Dict[str, Any]]:
        """
        Generate feature attributions for predictions.

        Args:
            X: Feature matrix (n_samples, n_features).
            top_n: Number of top features to return.

        Returns:
            List of explanation dicts, one per sample.
        """
        ...
