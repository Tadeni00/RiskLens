"""
RiskLens — Phase 3: Confidence Estimator
Determines whether CatBoost's prediction is confident enough to return directly,
or whether FT-Transformer consultation is needed.

Design:
- Initially computes confidence from calibrated probability distance to threshold
- Extensible for advanced uncertainty estimation (MC dropout, conformal prediction)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from loguru import logger


@dataclass
class ConfidenceEstimatorConfig:
    """Configuration for the confidence estimator."""

    threshold: float = 0.92
    uncertainty_margin: float = 0.05
    use_conformal: bool = False
    conformal_alpha: float = 0.1


class ConfidenceEstimator:
    """
    Estimates whether CatBoost is confident enough for a given prediction.

    When the model is NOT confident, the transaction is routed to
    FT-Transformer for a second opinion.

    Confidence is computed as:
    - Distance from the calibrated probability to the decision boundary
    - Optionally: conformal prediction uncertainty sets

    The module is designed for extensibility — more advanced uncertainty
    estimation methods (MC dropout, ensemble disagreement) can be added
    without changing the interface.
    """

    def __init__(self, config: Optional[ConfidenceEstimatorConfig] = None):
        self.config = config or ConfidenceEstimatorConfig()
        self._conformal_scores: Optional[np.ndarray] = None
        self._conformal_threshold: Optional[float] = None

    def estimate(self, probability: float) -> float:
        """
        Estimate confidence for a single prediction.

        Confidence is the minimum distance from the probability to either
        decision boundary (0 or 1), scaled to [0, 1].
        """
        dist_to_0 = probability
        dist_to_1 = 1.0 - probability
        min_distance = min(dist_to_0, dist_to_1)

        # Scale: 0 at boundary, 1 at center (0.5)
        confidence = min_distance * 2.0

        return float(np.clip(confidence, 0.0, 1.0))

    def is_confident(self, probability: float) -> bool:
        """
        Determine if CatBoost is confident enough for this prediction.

        Returns True if the prediction should be returned directly
        without FT-Transformer consultation.
        """
        confidence = self.estimate(probability)
        return confidence >= self.config.threshold

    def threshold(self) -> float:
        """Return the current confidence threshold."""
        return self.config.threshold

    def set_threshold(self, threshold: float) -> None:
        """Update the confidence threshold."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")
        self.config.threshold = threshold
        logger.info("Confidence threshold updated to {:.3f}", threshold)

    def fit_conformal(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """
        Fit conformal prediction threshold on a calibration set.

        This provides distribution-free uncertainty quantification:
        with probability >= (1 - alpha), the true label is in the
        prediction set.
        """
        # Conformal scores: 1 - p(y=1) for positive, p(y=1) for negative
        conformal_scores = np.where(
            labels == 1,
            1 - probabilities,
            probabilities,
        )
        self._conformal_scores = conformal_scores
        self._conformal_threshold = float(
            np.quantile(
                conformal_scores,
                1 - self.config.conformal_alpha,
                method="higher",
            )
        )
        logger.info(
            "Conformal threshold fitted: {:.4f} (alpha={:.2f})",
            self._conformal_threshold,
            self.config.conformal_alpha,
        )

    def predict_set(self, probability: float) -> bool:
        """
        Conformal prediction: returns True if the prediction set
        includes class 1 (fraud).

        Only available after fit_conformal() has been called.
        """
        if self._conformal_threshold is None:
            # Fallback to distance-based confidence
            return self.is_confident(probability)

        score = 1 - probability  # fraud side score
        return score > self._conformal_threshold
