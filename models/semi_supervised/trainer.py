"""
RiskLens — Phase 2: Semi-Supervised Training Pipeline

Manages the full training lifecycle for the TabPFN-based semi-supervised model:
  Pseudo-label generation → Dataset construction → TabPFN fit → Calibration → Validation

TabPFN uses in-context learning: calling ``fit()`` stores the labelled data.
No gradient-based training loop is required.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from models.cold_start.ensemble import ColdStartEnsemble
from models.semi_supervised.tabpfn import TabPFNModel
from models.semi_supervised.prediction import PseudoLabelResult
from scoring.calibration import ProbabilityCalibrator

import logging

logger = logging.getLogger(__name__)


@dataclass
class SemiSupervisedConfig:
    """Configuration for Phase 2 training."""

    # TabPFN hyperparameters
    n_estimators: int = 4
    ignore_pretraining_limits: bool = True

    # Pseudo-labeling
    pseudo_label_threshold: float = 0.95
    pseudo_low_threshold: float = 0.10
    pseudo_label_weight: float = 1.0
    confirmed_label_weight: float = 3.0

    # Calibration
    calibration_method: str = "isotonic"

    # Data
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    random_seed: int = 42


@dataclass
class SemiSupervisedTrainingResult:
    """Result of a Phase 2 training run."""

    model_version: str = ""
    n_confirmed: int = 0
    n_pseudo: int = 0
    n_total: int = 0
    pr_auc: float = 0.0
    roc_auc: float = 0.0
    duration_seconds: float = 0.0
    calibration_error: float = 0.0
    status: str = "pending"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_version": self.model_version,
            "n_confirmed": self.n_confirmed,
            "n_pseudo": self.n_pseudo,
            "n_total": self.n_total,
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "duration_seconds": self.duration_seconds,
            "calibration_error": self.calibration_error,
            "status": self.status,
            "error": self.error,
        }


class SemiSupervisedTrainer:
    """
    Training pipeline for the Phase 2 TabPFN model.

    Workflow:
    1. Generate pseudo-labels from cold-start ensemble
    2. Combine confirmed + pseudo labels
    3. Fit TabPFN on combined dataset (in-context learning)
    4. Calibrate probabilities on validation set
    5. Validate on held-out set
    """

    def __init__(self, config: Optional[SemiSupervisedConfig] = None):
        self.config = config or SemiSupervisedConfig()

    def prepare_dataset(
        self,
        X_confirmed: np.ndarray,
        y_confirmed: np.ndarray,
        X_unlabelled: Optional[np.ndarray] = None,
        cold_start: Optional[ColdStartEnsemble] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PseudoLabelResult]:
        """
        Combine confirmed labels with pseudo-labels from cold-start.

        Returns:
            X_train: Combined feature matrix
            y_train: Combined labels
            sample_weights: Per-sample weights
            pseudo_result: Pseudo-label generation details
        """
        if (
            X_unlabelled is not None
            and len(X_unlabelled) > 0
            and cold_start is not None
        ):
            pseudo_result = self.generate_pseudo_labels(X_unlabelled, cold_start)
            X_pseudo = pseudo_result.X_pseudo
            y_pseudo = pseudo_result.y_pseudo

            if len(X_pseudo) > 0:
                X_combined = np.vstack([X_confirmed, X_pseudo])
                y_combined = np.concatenate([y_confirmed, y_pseudo])

                w_confirmed = np.full(
                    len(y_confirmed), self.config.confirmed_label_weight
                )
                w_pseudo = np.full(len(y_pseudo), self.config.pseudo_label_weight)
                weights = np.concatenate([w_confirmed, w_pseudo])
            else:
                X_combined = X_confirmed
                y_combined = y_confirmed
                weights = np.full(len(y_confirmed), 1.0)
        else:
            X_combined = X_confirmed
            y_combined = y_confirmed
            weights = np.full(len(y_confirmed), 1.0)
            pseudo_result = PseudoLabelResult(
                X_pseudo=np.array([]),
                y_pseudo=np.array([]),
            )

        logger.info(
            "Dataset prepared: %d confirmed + %d pseudo = %d total "
            "(fraud rate: %.3f%%)",
            len(y_confirmed),
            pseudo_result.high_conf_count + pseudo_result.low_conf_count,
            len(y_combined),
            y_combined.mean() * 100,
        )

        return X_combined, y_combined, weights, pseudo_result

    def generate_pseudo_labels(
        self,
        X_unlabelled: np.ndarray,
        cold_start: ColdStartEnsemble,
    ) -> PseudoLabelResult:
        """
        Score unlabelled data with cold-start ensemble and assign pseudo-labels.
        """
        scores = cold_start.score(X_unlabelled)

        high_mask = scores >= self.config.pseudo_label_threshold
        low_mask = scores <= self.config.pseudo_low_threshold
        confident_mask = high_mask | low_mask

        X_pseudo = X_unlabelled[confident_mask]
        y_pseudo = (
            scores[confident_mask] >= self.config.pseudo_label_threshold
        ).astype(int)

        n_high = int(high_mask.sum())
        n_low = int(low_mask.sum())
        n_review = int((~confident_mask).sum())

        logger.info(
            "Pseudo-labels: %d high-confidence fraud, %d low-confidence legit, "
            "%d sent to review (thresholds: high=%.2f, low=%.2f)",
            n_high,
            n_low,
            n_review,
            self.config.pseudo_label_threshold,
            self.config.pseudo_low_threshold,
        )

        return PseudoLabelResult(
            X_pseudo=X_pseudo,
            y_pseudo=y_pseudo,
            high_conf_count=n_high,
            low_conf_count=n_low,
            pseudo_high_threshold=self.config.pseudo_label_threshold,
            pseudo_low_threshold=self.config.pseudo_low_threshold,
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        dataset_version: str = "1.0.0",
    ) -> Tuple[TabPFNModel, SemiSupervisedTrainingResult]:
        """
        Fit TabPFN on the labelled dataset.

        Returns:
            (fitted wrapper, training result)
        """
        t_start = time.time()
        result = SemiSupervisedTrainingResult()

        try:
            input_dim = X.shape[1]
            feature_names = feature_names or [f"f_{i}" for i in range(input_dim)]

            # Split data
            X_train, X_val, y_train, y_val = self._split_data(X, y)

            result.n_confirmed = int(y.sum())
            result.n_total = len(y)

            # Build wrapper
            wrapper = TabPFNModel(
                input_dim=input_dim,
                feature_names=feature_names,
                calibration_method=self.config.calibration_method,
                n_estimators=self.config.n_estimators,
                ignore_pretraining_limits=self.config.ignore_pretraining_limits,
            )

            # Fit TabPFN (in-context learning — stores training data)
            wrapper.fit(X_train, y_train, sample_weights=None)

            # Calibrate on validation set
            val_probs_raw = wrapper.predict_proba(X_val)

            wrapper.calibrator = ProbabilityCalibrator(
                method=self.config.calibration_method
            )
            wrapper.calibrator.fit(val_probs_raw, y_val)

            # Final metrics
            val_probs_cal = wrapper.calibrator.transform(val_probs_raw)
            result.pr_auc = float(average_precision_score(y_val, val_probs_cal))
            result.roc_auc = float(roc_auc_score(y_val, val_probs_cal))

            # Compute calibration error
            fraction_pos, mean_predicted = calibration_curve(
                y_val, val_probs_cal, n_bins=10
            )
            result.calibration_error = float(
                np.mean(np.abs(fraction_pos - mean_predicted))
            )

            result.status = "completed"

            wrapper.is_fitted = True
            wrapper.pr_auc_ = result.pr_auc
            wrapper.roc_auc_ = result.roc_auc
            wrapper.trained_at = datetime.now(timezone.utc).isoformat()
            wrapper.model_version = f"v2_tabpfn_{int(time.time())}"

            result.model_version = wrapper.model_version

        except Exception as exc:
            result.status = "failed"
            result.error = str(exc)
            logger.error("TabPFN training failed: %s", exc)
            raise

        finally:
            result.duration_seconds = time.time() - t_start

        logger.info(
            "TabPFN trained in %.1fs — PR-AUC: %.4f, ROC-AUC: %.4f, "
            "calibration error: %.4f",
            result.duration_seconds,
            result.pr_auc,
            result.roc_auc,
            result.calibration_error,
        )

        return wrapper, result

    def _split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        val_ratio_adjusted = self.config.val_ratio / (
            self.config.train_ratio + self.config.val_ratio
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=val_ratio_adjusted,
            random_state=self.config.random_seed,
            stratify=y,
        )

        return X_train, X_val, y_train, y_val
