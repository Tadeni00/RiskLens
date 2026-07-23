"""
FraudTrap — Phase 2: TabPFN (Tabular Prior-data Fitted Network)

A pretrained tabular foundation model by Prior Labs used as the core of
the semi-supervised learning layer. TabPFN performs prediction via
in-context learning — it stores the labelled training set and computes
predictions in a single forward pass through a transformer architecture.

Unlike gradient-boosted trees or neural networks trained from scratch:
  - It is pretrained on synthetic tabular tasks
  - It excels on small-to-medium labelled datasets (≤50 000 samples)
  - It provides naturally calibrated probabilities
  - It requires no hyperparameter tuning for strong baselines

Architecture:
  Input Features → Distribution Embedder → Row/Cross-row Attention
  → Per-row Tokens → Calibrated Class Probabilities

Reference:
  Hollmann et al., "Accurate predictions on small data with a tabular
  foundation model", Nature (2025).
  https://github.com/PriorLabs/TabPFN
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

from models.semi_supervised.prediction import SemiSupervisedPrediction

logger = logging.getLogger(__name__)


# ── Entropy-based Uncertainty ─────────────────────────────────────────────────


def _entropy_uncertainty(probas: np.ndarray) -> np.ndarray:
    """
    Compute prediction uncertainty from class probability entropy.

    For binary classification with probabilities p = [p_legit, p_fraud]:
        H(p) = -p_legit * log(p_legit) - p_fraud * log(p_fraud)
        Normalised: H_norm = H(p) / log(2)   (max entropy for 2 classes)

    Returns values in [0, 1] where 0 = certain, 1 = maximally uncertain.
    """
    eps = 1e-12
    probas_clipped = np.clip(probas, eps, 1.0 - eps)
    if probas_clipped.ndim == 1:
        p = np.stack([1.0 - probas_clipped, probas_clipped], axis=1)
    else:
        p = probas_clipped
    H = -(p * np.log(p)).sum(axis=1)
    return H / np.log(2)


# ── TabPFN Model ─────────────────────────────────────────────────────────────


class TabPFNModel:
    """
    Production wrapper around Prior Labs TabPFN for FraudTrap's semi-supervised
    layer.

    Interface:
        fit(X, y, sample_weights)          – store labelled data
        predict(X)                         – hard class predictions
        predict_proba(X)                   – calibrated fraud probabilities
        predict_with_uncertainty(X)        – typed predictions with uncertainty
        score(X)                           – alias for predict_proba
        explain(X, top_n)                  – feature attributions
        save(path) / load(path)            – persistence

    TabPFN performs in-context learning: calling ``fit()`` stores the
    labelled dataset; calling ``predict*`` runs a forward pass that
    conditions on the stored data.  There is no gradient-based training.

    Uncertainty is derived from prediction entropy.  This is a meaningful
    signal for TabPFN because its transformer architecture produces
    well-calibrated probabilities even on small datasets.
    """

    def __init__(
        self,
        input_dim: int = 0,
        feature_names: Optional[List[str]] = None,
        calibration_method: str = "isotonic",
        model_version: str = "1.0.0",
        n_estimators: int = 4,
        ignore_pretraining_limits: bool = True,
    ):
        self.input_dim = input_dim
        self.feature_names = feature_names or []
        self.calibration_method = calibration_method
        self.model_version = model_version
        self.n_estimators = n_estimators
        self.ignore_pretraining_limits = ignore_pretraining_limits

        self.scaler = StandardScaler()
        self.calibrator: Optional[Any] = None
        self.is_fitted = False

        # Version pinning / metadata
        self.training_hash: Optional[str] = None
        self.feature_hash: Optional[str] = None
        self.dataset_hash: Optional[str] = None
        self.trained_at: Optional[str] = None
        self.confirmed_label_count: int = 0
        self.pseudo_label_count: int = 0
        self.training_iteration: int = 0
        self.pr_auc_: float = 0.0
        self.roc_auc_: float = 0.0

        # TabPFN classifier (lazy — created in fit())
        self._classifier: Any = None

    # ── Internal helpers ──────────────────────────────────────────────────

    def _get_classifier(self) -> Any:
        """Lazily import and return a TabPFNClassifier instance."""
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as exc:
            raise ImportError(
                "tabpfn package is required.  Install with: pip install tabpfn"
            ) from exc

        return TabPFNClassifier(
            n_estimators=self.n_estimators,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
        )

    def _compute_hashes(self, X: np.ndarray, y: np.ndarray) -> Dict[str, str]:
        feature_str = (
            "|".join(self.feature_names) if self.feature_names else str(X.shape[1])
        )
        feature_hash = hashlib.sha256(feature_str.encode()).hexdigest()[:16]

        if len(X) > 10_000:
            idx = np.random.default_rng(42).choice(len(X), 10_000, replace=False)
            X_sample, y_sample = X[idx], y[idx]
        else:
            X_sample, y_sample = X, y
        data_stats = np.concatenate(
            [
                X_sample.mean(axis=0),
                X_sample.std(axis=0),
                [y_sample.mean(), len(y_sample)],
            ]
        )
        dataset_hash = hashlib.sha256(data_stats.tobytes()).hexdigest()[:16]
        return {"feature_hash": feature_hash, "dataset_hash": dataset_hash}

    # ── Public API ────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> "TabPFNModel":
        """
        Fit the TabPFN model on the labelled training data.

        In TabPFN, ``fit`` stores the training set for in-context
        learning at prediction time.  No gradient updates occur.
        """
        t0 = time.time()

        self.input_dim = X.shape[1]
        X_scaled = self.scaler.fit_transform(X)

        self._classifier = self._get_classifier()
        self._classifier.fit(X_scaled, y)

        self.is_fitted = True
        self.trained_at = datetime.now(timezone.utc).isoformat()

        hashes = self._compute_hashes(X, y)
        self.feature_hash = hashes["feature_hash"]
        self.dataset_hash = hashes["dataset_hash"]

        logger.info(
            "TabPFN fitted on %d samples (%d features) in %.1fs",
            X.shape[0],
            X.shape[1],
            time.time() - t0,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard class predictions (0 or 1)."""
        if not self.is_fitted:
            raise RuntimeError("TabPFNModel must be fitted before predict")
        X_scaled = self.scaler.transform(X)
        return self._classifier.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated fraud probabilities (1-D array)."""
        if not self.is_fitted:
            raise RuntimeError("TabPFNModel must be fitted before scoring")
        X_scaled = self.scaler.transform(X)
        probas_2d = self._classifier.predict_proba(X_scaled)
        fraud_prob = probas_2d[:, 1]

        if self.calibrator is not None:
            fraud_prob = self.calibrator.transform(fraud_prob)
        return fraud_prob

    def score(self, X: np.ndarray) -> np.ndarray:
        """Alias for predict_proba — returns calibrated fraud probabilities."""
        return self.predict_proba(X)

    def predict_with_uncertainty(self, X: np.ndarray) -> List[SemiSupervisedPrediction]:
        """Return fully typed predictions with uncertainty estimates."""
        if not self.is_fitted:
            raise RuntimeError("TabPFNModel must be fitted before scoring")
        X_scaled = self.scaler.transform(X)
        probas_2d = self._classifier.predict_proba(X_scaled)
        fraud_prob = probas_2d[:, 1]

        if self.calibrator is not None:
            fraud_prob = self.calibrator.transform(fraud_prob)

        uncertainty = _entropy_uncertainty(probas_2d)
        confidence = 1.0 - uncertainty

        return [
            SemiSupervisedPrediction(
                probability=float(fraud_prob[i]),
                confidence=float(confidence[i]),
                uncertainty=float(uncertainty[i]),
                model_version=self.model_version,
            )
            for i in range(len(X))
        ]

    def explain(self, X: np.ndarray, top_n: int = 8) -> List[Dict[str, Any]]:
        """
        Feature attribution via permutation-based importance.

        For each sample, measure how much each feature contributes to the
        prediction by observing the change in fraud probability when that
        feature is replaced with its median value from the training set.
        """
        if not self.is_fitted:
            raise RuntimeError("TabPFNModel must be fitted before explain")

        X_scaled = self.scaler.transform(X)
        base_probas = self._classifier.predict_proba(X_scaled)
        base_fraud = base_probas[:, 1]

        # Reference: median of training data (stored in scaler)
        x_median = self.scaler.center_

        feature_names = (
            self.feature_names
            if self.feature_names
            else [f"f_{j}" for j in range(self.input_dim)]
        )

        explanations = []
        for i in range(len(X)):
            x_orig = X_scaled[i : i + 1].copy()
            feat_importances = []

            for j in range(self.input_dim):
                x_perturbed_j = x_orig.copy()
                x_perturbed_j[0, j] = x_median[j]
                p_perturbed = self._classifier.predict_proba(x_perturbed_j)[0, 1]
                importance = abs(float(base_fraud[i] - p_perturbed))
                feat_importances.append(
                    (feature_names[j], float(x_orig[0, j]), importance)
                )

            feat_importances.sort(key=lambda t: t[2], reverse=True)
            top_features = [
                {
                    "feature": name,
                    "value": val,
                    "contribution": imp,
                    "method": "permutation_importance",
                }
                for name, val, imp in feat_importances[:top_n]
            ]

            explanations.append(
                {
                    "model_type": "tabpfn",
                    "base_value": float(np.mean(base_fraud)),
                    "prediction_value": float(base_fraud[i]),
                    "top_features": top_features,
                    "components": {
                        "n_training_samples": self._n_training_samples(),
                        "model_version": self.model_version,
                    },
                }
            )
        return explanations

    def _n_training_samples(self) -> int:
        """Return the number of training samples stored by TabPFN."""
        if self._classifier is None:
            return 0
        X_train = getattr(self._classifier, "X_train_", None)
        if X_train is not None:
            return len(X_train)
        return 0

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Save the wrapper and all metadata to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        import joblib

        joblib.dump(self._classifier, path / "tabpfn_model.joblib")

        with open(path / "tabpfn_metadata.pkl", "wb") as f:
            pickle.dump(
                {
                    "input_dim": self.input_dim,
                    "feature_names": self.feature_names,
                    "calibration_method": self.calibration_method,
                    "model_version": self.model_version,
                    "n_estimators": self.n_estimators,
                    "ignore_pretraining_limits": self.ignore_pretraining_limits,
                    "scaler": self.scaler,
                    "calibrator": self.calibrator,
                    "is_fitted": self.is_fitted,
                    "training_hash": self.training_hash,
                    "feature_hash": self.feature_hash,
                    "dataset_hash": self.dataset_hash,
                    "trained_at": self.trained_at,
                    "confirmed_label_count": self.confirmed_label_count,
                    "pseudo_label_count": self.pseudo_label_count,
                    "training_iteration": self.training_iteration,
                    "pr_auc": self.pr_auc_,
                    "roc_auc": self.roc_auc_,
                },
                f,
            )

        logger.info("TabPFNModel saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "TabPFNModel":
        """Load a previously saved model from disk."""
        path = Path(path)

        # Try new filename first, fall back to legacy
        metadata_path = path / "tabpfn_metadata.pkl"
        if not metadata_path.exists():
            metadata_path = path / "netpfn_metadata.pkl"

        with open(metadata_path, "rb") as f:
            payload = pickle.load(f)

        obj = cls(
            input_dim=payload.get("input_dim", 0),
            feature_names=payload.get("feature_names", []),
            calibration_method=payload.get("calibration_method", "isotonic"),
            model_version=payload.get("model_version", "1.0.0"),
            n_estimators=payload.get("n_estimators", 4),
            ignore_pretraining_limits=payload.get("ignore_pretraining_limits", True),
        )

        obj.scaler = payload["scaler"]
        obj.calibrator = payload.get("calibrator")
        obj.is_fitted = payload.get("is_fitted", False)
        obj.training_hash = payload.get("training_hash")
        obj.feature_hash = payload.get("feature_hash")
        obj.dataset_hash = payload.get("dataset_hash")
        obj.trained_at = payload.get("trained_at")
        obj.confirmed_label_count = payload.get("confirmed_label_count", 0)
        obj.pseudo_label_count = payload.get("pseudo_label_count", 0)
        obj.training_iteration = payload.get("training_iteration", 0)
        obj.pr_auc_ = payload.get("pr_auc", 0.0)
        obj.roc_auc_ = payload.get("roc_auc", 0.0)

        import joblib

        model_path = path / "tabpfn_model.joblib"
        if model_path.exists():
            obj._classifier = joblib.load(model_path)
        else:
            obj._classifier = None
            obj.is_fitted = False

        logger.info(
            "TabPFNModel loaded from %s (version=%s)",
            path,
            obj.model_version,
        )
        return obj
