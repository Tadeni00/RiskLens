"""
RiskLens — SHAP Explainer
Computes local and global feature attributions for CatBoost models.
Uses TreeExplainer for exact Shapley values in O(TLD) time.
"""

from __future__ import annotations
import time
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
from loguru import logger

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed — feature attribution unavailable")

from models.explainability.types import FeatureAttribution, SHAPExplanation


class SHAPExplainer:
    """
    SHAP-based feature attribution for CatBoost (and other tree models).

    Uses shap.TreeExplainer for exact, fast computations.
    Caches the explainer object to avoid repeated initialization.

    Performance: <20ms for single-sample explanation on CatBoost.
    """

    def __init__(self, top_features: int = 5):
        self.top_features = top_features
        self._explainer: Optional[Any] = None
        self._feature_names: List[str] = []
        self._model = None
        self._is_fitted = False

    def fit(self, model, X_background: np.ndarray, feature_names: List[str]) -> None:
        """
        Initialize the SHAP explainer with a background dataset.

        Args:
            model: Fitted CatBoost model (or CatBoostClassifier)
            X_background: Background dataset for SHAP (subset of training data)
            feature_names: Ordered list of feature names
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available, skipping explainer initialization")
            return

        self._model = model
        self._feature_names = feature_names

        try:
            # CatBoost has native SHAP support via TreeExplainer
            self._explainer = shap.TreeExplainer(model)
            self._is_fitted = True
            logger.info(
                "SHAP TreeExplainer fitted (features={}, background={})",
                len(feature_names),
                len(X_background),
            )
        except Exception as exc:
            logger.warning(
                "SHAP TreeExplainer init failed, trying KernelExplainer: {}", exc
            )
            try:
                # Fallback: KernelExplainer (slower but works for any model)
                background = shap.sample(X_background, min(100, len(X_background)))
                self._explainer = shap.KernelExplainer(
                    lambda x: (
                        model.predict_proba(x)[:, 1]
                        if hasattr(model, "predict_proba")
                        else model.predict(x)
                    ),
                    background,
                )
                self._is_fitted = True
                logger.info("SHAP KernelExplainer fitted as fallback")
            except Exception as exc2:
                logger.error("SHAP explainer init failed completely: {}", exc2)
                self._is_fitted = False

    def explain(self, X: np.ndarray, top_n: Optional[int] = None) -> SHAPExplanation:
        """
        Compute SHAP explanation for a single transaction.

        Args:
            X: Feature array shape (1, n_features) or (n_features,)
            top_n: Number of top features to return (default: self.top_features)

        Returns:
            SHAPExplanation with feature attributions
        """
        t_start = time.perf_counter()
        top_n = top_n or self.top_features

        if not self._is_fitted or self._explainer is None:
            return self._fallback_explanation(X, t_start)

        try:
            X_input = X.reshape(1, -1) if X.ndim == 1 else X
            shap_values = self._explainer.shap_values(X_input)

            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                sv = shap_values

            sv_flat = sv.flatten() if sv.ndim > 1 else sv
            x_flat = X_input.flatten()

            # Get fraud probability
            fraud_prob = float(np.clip(self._get_prediction(X_input), 0.0, 1.0))

            # Base value
            expected_value = self._explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                base_value = float(
                    expected_value[1] if len(expected_value) > 1 else expected_value[0]
                )
            else:
                base_value = float(expected_value)

            # Build attributions sorted by absolute impact
            attributions = []
            for i in range(len(sv_flat)):
                fname = (
                    self._feature_names[i]
                    if i < len(self._feature_names)
                    else f"feature_{i}"
                )
                impact = float(sv_flat[i])
                attributions.append(
                    FeatureAttribution(
                        feature=fname,
                        value=float(x_flat[i]),
                        impact=impact,
                        direction="increase" if impact > 0 else "decrease",
                        method="shap",
                    )
                )

            # Sort by absolute impact, take top N
            attributions.sort(key=lambda a: abs(a.impact), reverse=True)
            top_attributions = tuple(attributions[:top_n])

            latency_ms = (time.perf_counter() - t_start) * 1000

            return SHAPExplanation(
                fraud_probability=fraud_prob,
                base_value=base_value,
                top_features=top_attributions,
                all_shap_values=tuple(sv_flat.tolist()),
                latency_ms=round(latency_ms, 2),
            )

        except Exception as exc:
            logger.warning("SHAP explanation failed: {}", exc)
            return self._fallback_explanation(X, t_start)

    def explain_batch(
        self, X: np.ndarray, top_n: Optional[int] = None
    ) -> List[SHAPExplanation]:
        """Compute SHAP explanations for a batch of transactions."""
        return [self.explain(X[i : i + 1], top_n) for i in range(len(X))]

    def global_importance(
        self, X: np.ndarray, top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Compute global feature importance via mean |SHAP| over a dataset.

        Args:
            X: Dataset of shape (n_samples, n_features)
            top_n: Number of top features to return

        Returns:
            List of {"feature": name, "importance": float}
        """
        if not self._is_fitted or self._explainer is None:
            return []

        top_n = top_n or self.top_features

        try:
            shap_values = self._explainer.shap_values(X)
            if isinstance(shap_values, list):
                sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                sv = shap_values

            mean_abs = np.mean(np.abs(sv), axis=0)
            indices = np.argsort(mean_abs)[::-1][:top_n]

            return [
                {
                    "feature": (
                        self._feature_names[i]
                        if i < len(self._feature_names)
                        else f"feature_{i}"
                    ),
                    "importance": float(mean_abs[i]),
                }
                for i in indices
            ]
        except Exception as exc:
            logger.warning("Global SHAP importance failed: {}", exc)
            return []

    def _get_prediction(self, X: np.ndarray) -> float:
        """Get model prediction for a single sample."""
        try:
            if hasattr(self._model, "predict_proba"):
                proba = self._model.predict_proba(X)
                return float(proba[0, 1]) if proba.ndim > 1 else float(proba[0])
            return float(self._model.predict(X)[0])
        except Exception:
            return 0.5

    def _fallback_explanation(self, X: np.ndarray, t_start: float) -> SHAPExplanation:
        """Fallback when SHAP is unavailable — returns empty attributions."""
        latency_ms = (time.perf_counter() - t_start) * 1000
        return SHAPExplanation(
            fraud_probability=0.5,
            base_value=0.0,
            top_features=(),
            latency_ms=round(latency_ms, 2),
        )

    def save(self, path: Path) -> None:
        """Persist the SHAP explainer and metadata."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "shap_explainer.pkl", "wb") as f:
            pickle.dump(
                {
                    "explainer": self._explainer,
                    "feature_names": self._feature_names,
                    "top_features": self.top_features,
                    "is_fitted": self._is_fitted,
                },
                f,
            )

        logger.info("SHAPExplainer saved to {}", path)

    @classmethod
    def load(cls, path: Path) -> "SHAPExplainer":
        """Load a persisted SHAP explainer."""
        path = Path(path)

        with open(path / "shap_explainer.pkl", "rb") as f:
            payload = pickle.load(f)

        obj = cls(top_features=payload["top_features"])
        obj._explainer = payload["explainer"]
        obj._feature_names = payload["feature_names"]
        obj._is_fitted = payload["is_fitted"]

        logger.info("SHAPExplainer loaded from {}", path)
        return obj
