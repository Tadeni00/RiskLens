"""
FraudTrap — Phase 3: Meta Fusion Layer
Combines CatBoost and FT-Transformer predictions when FT-Transformer
is invoked for low-confidence transactions.

Uses a trainable meta-learner (Logistic Regression or small GBDT)
to optimally combine the two model outputs.

Inputs:
  - CatBoost probability
  - FT-Transformer probability
  - CatBoost confidence
  - Optional behaviour/risk features

Output:
  - Final fraud probability
"""

from __future__ import annotations
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from loguru import logger


class MetaFusionLayer:
    """
    Trainable meta-fusion layer that combines CatBoost and FT-Transformer
    predictions into a single calibrated probability.

    Two fusion strategies:
    1. Logistic Regression (default): fast, interpretable, calibrated
    2. Small GBDT: more flexible, captures non-linear interactions

    The fusion model is trained on a held-out calibration set where both
    CatBoost and FT-Transformer predictions are available.
    """

    def __init__(
        self,
        method: str = "logistic_regression",
        use_meta_features: bool = True,
    ):
        """
        Args:
            method: "logistic_regression" or "gradient_boosting"
            use_meta_features: Whether to include additional meta-features
                (confidence, behaviour features) in the fusion
        """
        self.method = method
        self.use_meta_features = use_meta_features
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _build_meta_features(
        self,
        catboost_probs: np.ndarray,
        ft_probs: np.ndarray,
        catboost_confidences: np.ndarray,
        extra_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Build meta-features for the fusion model."""
        meta = np.column_stack(
            [
                catboost_probs,
                ft_probs,
                catboost_confidences,
                np.abs(catboost_probs - ft_probs),  # disagreement feature
            ]
        )

        if self.use_meta_features and extra_features is not None:
            meta = np.column_stack([meta, extra_features])

        return meta

    def fit(
        self,
        catboost_probs: np.ndarray,
        ft_probs: np.ndarray,
        catboost_confidences: np.ndarray,
        y_true: np.ndarray,
        extra_features: Optional[np.ndarray] = None,
    ) -> "MetaFusionLayer":
        """
        Train the meta-fusion model on a calibration set.

        Args:
            catboost_probs: CatBoost predictions on calibration set
            ft_probs: FT-Transformer predictions on calibration set
            catboost_confidences: CatBoost confidence scores
            y_true: True labels
            extra_features: Optional additional meta-features
        """
        meta_X = self._build_meta_features(
            catboost_probs, ft_probs, catboost_confidences, extra_features
        )
        meta_X_scaled = self.scaler.fit_transform(meta_X)

        if self.method == "logistic_regression":
            self.model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        elif self.method == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingClassifier

            self.model = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown fusion method: {self.method}")

        self.model.fit(meta_X_scaled, y_true)
        self.is_fitted = True

        logger.info(
            "MetaFusionLayer trained (method={}) on {} samples",
            self.method,
            len(y_true),
        )
        return self

    def predict(
        self,
        catboost_prob: float,
        ft_prob: float,
        catboost_confidence: float,
        extra_features: Optional[np.ndarray] = None,
    ) -> float:
        """
        Fuse CatBoost and FT-Transformer predictions.

        Returns final calibrated fraud probability.
        """
        if not self.is_fitted:
            # Fallback: weighted average
            return 0.7 * catboost_prob + 0.3 * ft_prob

        meta_X = self._build_meta_features(
            np.array([catboost_prob]),
            np.array([ft_prob]),
            np.array([catboost_confidence]),
            extra_features.reshape(1, -1) if extra_features is not None else None,
        )
        meta_X_scaled = self.scaler.transform(meta_X)

        raw = self.model.predict_proba(meta_X_scaled)[:, 1]
        return float(raw[0])

    def predict_batch(
        self,
        catboost_probs: np.ndarray,
        ft_probs: np.ndarray,
        catboost_confidences: np.ndarray,
        extra_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Batch fusion for efficiency."""
        if not self.is_fitted:
            return 0.7 * catboost_probs + 0.3 * ft_probs

        meta_X = self._build_meta_features(
            catboost_probs, ft_probs, catboost_confidences, extra_features
        )
        meta_X_scaled = self.scaler.transform(meta_X)
        raw = self.model.predict_proba(meta_X_scaled)[:, 1]
        return raw

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "meta_fusion.pkl", "wb") as f:
            pickle.dump(
                {
                    "method": self.method,
                    "use_meta_features": self.use_meta_features,
                    "model": self.model,
                    "scaler": self.scaler,
                    "is_fitted": self.is_fitted,
                },
                f,
            )

        logger.info("MetaFusionLayer saved to {}", path)

    @classmethod
    def load(cls, path: Path) -> "MetaFusionLayer":
        path = Path(path)

        with open(path / "meta_fusion.pkl", "rb") as f:
            payload = pickle.load(f)

        obj = cls(
            method=payload["method"],
            use_meta_features=payload["use_meta_features"],
        )
        obj.model = payload["model"]
        obj.scaler = payload["scaler"]
        obj.is_fitted = payload["is_fitted"]

        logger.info("MetaFusionLayer loaded from {} (method={})", path, obj.method)
        return obj
