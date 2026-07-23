"""
FraudTrap — Champion Model (CatBoost) with Confidence-Aware Routing
Single-model production fraud detector with native categorical handling.
Includes optional FT-Transformer specialist consultation for low-confidence cases.

Architecture:
  Transaction → CatBoost → Calibrated Probability → Confidence Estimator
    → High Confidence? → Return prediction directly
    → Low Confidence?  → FT-Transformer → Meta Fusion → Final probability
"""

from __future__ import annotations
import hashlib
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from loguru import logger

from scoring.calibration import ProbabilityCalibrator
from models.supervised.prediction import SupervisedPrediction
from models.supervised.confidence import ConfidenceEstimator, ConfidenceEstimatorConfig
from models.supervised.meta_fusion import MetaFusionLayer


class ChampionModel:
    """
    Production CatBoost fraud detector with confidence-aware routing.

    Key features:
    - Native categorical feature handling (no target encoding leakage)
    - Built-in class imbalance handling
    - Fast inference with GPU support
    - Built-in feature importance
    - Probability calibration support
    - Version pinning for reproducibility
    - Optional FT-Transformer specialist for low-confidence cases
    - Trainable meta-fusion layer
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[int]] = None,
        cat_feature_names: Optional[List[str]] = None,
        iterations: int = 500,
        depth: int = 6,
        learning_rate: float = 0.05,
        confidence_threshold: float = 0.92,
        enable_specialist: bool = False,
        fusion_method: str = "logistic_regression",
        **catboost_kwargs,
    ):
        self.feature_names = feature_names or []
        self.categorical_features = categorical_features or []
        self.cat_feature_names = cat_feature_names or []

        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.catboost_kwargs = catboost_kwargs

        self.model: Optional[CatBoostClassifier] = None
        self.calibrator: Optional[ProbabilityCalibrator] = None
        self.is_fitted: bool = False

        # Metrics
        self.pr_auc_: float = 0.0
        self.f2_score_: float = 0.0
        self.roc_auc_: float = 0.0

        # Calibration
        self.calibration_method: str = "isotonic"

        # Version pinning
        self.model_version: str = "1.0.0"
        self.training_hash: Optional[str] = None
        self.feature_hash: Optional[str] = None
        self.dataset_hash: Optional[str] = None
        self.trained_at: Optional[str] = None

        # Feature importance
        self.feature_importance_: Optional[np.ndarray] = None

        # ── Confidence-aware routing ──────────────────────────────────────────
        self.enable_specialist = enable_specialist
        self.confidence_estimator = ConfidenceEstimator(
            config=ConfidenceEstimatorConfig(threshold=confidence_threshold)
        )
        self.ft_transformer = None  # Lazy-loaded FTTransformerPredictor
        self.meta_fusion = (
            MetaFusionLayer(method=fusion_method) if enable_specialist else None
        )

    # ── Hash computation ────────────────────────────────────────────────────────

    def _compute_hashes(self, X: np.ndarray, y: np.ndarray) -> dict[str, str]:
        """Compute deterministic hashes for version pinning."""
        # Feature hash
        feature_str = (
            "|".join(self.feature_names) if self.feature_names else str(X.shape[1])
        )
        feature_hash = hashlib.sha256(feature_str.encode()).hexdigest()[:16]

        # Dataset hash (sample statistics)
        if len(X) > 10000:
            idx = np.random.choice(len(X), 10000, replace=False)
            X_sample = X[idx]
            y_sample = y[idx]
        else:
            X_sample = X
            y_sample = y
        data_stats = np.concatenate(
            [
                X_sample.mean(axis=0),
                X_sample.std(axis=0),
                [y_sample.mean(), len(y_sample)],
            ]
        )
        dataset_hash = hashlib.sha256(data_stats.tobytes()).hexdigest()[:16]

        # Training hash (hyperparameters)
        train_config = {
            "iterations": self.iterations,
            "depth": self.depth,
            "learning_rate": self.learning_rate,
            **self.catboost_kwargs,
        }
        training_hash = hashlib.sha256(
            json.dumps(train_config, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        return {
            "feature_hash": feature_hash,
            "dataset_hash": dataset_hash,
            "training_hash": training_hash,
            "model_version": f"v1_cb_{training_hash[:8]}",
        }

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        categorical_indices: Optional[List[int]] = None,
        cat_feature_names: Optional[List[str]] = None,
        calibration_method: str = "isotonic",
        calibrate: bool = True,
        **catboost_fit_kwargs,
    ) -> "ChampionModel":
        """
        Train the CatBoost champion model.

        Args:
            X: Feature matrix
            y: Labels (0=legit, 1=fraud)
            feature_names: List of feature names
            categorical_indices: Indices of categorical features
            cat_feature_names: Names of categorical features (for CatBoost Pool)
            calibration_method: "isotonic" or "platt"
            calibrate: Whether to calibrate probabilities
            **catboost_fit_kwargs: Additional CatBoost fit parameters
        """
        logger.info(
            "ChampionModel.fit: {} samples, {} features, {:.3%} fraud rate",
            *X.shape,
            y.mean(),
        )

        self.feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]
        self.categorical_features = categorical_indices or []
        self.cat_feature_names = cat_feature_names or []
        self.calibration_method = calibration_method

        # Compute version hashes
        hashes = self._compute_hashes(X, y)
        self.feature_hash = hashes["feature_hash"]
        self.dataset_hash = hashes["dataset_hash"]
        self.training_hash = hashes["training_hash"]
        self.model_version = hashes["model_version"]
        self.trained_at = datetime.now(timezone.utc).isoformat()

        # Create CatBoost Pool for efficient training
        pool = Pool(
            data=X,
            label=y,
            cat_features=self.categorical_features,
            feature_names=self.feature_names,
        )

        # Initialize CatBoost with fraud-optimized defaults
        self.model = CatBoostClassifier(
            iterations=1000,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            bootstrap_type="Bernoulli",
            subsample=0.8,
            random_seed=42,
            verbose=100,
            early_stopping_rounds=50,
            **self.catboost_kwargs,
        )

        # Train with early stopping using validation split
        from sklearn.model_selection import train_test_split

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        train_pool = Pool(
            data=X_train,
            label=y_train,
            cat_features=self.categorical_features,
            feature_names=self.feature_names,
        )
        val_pool = Pool(
            data=X_val,
            label=y_val,
            cat_features=self.categorical_features,
            feature_names=self.feature_names,
        )

        logger.info("Training CatBoost champion model...")
        self.model.fit(
            train_pool, eval_set=val_pool, verbose=100, **catboost_fit_kwargs
        )

        # Get best iteration
        best_iter = self.model.get_best_iteration()
        logger.info("Best iteration: {}", best_iter)

        # ── Calibration ────────────────────────────────────────────────────────
        if calibrate:
            logger.info("Calibrating probabilities with {}...", calibration_method)
            self.calibrator = ProbabilityCalibrator(method=calibration_method)

            # Get raw probabilities on validation set
            val_probs = self.model.predict_proba(X_val)[:, 1]
            self.calibrator.fit(val_probs, y_val)
            logger.info("Calibration complete")

        # ── Metrics ─────────────────────────────────────────────────────────────
        val_probs = self.predict_proba(X_val)
        self._compute_metrics(y_val, val_probs)

        # Feature importance
        self.feature_importance_ = self.model.get_feature_importance()

        self.is_fitted = True
        logger.info(
            "ChampionModel trained — PR-AUC: {:.4f}, ROC-AUC: {:.4f}, F2: {:.4f}",
            self.pr_auc_,
            self.roc_auc_,
            self.f2_score_,
        )
        return self

    def _compute_metrics(self, y_true: np.ndarray, y_probs: np.ndarray) -> None:
        from sklearn.metrics import (
            average_precision_score,
            roc_auc_score,
            precision_score,
            recall_score,
            fbeta_score,
        )

        self.pr_auc_ = float(average_precision_score(y_true, y_probs))
        self.roc_auc_ = float(roc_auc_score(y_true, y_probs))

        # F2 score at 0.5 threshold
        y_pred = (y_probs >= 0.5).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        self.f2_score_ = float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))

        logger.info(
            "Validation metrics — PR-AUC: {:.4f}, ROC-AUC: {:.4f}, F2: {:.4f}",
            self.pr_auc_,
            self.roc_auc_,
            self.f2_score_,
        )

    # ── Specialist Setup ────────────────────────────────────────────────────────

    def setup_specialist(
        self,
        ft_transformer: Any,
        meta_fusion: Optional[MetaFusionLayer] = None,
        confidence_threshold: float = 0.92,
    ) -> None:
        """
        Configure the FT-Transformer specialist and meta-fusion layer.

        Args:
            ft_transformer: Trained FTTransformerPredictor instance
            meta_fusion: Trained MetaFusionLayer (optional)
            confidence_threshold: Threshold below which FT is consulted
        """
        self.ft_transformer = ft_transformer
        if meta_fusion is not None:
            self.meta_fusion = meta_fusion
        self.enable_specialist = True
        self.confidence_estimator.set_threshold(confidence_threshold)
        logger.info(
            "Specialist configured: confidence_threshold={:.3f}",
            confidence_threshold,
        )

    def fit_specialist(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        ft_transformer: Any,
    ) -> None:
        """
        Train the meta-fusion layer on a calibration set where both
        CatBoost and FT-Transformer predictions are available.

        Args:
            X_cal: Calibration features
            y_cal: Calibration labels
            ft_transformer: Trained FTTransformerPredictor
        """
        cat_probs = self.predict_proba(X_cal)
        ft_probs = ft_transformer.predict_proba(X_cal)
        confidences = np.array(
            [self.confidence_estimator.estimate(p) for p in cat_probs]
        )

        self.meta_fusion = MetaFusionLayer()
        self.meta_fusion.fit(
            catboost_probs=cat_probs,
            ft_probs=ft_probs,
            catboost_confidences=confidences,
            y_true=y_cal,
        )
        self.ft_transformer = ft_transformer
        self.enable_specialist = True

        # Fit conformal on calibration set
        self.confidence_estimator.fit_conformal(cat_probs, y_cal)

        logger.info("Specialist trained and configured")

    # ── Scoring ────────────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated fraud probabilities."""
        if not self.is_fitted and self.model is None:
            raise RuntimeError("ChampionModel must be fitted before scoring")

        raw_probs = self.model.predict_proba(X)[:, 1]

        if self.calibrator:
            return self.calibrator.transform(raw_probs)
        return raw_probs

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated fraud probabilities (alias for predict_proba)."""
        return self.predict_proba(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions at given threshold."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def score_with_confidence(self, X: np.ndarray) -> List[SupervisedPrediction]:
        """
        Score with confidence-aware routing.

        For each transaction:
        1. Get CatBoost calibrated probability
        2. Estimate confidence
        3. If confident → return directly
        4. If not confident → consult FT-Transformer → meta-fuse

        Returns list of strongly typed SupervisedPrediction objects.
        """
        if not self.is_fitted and self.model is None:
            raise RuntimeError("ChampionModel must be fitted before scoring")

        t_start = time.perf_counter()

        cat_probs = self.predict_proba(X)
        results: List[SupervisedPrediction] = []

        # Batch collect low-confidence samples for FT-Transformer
        low_conf_indices: List[int] = []
        low_conf_cat_probs: List[float] = []

        for i in range(len(X)):
            prob = float(cat_probs[i])
            confidence = self.confidence_estimator.estimate(prob)

            if self.enable_specialist and not self.confidence_estimator.is_confident(
                prob
            ):
                low_conf_indices.append(i)
                low_conf_cat_probs.append(prob)

        # Batch FT-Transformer prediction for efficiency
        ft_probs_map: Dict[int, float] = {}
        if low_conf_indices and self.ft_transformer is not None:
            X_low = X[low_conf_indices]
            ft_probs = self.ft_transformer.predict_proba(X_low)
            for j, idx in enumerate(low_conf_indices):
                ft_probs_map[idx] = float(ft_probs[j])

        # Assemble final predictions
        for i in range(len(X)):
            prob = float(cat_probs[i])
            confidence = self.confidence_estimator.estimate(prob)
            ft_invoked = i in ft_probs_map
            fusion_output = None

            if ft_invoked:
                ft_prob = ft_probs_map[i]
                if self.meta_fusion is not None and self.meta_fusion.is_fitted:
                    fusion_output = self.meta_fusion.predict(
                        catboost_prob=prob,
                        ft_prob=ft_prob,
                        catboost_confidence=confidence,
                    )
                else:
                    # Fallback: weighted average
                    fusion_output = 0.7 * prob + 0.3 * ft_prob
                prob = fusion_output

            latency_ms = (time.perf_counter() - t_start) * 1000

            results.append(
                SupervisedPrediction(
                    probability=float(np.clip(prob, 0.0, 1.0)),
                    confidence=float(np.clip(confidence, 0.0, 1.0)),
                    ft_invoked=ft_invoked,
                    fusion_output=fusion_output,
                    latency_ms=latency_ms,
                    model_version=self.model_version,
                    catboost_version=self.model_version,
                    ft_transformer_version=(
                        self.ft_transformer.model_version
                        if self.ft_transformer is not None
                        else ""
                    ),
                )
            )

        return results

    # ── Feature Importance ─────────────────────────────────────────────────────

    def get_feature_importance(self, top_n: int = 20) -> List[dict]:
        """Get top-N feature importances."""
        if self.feature_importance_ is None:
            return []

        importance_dict = dict(zip(self.feature_names, self.feature_importance_))
        sorted_features = sorted(
            importance_dict.items(), key=lambda x: x[1], reverse=True
        )

        return [
            {"feature": feat, "importance": float(imp)}
            for feat, imp in sorted_features[:top_n]
        ]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save CatBoost model in native format
        self.model.save_model(str(path / "champion_model.cbm"))

        # Save metadata and calibrator
        with open(path / "champion_metadata.pkl", "wb") as f:
            pickle.dump(
                {
                    "feature_names": self.feature_names,
                    "categorical_features": self.categorical_features,
                    "cat_feature_names": self.cat_feature_names,
                    "calibration_method": self.calibration_method,
                    "pr_auc": self.pr_auc_,
                    "f2_score": self.f2_score_,
                    "roc_auc": self.roc_auc_,
                    "is_fitted": self.is_fitted,
                    "model_version": self.model_version,
                    "training_hash": self.training_hash,
                    "feature_hash": self.feature_hash,
                    "dataset_hash": self.dataset_hash,
                    "trained_at": self.trained_at,
                    "iterations": self.iterations,
                    "depth": self.depth,
                    "learning_rate": self.learning_rate,
                    "catboost_kwargs": self.catboost_kwargs,
                    "calibrator": self.calibrator,
                    # Specialist config
                    "enable_specialist": self.enable_specialist,
                    "confidence_threshold": self.confidence_estimator.config.threshold,
                    "fusion_method": (
                        self.meta_fusion.method
                        if self.meta_fusion
                        else "logistic_regression"
                    ),
                },
                f,
            )

        # Save feature importance
        if self.feature_importance_ is not None:
            importance_df = pd.DataFrame(
                {"feature": self.feature_names, "importance": self.feature_importance_}
            )
            importance_df.to_csv(path / "feature_importance.csv", index=False)

        # Save specialist components
        if self.enable_specialist and self.meta_fusion is not None:
            self.meta_fusion.save(path / "meta_fusion")

        logger.info(
            "ChampionModel saved to {} (specialist={})", path, self.enable_specialist
        )

    @classmethod
    def load(cls, path: Path) -> "ChampionModel":
        path = Path(path)

        # Load metadata first
        with open(path / "champion_metadata.pkl", "rb") as f:
            payload = pickle.load(f)

        # Create instance
        obj = cls(
            feature_names=payload["feature_names"],
            categorical_features=payload["categorical_features"],
            cat_feature_names=payload.get("cat_feature_names"),
            iterations=payload.get("iterations", 1000),
            depth=payload.get("depth", 6),
            learning_rate=payload.get("learning_rate", 0.05),
            enable_specialist=payload.get("enable_specialist", False),
            confidence_threshold=payload.get("confidence_threshold", 0.92),
            fusion_method=payload.get("fusion_method", "logistic_regression"),
        )

        # Restore attributes
        for k, v in payload.items():
            if k not in (
                "calibrator",
                "enable_specialist",
                "confidence_threshold",
                "fusion_method",
            ):
                setattr(obj, k, v)

        obj.calibrator = payload.get("calibrator")

        # Load CatBoost model
        obj.model = CatBoostClassifier()
        obj.model.load_model(str(path / "champion_model.cbm"))

        obj.is_fitted = True

        # Load specialist components
        meta_fusion_path = path / "meta_fusion"
        if meta_fusion_path.exists():
            obj.meta_fusion = MetaFusionLayer.load(meta_fusion_path)
            logger.info("Meta-fusion loaded from {}", meta_fusion_path)

        logger.info(
            "ChampionModel loaded from {} (version={}, specialist={})",
            path,
            obj.model_version,
            obj.enable_specialist,
        )
        return obj


# ── Convenience function for quick training ──────────────────────────────────


def train_champion(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    categorical_indices: Optional[List[int]] = None,
    cat_feature_names: Optional[List[str]] = None,
    calibration_method: str = "isotonic",
    **kwargs,
) -> ChampionModel:
    """Train a ChampionModel with sensible defaults."""
    model = ChampionModel()
    return model.fit(
        X,
        y,
        feature_names=feature_names,
        categorical_indices=categorical_indices,
        cat_feature_names=cat_feature_names,
        calibration_method=calibration_method,
        **kwargs,
    )
