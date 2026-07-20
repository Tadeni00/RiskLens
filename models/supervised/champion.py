"""
FraudTrap — Champion Model (CatBoost)
Single-model production fraud detector with native categorical handling.
Replaces the full stacking ensemble for production inference.
"""
from __future__ import annotations
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from loguru import logger

from scoring.calibration import ProbabilityCalibrator


class ChampionModel:
    """
    Production CatBoost fraud detector.
    
    Key features:
    - Native categorical feature handling (no target encoding leakage)
    - Built-in class imbalance handling
    - Fast inference with GPU support
    - Built-in feature importance
    - Probability calibration support
    - Version pinning for reproducibility
    """
    
    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[int]] = None,
        cat_feature_names: Optional[List[str]] = None,
        iterations: int = 500,
        depth: int = 6,
        learning_rate: float = 0.05,
        **catboost_kwargs
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

    # ── Hash computation ────────────────────────────────────────────────────────
    
    def _compute_hashes(self, X: np.ndarray, y: np.ndarray) -> dict[str, str]:
        """Compute deterministic hashes for version pinning."""
        # Feature hash
        feature_str = "|".join(self.feature_names) if self.feature_names else str(X.shape[1])
        feature_hash = hashlib.sha256(feature_str.encode()).hexdigest()[:16]
        
        # Dataset hash (sample statistics)
        if len(X) > 10000:
            idx = np.random.choice(len(X), 10000, replace=False)
            X_sample = X[idx]
            y_sample = y[idx]
        else:
            X_sample = X
            y_sample = y
        data_stats = np.concatenate([
            X_sample.mean(axis=0), 
            X_sample.std(axis=0), 
            [y_sample.mean(), len(y_sample)]
        ])
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
        **catboost_fit_kwargs
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
            *X.shape, y.mean()
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
            feature_names=self.feature_names
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
            **self.catboost_kwargs
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
            feature_names=self.feature_names
        )
        val_pool = Pool(
            data=X_val,
            label=y_val,
            cat_features=self.categorical_features,
            feature_names=self.feature_names
        )
        
        logger.info("Training CatBoost champion model...")
        self.model.fit(
            train_pool,
            eval_set=val_pool,
            verbose=100,
            **catboost_fit_kwargs
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
            self.pr_auc_, self.roc_auc_, self.f2_score_
        )
        return self
    
    def _compute_metrics(self, y_true: np.ndarray, y_probs: np.ndarray) -> None:
        from sklearn.metrics import (
            average_precision_score, roc_auc_score, precision_score, 
            recall_score, fbeta_score
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
            self.pr_auc_, self.roc_auc_, self.f2_score_
        )

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

    # ── Feature Importance ─────────────────────────────────────────────────────
    
    def get_feature_importance(self, top_n: int = 20) -> List[dict]:
        """Get top-N feature importances."""
        if self.feature_importance_ is None:
            return []
        
        importance_dict = dict(zip(self.feature_names, self.feature_importance_))
        sorted_features = sorted(
            importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
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
            pickle.dump({
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
            }, f)
        
        # Save feature importance
        if self.feature_importance_ is not None:
            importance_df = pd.DataFrame({
                "feature": self.feature_names,
                "importance": self.feature_importance_
            })
            importance_df.to_csv(path / "feature_importance.csv", index=False)
        
        logger.info("ChampionModel saved to {}", path)

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
        )
        
        # Restore attributes
        for k, v in payload.items():
            if k not in ("calibrator",):
                setattr(obj, k, v)
        
        obj.calibrator = payload.get("calibrator")
        
        # Load CatBoost model
        obj.model = CatBoostClassifier()
        obj.model.load_model(str(path / "champion_model.cbm"))
        
        obj.is_fitted = True
        logger.info("ChampionModel loaded from {} (version={})", path, obj.model_version)
        return obj


# ── Convenience function for quick training ──────────────────────────────────

def train_champion(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    categorical_indices: Optional[List[int]] = None,
    cat_feature_names: Optional[List[str]] = None,
    calibration_method: str = "isotonic",
    **kwargs
) -> ChampionModel:
    """Train a ChampionModel with sensible defaults."""
    model = ChampionModel()
    return model.fit(
        X, y,
        feature_names=feature_names,
        categorical_indices=categorical_indices,
        cat_feature_names=cat_feature_names,
        calibration_method=calibration_method,
        **kwargs
    )