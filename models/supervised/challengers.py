"""
FraudTrap — Challenger Framework
Offline-only challenger models for Champion-Challenger architecture.
Supports XGBoost, LightGBM, FT-Transformer, and TabNet.
"""
from __future__ import annotations
import hashlib
import json
import pickle
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
from loguru import logger

from scoring.calibration import ProbabilityCalibrator


class BaseChallenger(ABC):
    """
    Abstract base class for challenger models.
    
    Challengers are trained offline and evaluated continuously.
    They are NEVER used in production inference.
    """
    
    def __init__(
        self,
        algorithm: str,
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[int]] = None,
        calibration_method: str = "isotonic",
    ):
        self.algorithm = algorithm
        self.feature_names = feature_names or []
        self.categorical_features = categorical_features or []
        self.calibration_method = calibration_method
        
        self.model = None
        self.calibrator: Optional[ProbabilityCalibrator] = None
        self.is_fitted: bool = False
        
        # Metrics
        self.pr_auc_: float = 0.0
        self.roc_auc_: float = 0.0
        self.f2_score_: float = 0.0
        self.fpr_: float = 0.0
        self.calibration_error_: float = 0.0
        
        # Version pinning
        self.model_version: str = "1.0.0"
        self.training_hash: Optional[str] = None
        self.feature_hash: Optional[str] = None
        self.dataset_hash: Optional[str] = None
        self.trained_at: Optional[str] = None
    
    @abstractmethod
    def _build_model(self, **kwargs):
        """Build the underlying model."""
        pass
    
    @abstractmethod
    def _fit_model(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit the underlying model."""
        pass
    
    @abstractmethod
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get raw probabilities from the model."""
        pass
    
    def _compute_hashes(self, X: np.ndarray, y: np.ndarray) -> dict[str, str]:
        """Compute deterministic hashes for version pinning."""
        feature_str = "|".join(self.feature_names) if self.feature_names else str(X.shape[1])
        feature_hash = hashlib.sha256(feature_str.encode()).hexdigest()[:16]
        
        if len(X) > 10000:
            idx = np.random.choice(len(X), 10000, replace=False)
            X_sample, y_sample = X[idx], y[idx]
        else:
            X_sample, y_sample = X, y
        
        data_stats = np.concatenate([
            X_sample.mean(axis=0),
            X_sample.std(axis=0),
            [y_sample.mean(), len(y_sample)]
        ])
        dataset_hash = hashlib.sha256(data_stats.tobytes()).hexdigest()[:16]
        
        return {
            "feature_hash": feature_hash,
            "dataset_hash": dataset_hash,
        }
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        categorical_indices: Optional[List[int]] = None,
        calibrate: bool = True,
        **kwargs
    ) -> "BaseChallenger":
        """
        Train the challenger model.
        
        Args:
            X: Feature matrix
            y: Labels (0=legit, 1=fraud)
            feature_names: List of feature names
            categorical_indices: Indices of categorical features
            calibrate: Whether to calibrate probabilities
            **kwargs: Additional training parameters
        """
        logger.info(
            "Training {} challenger: {} samples, {} features, {:.3%} fraud rate",
            self.algorithm, *X.shape, y.mean()
        )
        
        self.feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]
        self.categorical_features = categorical_indices or []
        
        # Compute hashes
        hashes = self._compute_hashes(X, y)
        self.feature_hash = hashes["feature_hash"]
        self.dataset_hash = hashes["dataset_hash"]
        self.training_hash = hashlib.sha256(
            json.dumps(kwargs, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        self.model_version = f"v1_{self.algorithm}_{self.training_hash[:8]}"
        self.trained_at = datetime.now(timezone.utc).isoformat()
        
        # Build and fit model
        self.model = self._build_model(**kwargs)
        self._fit_model(X, y, **kwargs)
        
        # Calibration
        if calibrate:
            self._calibrate(X, y)
        
        self.is_fitted = True
        logger.info("{} challenger trained — PR-AUC: {:.4f}", self.algorithm, self.pr_auc_)
        return self
    
    def _calibrate(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit probability calibrator on validation set."""
        from sklearn.model_selection import train_test_split
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        raw_probs = self._predict_proba(X_val)
        
        self.calibrator = ProbabilityCalibrator(method=self.calibration_method)
        self.calibrator.fit(raw_probs, y_val)
        
        # Compute calibration error
        from sklearn.calibration import calibration_curve
        cal_probs = self.calibrator.transform(raw_probs)
        fraction_pos, mean_predicted = calibration_curve(y_val, cal_probs, n_bins=10)
        self.calibration_error_ = float(np.mean(np.abs(fraction_pos - mean_predicted)))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated fraud probabilities."""
        if not self.is_fitted:
            raise RuntimeError(f"{self.algorithm} challenger must be fitted before scoring")
        
        raw_probs = self._predict_proba(X)
        
        if self.calibrator:
            return self.calibrator.transform(raw_probs)
        return raw_probs
    
    def score(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated fraud probabilities (alias for predict_proba)."""
        return self.predict_proba(X)
    
    def compute_metrics(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        fpr_threshold: float = 0.01,
    ) -> dict:
        """
        Compute evaluation metrics.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            fpr_threshold: FPR threshold for computing FPR@threshold
        
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import (
            average_precision_score, roc_auc_score, precision_score,
            recall_score, fbeta_score, confusion_matrix
        )
        
        probs = self.predict_proba(X_val)
        
        self.pr_auc_ = float(average_precision_score(y_val, probs))
        self.roc_auc_ = float(roc_auc_score(y_val, probs))
        
        # F2 score at 0.5 threshold
        y_pred = (probs >= 0.5).astype(int)
        self.f2_score_ = float(fbeta_score(y_val, y_pred, beta=2, zero_division=0))
        
        # FPR at threshold
        tn, fp, fn, tp = confusion_matrix(y_val, (probs >= fpr_threshold).astype(int)).ravel()
        self.fpr_ = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        
        logger.info(
            "{} metrics — PR-AUC: {:.4f}, ROC-AUC: {:.4f}, F2: {:.4f}, FPR: {:.4f}",
            self.algorithm, self.pr_auc_, self.roc_auc_, self.f2_score_, self.fpr_
        )
        
        return {
            "pr_auc": self.pr_auc_,
            "roc_auc": self.roc_auc_,
            "f2_score": self.f2_score_,
            "fpr": self.fpr_,
            "calibration_error": self.calibration_error_,
        }
    
    def get_feature_importance(self, top_n: int = 20) -> List[dict]:
        """Get top-N feature importances."""
        if not hasattr(self.model, "feature_importances_"):
            return []
        
        importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
        sorted_features = sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {"feature": feat, "importance": float(imp)}
            for feat, imp in sorted_features[:top_n]
        ]
    
    def save(self, path: Path) -> None:
        """Save challenger model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        with open(path / "model.pkl", "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_names": self.feature_names,
                "categorical_features": self.categorical_features,
                "algorithm": self.algorithm,
                "pr_auc": self.pr_auc_,
                "roc_auc": self.roc_auc_,
                "f2_score": self.f2_score_,
                "fpr": self.fpr_,
                "calibration_error": self.calibration_error_,
                "model_version": self.model_version,
                "training_hash": self.training_hash,
                "feature_hash": self.feature_hash,
                "dataset_hash": self.dataset_hash,
                "trained_at": self.trained_at,
            }, f)
        
        # Save calibrator
        if self.calibrator:
            self.calibrator.save(path / "calibrator")
        
        logger.info("{} challenger saved to {}", self.algorithm, path)
    
    @classmethod
    def load(cls, path: Path) -> "BaseChallenger":
        """Load challenger model from disk."""
        path = Path(path)
        
        with open(path / "model.pkl", "rb") as f:
            payload = pickle.load(f)
        
        obj = cls(
            feature_names=payload["feature_names"],
            categorical_features=payload.get("categorical_features", []),
        )
        
        obj.model = payload["model"]
        obj.algorithm = payload["algorithm"]
        obj.pr_auc_ = payload["pr_auc"]
        obj.roc_auc_ = payload["roc_auc"]
        obj.f2_score_ = payload["f2_score"]
        obj.fpr_ = payload["fpr"]
        obj.calibration_error_ = payload["calibration_error"]
        obj.model_version = payload["model_version"]
        obj.training_hash = payload["training_hash"]
        obj.feature_hash = payload["feature_hash"]
        obj.dataset_hash = payload["dataset_hash"]
        obj.trained_at = payload["trained_at"]
        obj.is_fitted = True
        
        # Load calibrator
        calibrator_path = path / "calibrator"
        if calibrator_path.exists():
            obj.calibrator = ProbabilityCalibrator.load(calibrator_path)
        
        logger.info("{} challenger loaded from {} (version={})", obj.algorithm, path, obj.model_version)
        return obj


# ═══════════════════════════════════════════════════════════════════════════════
# XGBoost Challenger
# ═══════════════════════════════════════════════════════════════════════════════

class XGBoostChallenger(BaseChallenger):
    """XGBoost challenger model."""
    
    def __init__(self, **kwargs):
        super().__init__(algorithm="xgboost", **kwargs)
    
    def _build_model(self, **kwargs):
        from xgboost import XGBClassifier
        
        fraud_rate = kwargs.get("fraud_rate", 0.1)
        scale_pos_weight = (1 - fraud_rate) / max(fraud_rate, 1e-6)
        
        return XGBClassifier(
            n_estimators=kwargs.get("n_estimators", 500),
            max_depth=kwargs.get("max_depth", 6),
            learning_rate=kwargs.get("learning_rate", 0.05),
            subsample=kwargs.get("subsample", 0.8),
            colsample_bytree=kwargs.get("colsample_bytree", 0.8),
            min_child_weight=kwargs.get("min_child_weight", 3),
            reg_alpha=kwargs.get("reg_alpha", 0.1),
            reg_lambda=kwargs.get("reg_lambda", 1.0),
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1,
        )
    
    def _fit_model(self, X: np.ndarray, y: np.ndarray, **kwargs):
        self.model.fit(X, y)
    
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


# ═══════════════════════════════════════════════════════════════════════════════
# LightGBM Challenger
# ═══════════════════════════════════════════════════════════════════════════════

class LightGBMChallenger(BaseChallenger):
    """LightGBM challenger model."""
    
    def __init__(self, **kwargs):
        super().__init__(algorithm="lightgbm", **kwargs)
    
    def _build_model(self, **kwargs):
        from lightgbm import LGBMClassifier
        
        return LGBMClassifier(
            n_estimators=kwargs.get("n_estimators", 500),
            max_depth=kwargs.get("max_depth", 6),
            learning_rate=kwargs.get("learning_rate", 0.05),
            num_leaves=kwargs.get("num_leaves", 63),
            subsample=kwargs.get("subsample", 0.8),
            colsample_bytree=kwargs.get("colsample_bytree", 0.8),
            is_unbalance=True,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    
    def _fit_model(self, X: np.ndarray, y: np.ndarray, **kwargs):
        self.model.fit(X, y)
    
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


# ═══════════════════════════════════════════════════════════════════════════════
# FT-Transformer Challenger (Tabular Transformer)
# ═══════════════════════════════════════════════════════════════════════════════

class FTTransformerChallenger(BaseChallenger):
    """
    FT-Transformer challenger model.
    
    A simple tabular transformer implementation for fraud detection.
    Uses feature tokenization and self-attention.
    """
    
    def __init__(self, **kwargs):
        super().__init__(algorithm="ft_transformer", **kwargs)
        self._device = kwargs.get("device", "cpu")
        self._n_epochs = kwargs.get("n_epochs", 100)
        self._batch_size = kwargs.get("batch_size", 256)
        self._learning_rate = kwargs.get("learning_rate", 1e-3)
        self._d_token = kwargs.get("d_token", 64)
        self._n_heads = kwargs.get("n_heads", 4)
        self._n_layers = kwargs.get("n_layers", 2)
        self._dropout = kwargs.get("dropout", 0.1)
        self._scaler = None
        self._encoder = None
    
    def _build_model(self, **kwargs):
        # Use sklearn as fallback if torch not available
        try:
            import torch
            import torch.nn as nn
            
            self._encoder = self._FTTransformerEncoder(
                n_features=len(self.feature_names),
                d_token=self._d_token,
                n_heads=self._n_heads,
                n_layers=self._n_layers,
                dropout=self._dropout,
            )
            return self._encoder
        except ImportError:
            # Fallback to sklearn model
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                random_state=42,
            )
    
    def _fit_model(self, X: np.ndarray, y: np.ndarray, **kwargs):
        from sklearn.preprocessing import StandardScaler
        
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X_scaled).to(self._device)
            y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self._device)
            
            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)
            
            # Training loop
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self._learning_rate)
            criterion = nn.BCELoss()
            
            self.model.train()
            for epoch in range(self._n_epochs):
                total_loss = 0
                for batch_X, batch_y in loader:
                    optimizer.zero_grad()
                    output = self.model(batch_X)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                if (epoch + 1) % 10 == 0:
                    logger.debug("Epoch {}/{}: loss={:.4f}", epoch + 1, self._n_epochs, total_loss / len(loader))
        except ImportError:
            # Fallback to sklearn fitting
            self.model.fit(X_scaled, y)
    
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        
        try:
            import torch
            
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled).to(self._device)
                output = self.model(X_tensor).cpu().numpy().ravel()
                return output
        except ImportError:
            return self.model.predict_proba(X_scaled)[:, 1]
    
    class _FTTransformerEncoder:
        """Simple FT-Transformer encoder (placeholder when torch unavailable)."""
        
        def __init__(self, n_features, d_token=64, n_heads=4, n_layers=2, dropout=0.1):
            self.n_features = n_features
            self.d_token = d_token
            
            try:
                import torch
                import torch.nn as nn
                
                # Feature embeddings
                self.feature_embeddings = nn.Linear(1, d_token)
                self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
                
                # Transformer encoder
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_token,
                    nhead=n_heads,
                    dim_feedforward=d_token * 4,
                    dropout=dropout,
                    batch_first=True,
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
                
                # Classification head
                self.head = nn.Sequential(
                    nn.Linear(d_token, d_token),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_token, 1),
                    nn.Sigmoid(),
                )
                
                self._use_torch = True
            except ImportError:
                self._use_torch = False
                # Fallback to sklearn
                from sklearn.ensemble import GradientBoostingClassifier
                self.head = GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    random_state=42,
                )
        
        def forward(self, x):
            if not self._use_torch:
                return self.head.predict_proba(x)[:, 1]
            
            import torch
            
            batch_size = x.shape[0]
            
            # Embed each feature separately
            x = x.unsqueeze(-1)  # (batch, n_features, 1)
            x = self.feature_embeddings(x)  # (batch, n_features, d_token)
            
            # Prepend CLS token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            
            # Transformer
            x = self.transformer(x)
            
            # Use CLS token for classification
            cls_output = x[:, 0, :]
            return self.head(cls_output)
    
    def save(self, path: Path) -> None:
        """Override save to handle torch model."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model (torch or sklearn)
        try:
            import torch
            torch.save({
                "model_state": self.model.state_dict() if hasattr(self.model, 'state_dict') else None,
                "feature_names": self.feature_names,
                "algorithm": self.algorithm,
                "metrics": {
                    "pr_auc": self.pr_auc_,
                    "roc_auc": self.roc_auc_,
                    "f2_score": self.f2_score_,
                    "fpr": self.fpr_,
                    "calibration_error": self.calibration_error_,
                },
                "model_version": self.model_version,
                "training_hash": self.training_hash,
                "feature_hash": self.feature_hash,
                "dataset_hash": self.dataset_hash,
                "trained_at": self.trained_at,
                "scaler": self._scaler,
                "config": {
                    "d_token": self._d_token,
                    "n_heads": self._n_heads,
                    "n_layers": self._n_layers,
                    "dropout": self._dropout,
                },
            }, path / "model.pt")
        except ImportError:
            # Fallback to pickle for sklearn
            super().save(path)
    
    @classmethod
    def load(cls, path: Path) -> "FTTransformerChallenger":
        """Override load to handle torch model."""
        path = Path(path)
        
        try:
            import torch
            
            checkpoint = torch.load(path / "model.pt", map_location="cpu")
            
            obj = cls(
                feature_names=checkpoint["feature_names"],
                d_token=checkpoint["config"]["d_token"],
                n_heads=checkpoint["config"]["n_heads"],
                n_layers=checkpoint["config"]["n_layers"],
                dropout=checkpoint["config"]["dropout"],
            )
            
            obj.model = obj._build_model()
            if checkpoint["model_state"]:
                obj.model.load_state_dict(checkpoint["model_state"])
            
            obj._scaler = checkpoint["scaler"]
            obj.algorithm = checkpoint["algorithm"]
            obj.pr_auc_ = checkpoint["metrics"]["pr_auc"]
            obj.roc_auc_ = checkpoint["metrics"]["roc_auc"]
            obj.f2_score_ = checkpoint["metrics"]["f2_score"]
            obj.fpr_ = checkpoint["metrics"]["fpr"]
            obj.calibration_error_ = checkpoint["metrics"]["calibration_error"]
            obj.model_version = checkpoint["model_version"]
            obj.training_hash = checkpoint["training_hash"]
            obj.feature_hash = checkpoint["feature_hash"]
            obj.dataset_hash = checkpoint["dataset_hash"]
            obj.trained_at = checkpoint["trained_at"]
            obj.is_fitted = True
            
            # Load calibrator
            calibrator_path = path / "calibrator"
            if calibrator_path.exists():
                obj.calibrator = ProbabilityCalibrator.load(calibrator_path)
            
            return obj
        except ImportError:
            return super().load(path)


# ═══════════════════════════════════════════════════════════════════════════════
# TabNet Challenger
# ═══════════════════════════════════════════════════════════════════════════════

class TabNetChallenger(BaseChallenger):
    """
    TabNet challenger model.
    
    Uses sparse attention for feature selection.
    """
    
    def __init__(self, **kwargs):
        super().__init__(algorithm="tabnet", **kwargs)
        self._n_d = kwargs.get("n_d", 64)
        self._n_a = kwargs.get("n_a", 64)
        self._n_steps = kwargs.get("n_steps", 3)
        self._gamma = kwargs.get("gamma", 1.3)
        self._n_independent = kwargs.get("n_independent", 2)
        self._n_shared = kwargs.get("n_shared", 2)
        self._scaler = None
    
    def _build_model(self, **kwargs):
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
            
            return TabNetClassifier(
                n_d=self._n_d,
                n_a=self._n_a,
                n_steps=self._n_steps,
                gamma=self._gamma,
                n_independent=self._n_independent,
                n_shared=self._n_shared,
                optimizer_params=dict(lr=2e-2),
                scheduler_params={"step_size": 50, "gamma": 0.9},
                scheduler_fn="StepLR",
                mask_type="sparsemax",
                verbose=0,
                seed=42,
            )
        except ImportError:
            # Fallback to sklearn
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                random_state=42,
            )
    
    def _fit_model(self, X: np.ndarray, y: np.ndarray, **kwargs):
        from sklearn.preprocessing import StandardScaler
        
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
            eval_set = kwargs.get("eval_set")
            if eval_set:
                X_val, y_val = eval_set
                X_val_scaled = self._scaler.transform(X_val)
                self.model.fit(
                    X_scaled, y,
                    eval_set=[(X_val_scaled, y_val)],
                    eval_metric=["auc"],
                    max_epochs=kwargs.get("max_epochs", 100),
                    patience=kwargs.get("patience", 10),
                    batch_size=kwargs.get("batch_size", 256),
                    virtual_batch_size=kwargs.get("virtual_batch_size", 128),
                )
            else:
                self.model.fit(
                    X_scaled, y,
                    max_epochs=kwargs.get("max_epochs", 100),
                    patience=kwargs.get("patience", 10),
                    batch_size=kwargs.get("batch_size", 256),
                    virtual_batch_size=kwargs.get("virtual_batch_size", 128),
                )
        except ImportError:
            self.model.fit(X_scaled, y)
    
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def get_feature_importance(self, top_n: int = 20) -> List[dict]:
        """Get TabNet feature importance (attention masks)."""
        try:
            importance = self.model.feature_importances_
            importance_dict = dict(zip(self.feature_names, importance))
            sorted_features = sorted(
                importance_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return [
                {"feature": feat, "importance": float(imp)}
                for feat, imp in sorted_features[:top_n]
            ]
        except Exception:
            return []


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════

def create_challenger(algorithm: str, **kwargs) -> BaseChallenger:
    """
    Factory function to create a challenger model.
    
    Args:
        algorithm: Algorithm name (xgboost, lightgbm, ft_transformer, tabnet)
        **kwargs: Additional parameters
    
    Returns:
        Challenger model instance
    """
    challengers = {
        "xgboost": XGBoostChallenger,
        "lightgbm": LightGBMChallenger,
        "ft_transformer": FTTransformerChallenger,
        "tabnet": TabNetChallenger,
    }
    
    if algorithm.lower() not in challengers:
        raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(challengers.keys())}")
    
    return challengers[algorithm.lower()](**kwargs)


def get_available_algorithms() -> List[str]:
    """Get list of available challenger algorithms."""
    return ["xgboost", "lightgbm", "ft_transformer", "tabnet"]
