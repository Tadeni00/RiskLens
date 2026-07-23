"""
FraudTrap — Phase 3: FT-Transformer Specialist Model
A tabular transformer that provides second opinions on difficult transactions
where CatBoost lacks confidence.

FT-Transformer treats each feature as a token and applies self-attention
to capture complex feature interactions that tree-based models may miss.

This model is NEVER executed for every transaction — only when the
ConfidenceEstimator determines CatBoost is insufficiently confident.
"""
from __future__ import annotations
import hashlib
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from loguru import logger

from scoring.calibration import ProbabilityCalibrator


# ── Transformer Architecture ─────────────────────────────────────────────────

class FeatureTokenizer(nn.Module):
    """Tokenizes each feature into a d-dimensional embedding."""

    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token
        self.feature_embeddings = nn.Linear(1, d_token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features)
        batch_size = x.shape[0]
        # Embed each feature separately: (batch, n_features, 1)
        x = x.unsqueeze(-1)
        # (batch, n_features, d_token)
        x = self.feature_embeddings(x)
        # Prepend CLS token: (batch, n_features+1, d_token)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        return x


class FTTransformerBlock(nn.Module):
    """Transformer encoder block with pre-norm."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class FTTransformerEncoder(nn.Module):
    """Full FT-Transformer encoder."""

    def __init__(
        self,
        n_features: int,
        d_token: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.layers = nn.ModuleList([
            FTTransformerBlock(d_token, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer(x)
        for layer in self.layers:
            x = layer(x)
        # Use CLS token output
        cls_output = x[:, 0, :]
        return self.head(cls_output)


# ── Wrapper ──────────────────────────────────────────────────────────────────

class FTTransformerPredictor:
    """
    Production wrapper for the FT-Transformer specialist model.
    
    Handles:
    - Feature scaling (StandardScaler)
    - PyTorch ↔ NumPy interface
    - Probability calibration
    - Version pinning
    - Persistence (save/load)
    
    This model is only invoked when the ConfidenceEstimator determines
    CatBoost is insufficiently confident.
    """

    def __init__(
        self,
        n_features: int,
        feature_names: Optional[List[str]] = None,
        d_token: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        calibration_method: str = "isotonic",
        model_version: str = "1.0.0",
        device: str = "cpu",
    ):
        self.n_features = n_features
        self.feature_names = feature_names or []
        self.calibration_method = calibration_method
        self.model_version = model_version
        self.device = device

        self.scaler = StandardScaler()
        self.model = FTTransformerEncoder(
            n_features=n_features,
            d_token=d_token,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.calibrator: Optional[ProbabilityCalibrator] = None
        self.is_fitted = False

        # Version pinning
        self.training_hash: Optional[str] = None
        self.feature_hash: Optional[str] = None
        self.dataset_hash: Optional[str] = None
        self.trained_at: Optional[str] = None
        self.pr_auc_: float = 0.0
        self.roc_auc_: float = 0.0

        # Config for persistence
        self._config = {
            "d_token": d_token,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "dropout": dropout,
        }

    def _compute_hashes(self, X: np.ndarray, y: np.ndarray) -> Dict[str, str]:
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
            [y_sample.mean(), len(y_sample)],
        ])
        dataset_hash = hashlib.sha256(data_stats.tobytes()).hexdigest()[:16]

        return {"feature_hash": feature_hash, "dataset_hash": dataset_hash}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated fraud probabilities."""
        if not self.is_fitted:
            raise RuntimeError("FTTransformerPredictor must be fitted before scoring")

        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        self.model.eval()
        with torch.no_grad():
            raw_probs = self.model(X_tensor).cpu().numpy().ravel()

        if self.calibrator is not None:
            return self.calibrator.transform(raw_probs)
        return raw_probs

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated fraud probabilities (alias for predict_proba)."""
        return self.predict_proba(X)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), path / "ft_transformer_model.pt")

        with open(path / "ft_transformer_metadata.pkl", "wb") as f:
            pickle.dump({
                "n_features": self.n_features,
                "feature_names": self.feature_names,
                "calibration_method": self.calibration_method,
                "model_version": self.model_version,
                "scaler": self.scaler,
                "calibrator": self.calibrator,
                "is_fitted": self.is_fitted,
                "training_hash": self.training_hash,
                "feature_hash": self.feature_hash,
                "dataset_hash": self.dataset_hash,
                "trained_at": self.trained_at,
                "pr_auc": self.pr_auc_,
                "roc_auc": self.roc_auc_,
                "config": self._config,
            }, f)

        logger.info("FTTransformerPredictor saved to {}", path)

    @classmethod
    def load(cls, path: Path) -> "FTTransformerPredictor":
        path = Path(path)

        with open(path / "ft_transformer_metadata.pkl", "rb") as f:
            payload = pickle.load(f)

        config = payload.get("config", {})
        obj = cls(
            n_features=payload["n_features"],
            feature_names=payload.get("feature_names", []),
            d_token=config.get("d_token", 64),
            n_heads=config.get("n_heads", 4),
            n_layers=config.get("n_layers", 2),
            dropout=config.get("dropout", 0.1),
            calibration_method=payload.get("calibration_method", "isotonic"),
            model_version=payload.get("model_version", "1.0.0"),
        )

        obj.scaler = payload["scaler"]
        obj.calibrator = payload.get("calibrator")
        obj.is_fitted = payload.get("is_fitted", False)
        obj.training_hash = payload.get("training_hash")
        obj.feature_hash = payload.get("feature_hash")
        obj.dataset_hash = payload.get("dataset_hash")
        obj.trained_at = payload.get("trained_at")
        obj.pr_auc_ = payload.get("pr_auc", 0.0)
        obj.roc_auc_ = payload.get("roc_auc", 0.0)

        obj.model.load_state_dict(torch.load(
            path / "ft_transformer_model.pt", map_location="cpu"
        ))
        obj.model.eval()

        logger.info(
            "FTTransformerPredictor loaded from {} (version={})",
            path, obj.model_version,
        )
        return obj
