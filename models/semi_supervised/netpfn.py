"""
FraudTrap — Phase 2: NetPFN (Neural Prototypical Few-shot Network)
A prototype-based neural classifier designed for few-label, weak-label,
and pseudo-label fraud detection scenarios.

NetPFN learns class prototypes in a learned embedding space and classifies
by distance to prototypes. It excels when:
- Confirmed fraud labels are scarce (< 5000)
- Weak labels (chargebacks, analyst feedback) are noisy
- Pseudo-labels from cold-start are available
- Class imbalance is extreme

Architecture:
  Input Features → Encoder Network → Embedding Space →
  Class Prototypes → Distance-based Classification →
  Calibrated Fraud Probability + Uncertainty Estimate
"""
from __future__ import annotations
import hashlib
import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from loguru import logger

from scoring.calibration import ProbabilityCalibrator
from models.semi_supervised.prediction import SemiSupervisedPrediction


# ── Encoder Network ──────────────────────────────────────────────────────────

class PrototypeEncoder(nn.Module):
    """
    Feature encoder that maps raw transaction features into a learned
    embedding space where class prototypes are computed.
    
    Architecture: Linear → LayerNorm → ReLU → Linear → LayerNorm → ReLU → Linear
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        if hidden_dims is None:
            hidden_dims = [128, 96]

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, embedding_dim))
        layers.append(nn.LayerNorm(embedding_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── NetPFN Model ─────────────────────────────────────────────────────────────

class NetPFNModel(nn.Module):
    """
    Neural Prototypical Few-shot Network for semi-supervised fraud detection.
    
    Key properties:
    - Learns class prototypes from limited labelled data
    - Supports weak labels via weighted prototype updates
    - Produces calibrated probabilities + uncertainty estimates
    - Uncertainty = distance to nearest prototype relative to class spread
    
    Training modes:
    - Full supervision: use confirmed labels
    - Semi-supervised: combine confirmed + pseudo labels
    - Weak supervision: weight samples by label quality
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        temperature: float = 10.0,
        prototype_momentum: float = 0.9,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.prototype_momentum = prototype_momentum

        self.encoder = PrototypeEncoder(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        # Class prototypes (learned parameters)
        self.register_buffer(
            "prototypes",
            torch.zeros(2, embedding_dim),  # [legit_proto, fraud_proto]
        )
        self.register_buffer(
            "prototype_counts",
            torch.zeros(2, dtype=torch.long),
        )
        self.register_buffer(
            "class_spread",
            torch.ones(2, dtype=torch.float32),  # intra-class distance
        )

    def _compute_prototypes(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Update class prototypes using exponential moving average.
        If weights provided, use weighted averaging (for weak/pseudo labels).
        """
        with torch.no_grad():
            for c in range(2):
                mask = labels == c
                if mask.sum() == 0:
                    continue

                class_embeddings = embeddings[mask]
                if weights is not None:
                    class_weights = weights[mask].unsqueeze(1)
                    class_weights = class_weights / class_weights.sum()
                    proto = (class_embeddings * class_weights).sum(dim=0)
                else:
                    proto = class_embeddings.mean(dim=0)

                # EMA update
                self.prototypes[c] = (
                    self.prototype_momentum * self.prototypes[c]
                    + (1 - self.prototype_momentum) * proto
                )

                # Update class spread (mean intra-class distance)
                dists = torch.cdist(class_embeddings, self.prototypes[c].unsqueeze(0)).squeeze()
                self.class_spread[c] = 0.9 * self.class_spread[c] + 0.1 * dists.mean()

                self.prototype_counts[c] += int(mask.sum())

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        update_prototypes: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            probabilities: Fraud probabilities in [0, 1]
            confidence: Prediction confidence in [0, 1]
            uncertainty: Prediction uncertainty in [0, 1]
        """
        embeddings = self.encoder(x)

        if labels is not None and update_prototypes:
            self._compute_prototypes(embeddings, labels, weights)

        # Compute distances to prototypes
        dist_to_legit = torch.cdist(
            embeddings, self.prototypes[0:1]
        ).squeeze(-1)  # (batch,)
        dist_to_fraud = torch.cdist(
            embeddings, self.prototypes[1:2]
        ).squeeze(-1)  # (batch,)

        # Convert distances to probabilities via softmax with temperature
        neg_dists = torch.stack([-dist_to_legit, -dist_to_fraud], dim=1)
        probs = F.softmax(neg_dists * self.temperature, dim=1)

        fraud_prob = probs[:, 1]

        # Confidence: max class probability
        confidence = probs.max(dim=1).values

        # Uncertainty: entropy-based + prototype distance ratio
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
        max_entropy = np.log(2)
        entropy_uncertainty = entropy / max_entropy

        # Distance ratio: how close is the sample to the decision boundary
        total_dist = dist_to_legit + dist_to_fraud + 1e-8
        distance_uncertainty = torch.abs(dist_to_legit - dist_to_fraud) / total_dist
        distance_uncertainty = 1.0 - distance_uncertainty  # closer to boundary = more uncertain

        # Combined uncertainty
        uncertainty = 0.6 * entropy_uncertainty + 0.4 * distance_uncertainty
        uncertainty = uncertainty.clamp(0.0, 1.0)

        return fraud_prob, confidence, uncertainty

    def predict(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference mode forward pass (no prototype update)."""
        self.eval()
        with torch.no_grad():
            return self.forward(x, update_prototypes=False)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get embedding vectors for input features."""
        self.eval()
        with torch.no_grad():
            return self.encoder(x)

    def get_prototype_distances(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get distances to each class prototype."""
        self.eval()
        with torch.no_grad():
            embeddings = self.encoder(x)
            dist_legit = torch.cdist(
                embeddings, self.prototypes[0:1]
            ).squeeze(-1)
            dist_fraud = torch.cdist(
                embeddings, self.prototypes[1:2]
            ).squeeze(-1)
            return {
                "dist_legit": dist_legit,
                "dist_fraud": dist_fraud,
            }


# ── Loss Function ────────────────────────────────────────────────────────────

class NetPFNLoss(nn.Module):
    """
    Combined loss for NetPFN training:
    - Prototype-based cross-entropy loss
    - Prototype compactness regularizer (intra-class distances should be small)
    - Prototype separation regularizer (inter-class distances should be large)
    """

    def __init__(
        self,
        compactness_weight: float = 0.1,
        separation_weight: float = 0.1,
    ):
        super().__init__()
        self.compactness_weight = compactness_weight
        self.separation_weight = separation_weight

    def forward(
        self,
        fraud_prob: torch.Tensor,
        confidence: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[torch.Tensor],
        model: NetPFNModel,
    ) -> torch.Tensor:
        # Binary cross-entropy for classification
        bce = F.binary_cross_entropy(fraud_prob, labels.float(), reduction="none")
        if weights is not None:
            bce = bce * weights
        classification_loss = bce.mean()

        # Compactness: mean intra-class distance should be small
        compactness_loss = model.class_spread.mean()

        # Separation: distance between prototypes should be large
        proto_dist = torch.cdist(
            model.prototypes.unsqueeze(0),
            model.prototypes.unsqueeze(0),
        ).squeeze()
        # We want to maximize the off-diagonal distance
        separation_loss = -proto_dist[0, 1]

        total = (
            classification_loss
            + self.compactness_weight * compactness_loss
            + self.separation_weight * separation_loss
        )
        return total


# ── Wrapper with Scaler + Calibration ────────────────────────────────────────

class NetPFNWrapper:
    """
    Production wrapper around NetPFNModel.
    
    Handles:
    - Feature scaling (StandardScaler)
    - PyTorch ↔ NumPy interface
    - Probability calibration (Isotonic/Platt)
    - Strongly typed predictions
    - Version pinning
    - Persistence (save/load)
    """

    def __init__(
        self,
        input_dim: int,
        feature_names: Optional[List[str]] = None,
        embedding_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        temperature: float = 10.0,
        calibration_method: str = "isotonic",
        model_version: str = "1.0.0",
    ):
        self.input_dim = input_dim
        self.feature_names = feature_names or []
        self.calibration_method = calibration_method
        self.model_version = model_version

        self.scaler = StandardScaler()
        self.model = NetPFNModel(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            temperature=temperature,
        )
        self.calibrator: Optional[ProbabilityCalibrator] = None
        self.is_fitted = False

        # Version pinning
        self.training_hash: Optional[str] = None
        self.feature_hash: Optional[str] = None
        self.dataset_hash: Optional[str] = None
        self.trained_at: Optional[str] = None

        # Training stats
        self.confirmed_label_count: int = 0
        self.pseudo_label_count: int = 0
        self.training_iteration: int = 0
        self.pr_auc_: float = 0.0
        self.roc_auc_: float = 0.0

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
            raise RuntimeError("NetPFNWrapper must be fitted before scoring")

        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)

        fraud_prob, _, _ = self.model.predict(X_tensor)
        raw_probs = fraud_prob.numpy()

        if self.calibrator is not None:
            return self.calibrator.transform(raw_probs)
        return raw_probs

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> List[SemiSupervisedPrediction]:
        """Return fully typed predictions with uncertainty estimates."""
        if not self.is_fitted:
            raise RuntimeError("NetPFNWrapper must be fitted before scoring")

        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)

        fraud_prob, confidence, uncertainty = self.model.predict(X_tensor)

        probs = fraud_prob.numpy()
        if self.calibrator is not None:
            probs = self.calibrator.transform(probs)

        return [
            SemiSupervisedPrediction(
                probability=float(probs[i]),
                confidence=float(confidence[i]),
                uncertainty=float(uncertainty[i]),
                model_version=self.model_version,
            )
            for i in range(len(X))
        ]

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated fraud probabilities (alias for predict_proba)."""
        return self.predict_proba(X)

    def explain(
        self, X: np.ndarray, top_n: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Feature attribution via prototype distance decomposition.
        For each sample, compute contribution of each feature to
        the distance to each prototype.
        """
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)

        embeddings = self.model.get_embeddings(X_tensor)
        distances = self.model.get_prototype_distances(X_tensor)

        explanations = []
        for i in range(len(X)):
            dist_legit = float(distances["dist_legit"][i])
            dist_fraud = float(distances["dist_fraud"][i])
            total = dist_legit + dist_fraud + 1e-8

            fraud_prob = float(
                self.model.predict(X_tensor[i:i+1])[0][0]
            )

            # Feature contribution: per-feature deviation from prototype
            feature_names = (
                self.feature_names
                if self.feature_names
                else [f"f_{j}" for j in range(X.shape[1])]
            )

            fraud_proto = self.model.prototypes[1].numpy()
            legit_proto = self.model.prototypes[0].numpy()

            # Per-feature distance contribution
            feat_contributions = []
            for j, fname in enumerate(feature_names):
                feat_val = float(X_scaled[i, j])
                fraud_dist_j = abs(feat_val - fraud_proto[j])
                legit_dist_j = abs(feat_val - legit_proto[j])
                # Contribution: how much more/less aligned with fraud vs legit
                contribution = legit_dist_j - fraud_dist_j
                feat_contributions.append((fname, feat_val, contribution))

            feat_contributions.sort(key=lambda x: abs(x[2]), reverse=True)
            top_features = [
                {
                    "feature": name,
                    "value": val,
                    "contribution": float(contrib),
                    "method": "prototype_distance",
                }
                for name, val, contrib in feat_contributions[:top_n]
            ]

            explanations.append({
                "model_type": "netpfn",
                "base_value": 0.5,
                "prediction_value": fraud_prob,
                "top_features": top_features,
                "components": {
                    "dist_legit": dist_legit,
                    "dist_fraud": dist_fraud,
                    "prototype_counts": self.model.prototype_counts.tolist(),
                },
            })

        return explanations

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), path / "netpfn_model.pt")

        with open(path / "netpfn_metadata.pkl", "wb") as f:
            pickle.dump({
                "input_dim": self.input_dim,
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
                "confirmed_label_count": self.confirmed_label_count,
                "pseudo_label_count": self.pseudo_label_count,
                "training_iteration": self.training_iteration,
                "pr_auc": self.pr_auc_,
                "roc_auc": self.roc_auc_,
                "model_config": {
                    "embedding_dim": self.model.embedding_dim,
                    "temperature": self.model.temperature,
                    "prototype_momentum": self.model.prototype_momentum,
                },
                "prototypes": self.model.prototypes,
                "prototype_counts": self.model.prototype_counts,
                "class_spread": self.model.class_spread,
            }, f)

        logger.info("NetPFNWrapper saved to {}", path)

    @classmethod
    def load(cls, path: Path) -> "NetPFNWrapper":
        path = Path(path)

        with open(path / "netpfn_metadata.pkl", "rb") as f:
            payload = pickle.load(f)

        config = payload.get("model_config", {})
        obj = cls(
            input_dim=payload["input_dim"],
            feature_names=payload.get("feature_names", []),
            embedding_dim=config.get("embedding_dim", 64),
            temperature=config.get("temperature", 10.0),
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
        obj.confirmed_label_count = payload.get("confirmed_label_count", 0)
        obj.pseudo_label_count = payload.get("pseudo_label_count", 0)
        obj.training_iteration = payload.get("training_iteration", 0)
        obj.pr_auc_ = payload.get("pr_auc", 0.0)
        obj.roc_auc_ = payload.get("roc_auc", 0.0)

        # Restore model state
        obj.model.load_state_dict(torch.load(
            path / "netpfn_model.pt", map_location="cpu"
        ))
        obj.model.eval()

        # Restore buffers
        if "prototypes" in payload:
            obj.model.prototypes = payload["prototypes"]
        if "prototype_counts" in payload:
            obj.model.prototype_counts = payload["prototype_counts"]
        if "class_spread" in payload:
            obj.model.class_spread = payload["class_spread"]

        logger.info(
            "NetPFNWrapper loaded from {} (version={})",
            path, obj.model_version,
        )
        return obj
