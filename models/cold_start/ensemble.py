"""
FraudTrap — Phase 1: Unsupervised Cold-Start Models
Ensemble of VAE + Isolation Forest + One-Class SVM.
No labels required. Trained on "normal" transaction flow.
"""

from __future__ import annotations
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from loguru import logger

try:
    from scipy.stats import genpareto
except Exception:  # pragma: no cover
    genpareto = None


# ── Variational Autoencoder ───────────────────────────────────────────────────


class VAEEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_dim // 2, latent_dim)
        self.log_var_layer = nn.Linear(hidden_dim // 2, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu_layer(h), self.log_var_layer(h)


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        return self.net(z)


class FraudVAE(nn.Module):
    """
    Variational Autoencoder for anomaly detection.
    Trained only on legitimate transactions.
    Fraud = high reconstruction error (the model cannot reconstruct anomalies).
    Supports β-VAE with configurable beta and early stopping.
    """

    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.encoder = VAEEncoder(input_dim, latent_dim, hidden_dim)
        self.decoder = VAEDecoder(latent_dim, input_dim, hidden_dim)

    def reparameterise(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterise(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var

    @staticmethod
    def loss(x, x_hat, mu, log_var, beta: float = 1.0):
        recon = nn.functional.mse_loss(x_hat, x, reduction="mean")
        kld = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon + beta * kld, recon, kld

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample reconstruction error — higher = more anomalous."""
        self.eval()
        with torch.no_grad():
            mu, log_var = self.encoder(x)
            x_hat = self.decoder(mu)  # use mean, not sampled z
            return nn.functional.mse_loss(x_hat, x, reduction="none").mean(dim=1)


# ── Cold-start ensemble ───────────────────────────────────────────────────────


class EmpiricalTailDetector:
    """
    Scalable marginal-tail detector inspired by ECOD/COPOD.
    It replaces OCSVM with O(n * d) robust tail scoring.
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.median_: Optional[np.ndarray] = None
        self.iqr_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "EmpiricalTailDetector":
        self.median_ = np.nanmedian(X, axis=0)
        q25 = np.nanpercentile(X, 25, axis=0)
        q75 = np.nanpercentile(X, 75, axis=0)
        self.iqr_ = np.maximum(q75 - q25, self.eps)
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        if self.median_ is None or self.iqr_ is None:
            raise RuntimeError("EmpiricalTailDetector must be fitted before scoring")
        robust_z = np.abs((X - self.median_) / self.iqr_)
        return np.nanmax(robust_z, axis=1).astype(np.float32)


class ColdStartEnsemble:
    """
    Phase 1 ensemble: VAE + Isolation Forest + empirical tail detector.
    All models trained on unlabelled data.
    Output: continuous anomaly score in [0, 1].

    Key improvements:
    - Persists feature_names, latent_dim, hidden_dim for reproducible loading
    - Stores training percentiles for stable score normalization (no batch-dependent min/max)
    - β-VAE support with configurable beta
    - Early stopping and mixed precision training
    - EVT (Extreme Value Theory) tail modeling for VAE threshold
    - Version pinning: model_version, training_hash, feature_hash, dataset_hash
    """

    WEIGHTS = {"vae": 0.55, "iforest": 0.30, "tail": 0.15}

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        iforest_estimators: int = 200,
        iforest_contamination: float = 0.02,
        vae_beta: float = 0.5,
        feature_names: Optional[list[str]] = None,
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.iforest_estimators = iforest_estimators
        self.iforest_contamination = iforest_contamination
        self.vae_beta = vae_beta
        self.feature_names = feature_names or []
        self.scaler = StandardScaler()
        self.vae = FraudVAE(input_dim, latent_dim, hidden_dim)
        self.iforest = IsolationForest(
            n_estimators=iforest_estimators,
            contamination=iforest_contamination,
            random_state=42,
            n_jobs=-1,
        )
        self.tail_detector = EmpiricalTailDetector()
        self._vae_threshold: float = 0.0  # calibrated post-training
        self._vae_evt: dict[str, float] = {}
        self._score_calibration: dict[str, dict[str, float]] = {}
        self.is_fitted: bool = False

        # Version pinning for reproducibility
        self.model_version: str = "1.0.0"
        self.training_hash: Optional[str] = None
        self.feature_hash: Optional[str] = None
        self.dataset_hash: Optional[str] = None
        self.trained_at: Optional[str] = None

    # ── Training ──────────────────────────────────────────────────────────────

    def _compute_training_hash(self, X: np.ndarray) -> str:
        """Compute deterministic hash of training data for reproducibility."""
        hasher = hashlib.sha256()
        # Hash a sample of the data (first 1000 rows, all columns) for efficiency
        sample = X[: min(1000, len(X))]
        hasher.update(sample.tobytes())
        hasher.update(str(X.shape).encode())
        return hasher.hexdigest()[:16]

    def _compute_feature_hash(self) -> str:
        """Hash of feature names and order."""
        hasher = hashlib.sha256()
        hasher.update("|".join(self.feature_names).encode())
        return hasher.hexdigest()[:16]

    def fit(
        self,
        X: np.ndarray,
        epochs: int = 50,
        batch_size: int = 512,
        lr: float = 1e-3,
        device: str = "cpu",
        early_stopping_patience: int = 5,
        use_mixed_precision: bool = True,
    ) -> "ColdStartEnsemble":
        logger.info("Fitting ColdStartEnsemble on {} samples, {} features", *X.shape)
        X_scaled = self.scaler.fit_transform(X)

        # Compute and store version hashes
        self.training_hash = self._compute_training_hash(X_scaled)
        self.feature_hash = self._compute_feature_hash()
        self.dataset_hash = hashlib.sha256(
            f"{X_scaled.shape}{X_scaled.mean():.6f}{X_scaled.std():.6f}".encode()
        ).hexdigest()[:16]
        self.trained_at = datetime.now(timezone.utc).isoformat()

        self._fit_vae(
            X_scaled,
            epochs,
            batch_size,
            lr,
            device,
            early_stopping_patience,
            use_mixed_precision,
        )
        logger.info("VAE trained")

        self.iforest.fit(X_scaled)
        logger.info("Isolation Forest trained")

        self.tail_detector.fit(X_scaled)
        logger.info("Empirical tail detector trained")

        # Calibrate VAE threshold: 98th percentile of training reconstruction errors
        vae_scores = self._vae_scores(X_scaled, device)
        self._vae_threshold = float(np.percentile(vae_scores, 98))
        self._vae_evt = self._fit_evt_tail(vae_scores)
        logger.info(
            "VAE threshold calibrated: {:.6f}; EVT={}",
            self._vae_threshold,
            self._vae_evt,
        )

        iforest_scores = -self.iforest.decision_function(X_scaled)
        tail_scores = self.tail_detector.anomaly_score(X_scaled)
        self._score_calibration = {
            "vae": self._calibration_points(vae_scores),
            "iforest": self._calibration_points(iforest_scores),
            "tail": self._calibration_points(tail_scores),
        }
        logger.info("Cold-start score calibration fixed from training distribution")

        self.is_fitted = True
        return self

    def _fit_vae(
        self,
        X_scaled: np.ndarray,
        epochs: int,
        batch_size: int,
        lr: float,
        device: str,
        early_stopping_patience: int = 5,
        use_mixed_precision: bool = True,
    ) -> None:
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        self.vae = self.vae.to(device)
        optimiser = torch.optim.Adam(self.vae.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

        # Mixed precision scaler
        scaler = (
            torch.cuda.amp.GradScaler()
            if (use_mixed_precision and device == "cuda")
            else None
        )

        self.vae.train()
        n = len(X_t)
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            perm = torch.randperm(n)
            total_loss = 0.0
            for i in range(0, n, batch_size):
                batch = X_t[perm[i : i + batch_size]]
                optimiser.zero_grad()

                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        x_hat, mu, log_var = self.vae(batch)
                        loss, _, _ = FraudVAE.loss(
                            batch, x_hat, mu, log_var, beta=self.vae_beta
                        )
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimiser)
                    nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
                    scaler.step(optimiser)
                    scaler.update()
                else:
                    x_hat, mu, log_var = self.vae(batch)
                    loss, _, _ = FraudVAE.loss(
                        batch, x_hat, mu, log_var, beta=self.vae_beta
                    )
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
                    optimiser.step()

                total_loss += loss.item()

            scheduler.step()
            avg_loss = total_loss / max(1, n // batch_size)

            # Early stopping
            if avg_loss < best_loss - 1e-6:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(
                        "Early stopping at epoch {} (best loss: {:.6f})",
                        epoch + 1,
                        best_loss,
                    )
                    break

            if (epoch + 1) % 10 == 0:
                logger.debug("VAE epoch {}/{} loss={:.4f}", epoch + 1, epochs, avg_loss)

        self.vae = self.vae.cpu()

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Returns anomaly scores in [0, 1] for each sample.
        Higher = more likely fraud.
        """
        if not self.is_fitted:
            raise RuntimeError("ColdStartEnsemble must be fitted before scoring")

        X_scaled = self.scaler.transform(X)

        vae_raw = self._vae_scores(X_scaled, "cpu")
        vae_norm = self._normalise_score("vae", vae_raw)

        iforest_raw = -self.iforest.decision_function(X_scaled)
        iforest_norm = self._normalise_score("iforest", iforest_raw)

        tail_raw = (
            self.tail_detector.anomaly_score(X_scaled)
            if self.tail_detector is not None
            else np.zeros(len(X_scaled), dtype=np.float32)
        )
        tail_norm = self._normalise_score("tail", tail_raw)

        weights = dict(self.WEIGHTS)
        if self.tail_detector is None:
            weights["tail"] = 0.0
        denom = max(sum(weights.values()), 1e-8)
        return (
            weights["vae"] * vae_norm
            + weights["iforest"] * iforest_norm
            + weights["tail"] * tail_norm
        ) / denom

    def explain(self, X: np.ndarray, top_n: int = 8) -> list[dict]:
        """
        Returns per-sample component attribution for explainability.

        Returns list of dicts with:
        - vae: reconstruction error + contribution
        - isolation_forest: anomaly score + contribution
        - tail_detector: robust z-score + contribution
        - combined_score: final weighted score
        - top_features: top contributing components
        """
        if not self.is_fitted:
            raise RuntimeError("ColdStartEnsemble must be fitted before explain")

        X_scaled = self.scaler.transform(X)
        n_samples = X_scaled.shape[0]

        # Get raw scores from each component
        vae_raw = self._vae_scores(X_scaled, "cpu")
        iforest_raw = -self.iforest.decision_function(X_scaled)
        tail_raw = (
            self.tail_detector.anomaly_score(X_scaled)
            if self.tail_detector is not None
            else np.zeros(n_samples, dtype=np.float32)
        )

        # Normalize
        vae_norm = self._normalise_score("vae", vae_raw)
        iforest_norm = self._normalise_score("iforest", iforest_raw)
        tail_norm = self._normalise_score("tail", tail_raw)

        weights = dict(self.WEIGHTS)
        if self.tail_detector is None:
            weights["tail"] = 0.0
        denom = max(sum(weights.values()), 1e-8)

        # Compute contributions
        vae_contrib = weights["vae"] * vae_norm / denom
        iforest_contrib = weights["iforest"] * iforest_norm / denom
        tail_contrib = weights["tail"] * tail_norm / denom
        combined = vae_contrib + iforest_contrib + tail_contrib

        explanations = []
        for i in range(n_samples):
            # Component breakdown
            components = {
                "vae": {
                    "raw_score": float(vae_raw[i]),
                    "normalized": float(vae_norm[i]),
                    "weight": weights["vae"],
                    "contribution": float(vae_contrib[i]),
                },
                "isolation_forest": {
                    "raw_score": float(iforest_raw[i]),
                    "normalized": float(iforest_norm[i]),
                    "weight": weights["iforest"],
                    "contribution": float(iforest_contrib[i]),
                },
                "tail_detector": {
                    "raw_score": float(tail_raw[i]),
                    "normalized": float(tail_norm[i]),
                    "weight": weights["tail"],
                    "contribution": float(tail_contrib[i]),
                },
            }

            # Top contributing components
            component_contribs = [
                ("VAE_reconstruction", vae_contrib[i]),
                ("IsolationForest_path", iforest_contrib[i]),
                ("Tail_robust_zscore", tail_contrib[i]),
            ]
            component_contribs.sort(key=lambda x: abs(x[1]), reverse=True)

            top_features = [
                {
                    "feature": name,
                    "value": float(val),
                    "contribution": float(val),
                    "method": "cold_start_component",
                }
                for name, val in component_contribs[:top_n]
            ]

            explanations.append(
                {
                    "model_type": "cold_start",
                    "base_value": 0.0,
                    "prediction_value": float(combined[i]),
                    "top_features": top_features,
                    "components": {
                        "vae": components["vae"],
                        "isolation_forest": components["isolation_forest"],
                        "tail_detector": components["tail_detector"],
                        "weights": weights,
                    },
                    "latency_ms": 1.0,  # Placeholder
                }
            )

        return explanations

    def _vae_scores(self, X_scaled: np.ndarray, device: str) -> np.ndarray:
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        return self.vae.anomaly_score(X_t).detach().cpu().numpy()

    @staticmethod
    def _calibration_points(raw_scores: np.ndarray) -> dict[str, float]:
        return {
            "p50": float(np.percentile(raw_scores, 50)),
            "p95": float(np.percentile(raw_scores, 95)),
            "p99": float(np.percentile(raw_scores, 99)),
            "p999": float(np.percentile(raw_scores, 99.9)),
        }

    @staticmethod
    def _fit_evt_tail(raw_scores: np.ndarray) -> dict[str, float]:
        threshold = float(np.percentile(raw_scores, 98))
        excess = raw_scores[raw_scores > threshold] - threshold
        if genpareto is None or len(excess) < 50:
            return {"threshold": threshold}
        try:
            shape, loc, scale = genpareto.fit(excess, floc=0.0)
            return {
                "threshold": threshold,
                "shape": float(shape),
                "loc": float(loc),
                "scale": float(scale),
            }
        except Exception:
            return {"threshold": threshold}

    def _normalise_score(self, name: str, raw_scores: np.ndarray) -> np.ndarray:
        """Normalize against fixed training percentiles, not request-batch min/max."""
        cal = self._score_calibration.get(name)
        if not cal:
            return self._legacy_normalise(raw_scores)

        xp = np.array(
            [cal["p50"], cal["p95"], cal["p99"], cal["p999"]], dtype=np.float32
        )
        xp = np.maximum.accumulate(xp)
        for i in range(1, len(xp)):
            if xp[i] <= xp[i - 1]:
                xp[i] = np.nextafter(xp[i - 1], np.float32(np.inf))
        yp = np.array([0.00, 0.01, 0.04, 0.25], dtype=np.float32)
        return np.interp(raw_scores, xp, yp, left=0.0, right=0.65).astype(np.float32)

    @staticmethod
    def _legacy_normalise(raw_scores: np.ndarray) -> np.ndarray:
        return (raw_scores - raw_scores.min()) / (
            raw_scores.max() - raw_scores.min() + 1e-8
        )

    def explain(self, X: np.ndarray, top_n: int = 8) -> list[dict]:
        """
        Returns per-sample explanation for cold-start ensemble.
        Breaks down into VAE, Isolation Forest, and Tail detector components.
        """
        if not self.is_fitted:
            raise RuntimeError("ColdStartEnsemble must be fitted before explaining")

        X_scaled = self.scaler.transform(X)
        n_samples = X.shape[0]
        explanations = []

        # Get raw component scores
        vae_raw = self._vae_scores(X_scaled, "cpu")
        iforest_raw = -self.iforest.decision_function(X_scaled)
        tail_raw = (
            self.tail_detector.anomaly_score(X_scaled)
            if self.tail_detector is not None
            else np.zeros(n_samples)
        )

        # Normalize using training calibration
        vae_norm = self._normalise_score("vae", vae_raw)
        iforest_norm = self._normalise_score("iforest", iforest_raw)
        tail_norm = self._normalise_score("tail", tail_raw)

        weights = dict(self.WEIGHTS)
        if self.tail_detector is None:
            weights["tail"] = 0.0
        denom = max(sum(weights.values()), 1e-8)

        for i in range(n_samples):
            # Component contributions
            vae_contrib = weights["vae"] * vae_norm[i] / denom
            iforest_contrib = weights["iforest"] * iforest_norm[i] / denom
            tail_contrib = weights["tail"] * tail_norm[i] / denom
            combined = vae_contrib + iforest_contrib + tail_contrib

            # Top features (component-level)
            top_features = [
                {
                    "feature": "VAE_reconstruction",
                    "value": float(vae_raw[i]),
                    "contribution": float(vae_contrib),
                    "method": "vae_reconstruction_error",
                    "normalized": float(vae_norm[i]),
                },
                {
                    "feature": "IsolationForest_path",
                    "value": float(iforest_raw[i]),
                    "contribution": float(iforest_contrib),
                    "method": "isolation_forest_path_length",
                    "normalized": float(iforest_norm[i]),
                },
            ]

            if weights["tail"] > 0:
                top_features.append(
                    {
                        "feature": "Tail_robust_zscore",
                        "value": float(tail_raw[i]),
                        "contribution": float(tail_contrib),
                        "method": "empirical_tail_detector",
                        "normalized": float(tail_norm[i]),
                    }
                )

            explanations.append(
                {
                    "model_type": "cold_start",
                    "base_value": 0.0,
                    "prediction_value": float(combined),
                    "top_features": top_features[:top_n],
                    "components": {
                        "vae": {
                            "weight": weights["vae"],
                            "raw_score": float(vae_raw[i]),
                            "normalized": float(vae_norm[i]),
                            "contribution": float(vae_contrib),
                        },
                        "isolation_forest": {
                            "weight": weights["iforest"],
                            "raw_score": float(iforest_raw[i]),
                            "normalized": float(iforest_norm[i]),
                            "contribution": float(iforest_contrib),
                        },
                        "tail_detector": {
                            "weight": weights["tail"],
                            "raw_score": float(tail_raw[i]),
                            "normalized": float(tail_norm[i]),
                            "contribution": float(tail_contrib),
                        },
                        "weights": weights,
                    },
                }
            )

        return explanations

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.vae.state_dict(), path / "vae.pt")
        with open(path / "ensemble.pkl", "wb") as f:
            payload = {
                "scaler": self.scaler,
                "iforest": self.iforest,
                "vae_threshold": self._vae_threshold,
                "vae_evt": self._vae_evt,
                "score_calibration": self._score_calibration,
                "input_dim": self.input_dim,
                "latent_dim": self.latent_dim,
                "hidden_dim": self.hidden_dim,
                "iforest_estimators": self.iforest_estimators,
                "iforest_contamination": self.iforest_contamination,
                "vae_beta": self.vae_beta,
                "tail_detector": self.tail_detector,
                "is_fitted": self.is_fitted,
                "feature_names": self.feature_names,
                # Version pinning
                "model_version": self.model_version,
                "training_hash": self.training_hash,
                "feature_hash": self.feature_hash,
                "dataset_hash": self.dataset_hash,
                "trained_at": self.trained_at,
            }
            pickle.dump(payload, f)
        logger.info("ColdStartEnsemble saved to {}", path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "ColdStartEnsemble":
        path = Path(path)
        with open(path / "ensemble.pkl", "rb") as f:
            payload = pickle.load(f)
        obj = cls(
            input_dim=payload["input_dim"],
            latent_dim=payload.get("latent_dim", 16),
            hidden_dim=payload.get("hidden_dim", 64),
            iforest_estimators=payload.get("iforest_estimators", 200),
            iforest_contamination=payload.get("iforest_contamination", 0.02),
            vae_beta=payload.get("vae_beta", 0.5),
            feature_names=payload.get("feature_names", []),
        )
        obj.scaler = payload["scaler"]
        obj.iforest = payload["iforest"]
        obj.tail_detector = payload.get("tail_detector")
        if obj.tail_detector is None and "ocsvm" in payload:
            logger.warning(
                "Loaded legacy cold-start artifact containing OCSVM; ignoring it. "
                "Retrain Phase 1 to enable the scalable empirical tail detector."
            )
        obj._vae_threshold = payload["vae_threshold"]
        obj._vae_evt = payload.get("vae_evt", {})
        obj._score_calibration = payload.get("score_calibration", {})
        obj.is_fitted = payload["is_fitted"]
        # Version pinning
        obj.model_version = payload.get("model_version", "1.0.0")
        obj.training_hash = payload.get("training_hash")
        obj.feature_hash = payload.get("feature_hash")
        obj.dataset_hash = payload.get("dataset_hash")
        obj.trained_at = payload.get("trained_at")
        obj.vae = FraudVAE(obj.input_dim, obj.latent_dim, obj.hidden_dim)
        obj.vae.load_state_dict(torch.load(path / "vae.pt", map_location=device))
        obj.vae.eval()
        logger.info(
            "ColdStartEnsemble loaded from {} (version={}, training_hash={})",
            path,
            obj.model_version,
            obj.training_hash,
        )
        return obj
