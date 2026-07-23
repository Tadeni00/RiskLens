"""
FraudTrap — Phase 2: Semi-Supervised Training Pipeline
Manages the full training lifecycle for the NetPFN model:
  Pseudo-label generation → Dataset construction → Training → Calibration → Validation
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
from loguru import logger

from models.cold_start.ensemble import ColdStartEnsemble
from models.semi_supervised.netpfn import NetPFNModel, NetPFNWrapper, NetPFNLoss
from models.semi_supervised.prediction import (
    SemiSupervisedPrediction,
    PseudoLabelResult,
)
from scoring.calibration import ProbabilityCalibrator


@dataclass
class SemiSupervisedConfig:
    """Configuration for Phase 2 training."""

    # Model architecture
    embedding_dim: int = 64
    hidden_dims: Optional[List[int]] = None
    dropout: float = 0.2
    temperature: float = 10.0

    # Training hyperparameters
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 7
    compactness_weight: float = 0.1
    separation_weight: float = 0.1

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
    epochs_trained: int = 0
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
            "epochs_trained": self.epochs_trained,
            "calibration_error": self.calibration_error,
            "status": self.status,
            "error": self.error,
        }


class SemiSupervisedTrainer:
    """
    Training pipeline for the Phase 2 NetPFN model.

    Workflow:
    1. Generate pseudo-labels from cold-start ensemble
    2. Combine confirmed + pseudo labels
    3. Train NetPFN with prototype-based learning
    4. Calibrate probabilities
    5. Validate on held-out set
    """

    def __init__(self, config: Optional[SemiSupervisedConfig] = None):
        self.config = config or SemiSupervisedConfig()
        if self.config.hidden_dims is None:
            self.config.hidden_dims = [128, 96]

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
            "Dataset prepared: {} confirmed + {} pseudo = {} total "
            "(fraud rate: {:.3%})",
            len(y_confirmed),
            pseudo_result.high_conf_count + pseudo_result.low_conf_count,
            len(y_combined),
            y_combined.mean(),
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
            "Pseudo-labels: {} high-confidence fraud, {} low-confidence legit, "
            "{} sent to review (thresholds: high={:.2f}, low={:.2f})",
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
    ) -> Tuple[NetPFNWrapper, SemiSupervisedTrainingResult]:
        """
        Train the NetPFN model end-to-end.

        Returns:
            (trained wrapper, training result)
        """
        t_start = time.time()
        result = SemiSupervisedTrainingResult()

        try:
            input_dim = X.shape[1]
            feature_names = feature_names or [f"f_{i}" for i in range(input_dim)]

            # Split data
            X_train, X_val, y_train, y_val, w_train, w_val = self._split_data(
                X, y, sample_weights
            )

            result.n_confirmed = int(y.sum())
            result.n_total = len(y)

            # Build wrapper
            wrapper = NetPFNWrapper(
                input_dim=input_dim,
                feature_names=feature_names,
                embedding_dim=self.config.embedding_dim,
                hidden_dims=self.config.hidden_dims,
                dropout=self.config.dropout,
                temperature=self.config.temperature,
                calibration_method=self.config.calibration_method,
            )

            # Scale features
            X_train_scaled = wrapper.scaler.fit_transform(X_train)
            X_val_scaled = wrapper.scaler.transform(X_val)

            # Build model
            wrapper.model = NetPFNModel(
                input_dim=input_dim,
                embedding_dim=self.config.embedding_dim,
                hidden_dims=self.config.hidden_dims,
                dropout=self.config.dropout,
                temperature=self.config.temperature,
            )

            # Training loop
            optimizer = torch.optim.Adam(
                wrapper.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.epochs
            )
            criterion = NetPFNLoss(
                compactness_weight=self.config.compactness_weight,
                separation_weight=self.config.separation_weight,
            )

            X_train_t = torch.FloatTensor(X_train_scaled)
            y_train_t = torch.FloatTensor(y_train)
            w_train_t = torch.FloatTensor(w_train) if w_train is not None else None
            X_val_t = torch.FloatTensor(X_val_scaled)
            y_val_t = torch.FloatTensor(y_val)

            train_dataset = TensorDataset(
                X_train_t, y_train_t, *([w_train_t] if w_train_t is not None else [])
            )
            sampler = self._create_sampler(y_train, w_train)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                sampler=sampler,
                drop_last=False,
            )

            best_val_pr_auc = 0.0
            patience_counter = 0
            best_state = None

            wrapper.model.train()
            for epoch in range(self.config.epochs):
                epoch_loss = 0.0
                n_batches = 0

                for batch in train_loader:
                    batch_X = batch[0]
                    batch_y = batch[1]
                    batch_w = batch[2] if len(batch) > 2 else None

                    optimizer.zero_grad()

                    fraud_prob, confidence, uncertainty = wrapper.model(
                        batch_X, labels=batch_y, weights=batch_w
                    )
                    loss = criterion(
                        fraud_prob, confidence, batch_y, batch_w, wrapper.model
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(wrapper.model.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                scheduler.step()

                # Validate
                if (epoch + 1) % 5 == 0 or epoch == self.config.epochs - 1:
                    val_metrics = self._validate(wrapper, X_val_t, y_val_t)
                    val_pr_auc = val_metrics["pr_auc"]

                    logger.debug(
                        "Epoch {}/{}: loss={:.4f} val_pr_auc={:.4f}",
                        epoch + 1,
                        self.config.epochs,
                        epoch_loss / max(n_batches, 1),
                        val_pr_auc,
                    )

                    if val_pr_auc > best_val_pr_auc:
                        best_val_pr_auc = val_pr_auc
                        patience_counter = 0
                        best_state = {
                            k: v.clone() for k, v in wrapper.model.state_dict().items()
                        }
                    else:
                        patience_counter += 1
                        if patience_counter >= self.config.early_stopping_patience:
                            logger.info("Early stopping at epoch {}", epoch + 1)
                            break

            # Restore best model
            if best_state is not None:
                wrapper.model.load_state_dict(best_state)

            # Calibrate on validation set
            wrapper.model.eval()
            with torch.no_grad():
                val_probs_raw, _, _ = wrapper.model.predict(X_val_t)
                val_probs_raw = val_probs_raw.numpy()

            wrapper.calibrator = ProbabilityCalibrator(
                method=self.config.calibration_method
            )
            wrapper.calibrator.fit(val_probs_raw, y_val)

            # Final metrics
            val_probs_cal = wrapper.calibrator.transform(val_probs_raw)
            result.pr_auc = float(average_precision_score(y_val, val_probs_cal))
            result.roc_auc = float(roc_auc_score(y_val, val_probs_cal))

            # Compute calibration error
            from sklearn.calibration import calibration_curve

            fraction_pos, mean_predicted = calibration_curve(
                y_val, val_probs_cal, n_bins=10
            )
            result.calibration_error = float(
                np.mean(np.abs(fraction_pos - mean_predicted))
            )

            result.epochs_trained = epoch + 1
            result.status = "completed"

            wrapper.is_fitted = True
            wrapper.pr_auc_ = result.pr_auc
            wrapper.roc_auc_ = result.roc_auc
            wrapper.trained_at = datetime.now(timezone.utc).isoformat()
            wrapper.model_version = f"v2_netpfn_{int(time.time())}"

            result.model_version = wrapper.model_version

        except Exception as exc:
            result.status = "failed"
            result.error = str(exc)
            logger.error("NetPFN training failed: {}", exc)
            raise

        finally:
            result.duration_seconds = time.time() - t_start

        logger.info(
            "NetPFN trained in {:.1f}s — PR-AUC: {:.4f}, ROC-AUC: {:.4f}, "
            "calibration error: {:.4f}",
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
        weights: Optional[np.ndarray],
    ) -> Tuple:
        val_ratio_adjusted = self.config.val_ratio / (
            self.config.train_ratio + self.config.val_ratio
        )

        X_train_val, X_val, y_train_val, y_val, w_train_val, w_val = (
            train_test_split(
                X,
                y,
                weights,
                test_size=val_ratio_adjusted,
                random_state=self.config.random_seed,
                stratify=y,
            )
            if weights is not None
            else (
                *train_test_split(
                    X,
                    y,
                    test_size=val_ratio_adjusted,
                    random_state=self.config.random_seed,
                    stratify=y,
                ),
                None,
                None,
            )
        )

        return X_train_val, X_val, y_train_val, y_val, w_train_val, w_val

    def _create_sampler(
        self,
        y: np.ndarray,
        weights: Optional[np.ndarray],
    ) -> Optional[WeightedRandomSampler]:
        """Create weighted sampler for class imbalance."""
        if weights is not None:
            sample_weights = weights
        else:
            n_fraud = max(int(y.sum()), 1)
            n_legit = max(int(len(y) - n_fraud), 1)
            w_fraud = len(y) / (2 * n_fraud)
            w_legit = len(y) / (2 * n_legit)
            sample_weights = np.where(y == 1, w_fraud, w_legit)

        return WeightedRandomSampler(
            weights=sample_weights.astype(float),
            num_samples=len(y),
            replacement=True,
        )

    @torch.no_grad()
    def _validate(
        self,
        wrapper: NetPFNWrapper,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
    ) -> Dict[str, float]:
        """Validate model on held-out data."""
        wrapper.model.eval()
        fraud_prob, confidence, uncertainty = wrapper.model.predict(X_val)
        probs = fraud_prob.numpy()

        # Use internal calibrator if available
        if wrapper.calibrator is not None:
            probs = wrapper.calibrator.transform(probs)

        pr_auc = float(average_precision_score(y_val.numpy(), probs))
        return {"pr_auc": pr_auc}
