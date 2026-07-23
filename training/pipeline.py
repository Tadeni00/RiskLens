"""
FraudTrap — Training Pipeline
Manages the full model lifecycle:
  Phase 1 (unsupervised) → Phase 2 (semi-supervised TabPFN) → Phase 3 (supervised CatBoost + FT-Transformer)
Includes dataset construction with point-in-time correct feature joins,
delayed label handling, and automated phase transition gating.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from loguru import logger
from sklearn.metrics import average_precision_score, f1_score

from config.settings import get_settings
from models.cold_start.ensemble import ColdStartEnsemble
from models.semi_supervised.tabpfn import TabPFNModel
from models.semi_supervised.trainer import SemiSupervisedTrainer, SemiSupervisedConfig
from models.supervised.champion import ChampionModel
from models.supervised.ft_transformer import FTTransformerPredictor
from models.supervised.meta_fusion import MetaFusionLayer

settings = get_settings()

MODEL_DIR = Path("./artifacts/models")
DATA_DIR = Path("./artifacts/data")


# ── Phase enum ────────────────────────────────────────────────────────────────


class ModelPhase(str, Enum):
    UNSUPERVISED = "UNSUPERVISED"
    SEMI_SUPERVISED = "SEMI_SUPERVISED"
    SUPERVISED = "SUPERVISED"


# ── Phase state (persisted between runs) ──────────────────────────────────────


@dataclass
class PhaseState:
    tenant_id: str
    current_phase: ModelPhase = ModelPhase.UNSUPERVISED
    total_transactions: int = 0
    confirmed_fraud_labels: int = 0
    first_transaction_at: Optional[str] = None
    last_retrain_at: Optional[str] = None
    current_model_version: str = "none"
    metrics: dict = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}

    def to_json(self) -> str:
        d = asdict(self)
        d["current_phase"] = self.current_phase.value
        return json.dumps(d)

    @classmethod
    def from_json(cls, s: str) -> "PhaseState":
        d = json.loads(s)
        d["current_phase"] = ModelPhase(d["current_phase"])
        return cls(**d)

    def weeks_since_first_transaction(self) -> float:
        if not self.first_transaction_at:
            return 0.0
        first = datetime.fromisoformat(self.first_transaction_at)
        return (datetime.now(timezone.utc) - first).days / 7.0


# ── Dataset builder ───────────────────────────────────────────────────────────


class DatasetBuilder:
    """
    Constructs point-in-time correct training datasets.
    Handles the 70-day label lag buffer to ensure near-complete labels.
    """

    def __init__(self, clickhouse_conn=None):
        self._conn = clickhouse_conn  # injected; None = load from parquet for dev

    def build_unsupervised_dataset(
        self,
        tenant_id: str,
        lookback_days: int = 30,
    ) -> pd.DataFrame:
        """
        Returns unlabelled transaction feature vectors for Phase 1 training.
        Filters out known-fraud transactions if any labels exist.
        """
        logger.info("Building unsupervised dataset for tenant={}", tenant_id)
        df = self._load_features(tenant_id, lookback_days)
        # Drop any rows with confirmed fraud labels — train only on "normal"
        if "label" in df.columns:
            df = df[df["label"] != 1].drop(columns=["label"], errors="ignore")
        logger.info("Unsupervised dataset: {} rows", len(df))
        return df

    def build_supervised_dataset(
        self,
        tenant_id: str,
        label_lag_days: int = None,
        training_window_days: int = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Returns (X, y) with point-in-time correct features.
        Training window: T-{window}d to T-{lag}d (ensuring labels are complete).
        """
        lag = label_lag_days or settings.label_lag_days
        window = training_window_days or settings.training_window_days
        logger.info(
            "Building supervised dataset for tenant={}, window={}d, lag={}d",
            tenant_id,
            window,
            lag,
        )
        df = self._load_features(tenant_id, window)

        # Only include transactions old enough to have received labels
        cutoff = datetime.now(timezone.utc) - timedelta(days=lag)
        if "transaction_timestamp" in df.columns:
            df = df[pd.to_datetime(df["transaction_timestamp"]) < cutoff]

        if "label" not in df.columns or df["label"].sum() == 0:
            raise ValueError(
                f"No fraud labels available for tenant={tenant_id}. "
                "Cannot build supervised dataset."
            )

        # Filter non-fraud chargebacks (bad reason codes)
        if "chargeback_reason_code" in df.columns:
            non_fraud_codes = {"4853", "4855", "4859"}  # Visa: item not received, etc.
            mask = ~(
                (df["label"] == 1)
                & (df["chargeback_reason_code"].isin(non_fraud_codes))
            )
            df = df[mask]

        feature_cols = [
            c
            for c in df.columns
            if c
            not in (
                "label",
                "transaction_id",
                "tenant_id",
                "transaction_timestamp",
                "chargeback_reason_code",
            )
        ]
        X = df[feature_cols].fillna(0.0)
        y = df["label"].astype(int)

        logger.info(
            "Supervised dataset: {} rows, {:.3f}% fraud",
            len(y),
            100 * y.mean(),
        )
        return X, y

    def _load_features(self, tenant_id: str, lookback_days: int) -> pd.DataFrame:
        """Load from ClickHouse in production; from Parquet files in dev."""
        parquet_path = DATA_DIR / tenant_id / "features.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            if "transaction_timestamp" in df.columns:
                cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
                df = df[pd.to_datetime(df["transaction_timestamp"]) > cutoff]
            return df

        # Return synthetic data for testing when no real data available
        logger.warning(
            "No data found for tenant={}; generating synthetic data", tenant_id
        )
        return _generate_synthetic_data(n=10_000)


# ── Phase transition evaluator ────────────────────────────────────────────────


class PhaseTransitionEvaluator:
    """
    Evaluates whether the current phase's readiness criteria are met.
    All criteria must pass simultaneously.
    """

    def should_transition_to_semi(self, state: PhaseState) -> tuple[bool, dict]:
        checks = {
            "fraud_labels": state.confirmed_fraud_labels
            >= settings.phase1_min_fraud_labels,
            "transactions": state.total_transactions
            >= settings.phase1_min_transactions,
            "weeks": state.weeks_since_first_transaction() >= settings.phase1_min_weeks,
            "pr_auc": state.metrics.get("pr_auc", 0.0) >= settings.phase1_min_pr_auc,
        }
        return all(checks.values()), checks

    def should_transition_to_supervised(self, state: PhaseState) -> tuple[bool, dict]:
        checks = {
            "fraud_labels": state.confirmed_fraud_labels
            >= settings.phase2_min_fraud_labels,
            "pr_auc": state.metrics.get("pr_auc", 0.0) >= settings.phase2_min_pr_auc,
        }
        return all(checks.values()), checks


# ── Training orchestrator ─────────────────────────────────────────────────────


class TrainingPipeline:
    """
    End-to-end training orchestrator. Called by the weekly retrain scheduler
    or drift-triggered retraining.
    """

    def __init__(self, clickhouse_conn=None):
        self.dataset_builder = DatasetBuilder(clickhouse_conn)
        self.evaluator = PhaseTransitionEvaluator()
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    def run(self, tenant_id: str, state: PhaseState) -> PhaseState:
        """
        Main entry point. Detects current phase, trains appropriate model,
        evaluates transition criteria, and returns updated state.
        """
        logger.info(
            "TrainingPipeline.run: tenant={}, phase={}",
            tenant_id,
            state.current_phase,
        )

        with mlflow.start_run(
            experiment_id=self._get_experiment_id(),
            run_name=f"{tenant_id}_{state.current_phase}_{datetime.now(timezone.utc).date()}",
        ):
            mlflow.set_tags(
                {
                    "tenant_id": tenant_id,
                    "phase": state.current_phase.value,
                }
            )

            if state.current_phase == ModelPhase.UNSUPERVISED:
                state = self._train_phase1(tenant_id, state)
                ready, checks = self.evaluator.should_transition_to_semi(state)
                logger.info("Phase1→2 transition check: {}", checks)
                if ready:
                    logger.info(
                        "✓ Transitioning tenant={} to SEMI_SUPERVISED", tenant_id
                    )
                    state.current_phase = ModelPhase.SEMI_SUPERVISED

            elif state.current_phase == ModelPhase.SEMI_SUPERVISED:
                state = self._train_phase2(tenant_id, state)
                ready, checks = self.evaluator.should_transition_to_supervised(state)
                logger.info("Phase2→3 transition check: {}", checks)
                if ready:
                    logger.info("✓ Transitioning tenant={} to SUPERVISED", tenant_id)
                    state.current_phase = ModelPhase.SUPERVISED

            elif state.current_phase == ModelPhase.SUPERVISED:
                state = self._train_phase3(tenant_id, state)

            state.last_retrain_at = datetime.now(timezone.utc).isoformat()
            mlflow.log_metrics(state.metrics)
            mlflow.log_params(
                {
                    "phase": state.current_phase.value,
                    "tenant_id": tenant_id,
                    "fraud_labels": state.confirmed_fraud_labels,
                }
            )

        return state

    def _train_phase1(self, tenant_id: str, state: PhaseState) -> PhaseState:
        df = self.dataset_builder.build_unsupervised_dataset(tenant_id)
        feature_cols = [
            c
            for c in df.columns
            if c not in ("transaction_id", "tenant_id", "transaction_timestamp")
        ]
        X = df[feature_cols].fillna(0.0).values
        model = ColdStartEnsemble(input_dim=X.shape[1], feature_names=feature_cols)
        model.fit(X, epochs=30)
        save_path = MODEL_DIR / tenant_id / "phase1"
        model.save(save_path)
        # Evaluate on a held-out slice if labels available
        scores = model.score(X[:1000] if len(X) > 1000 else X)
        state.metrics = {
            "anomaly_score_mean": float(scores.mean()),
            "anomaly_score_p95": float(np.percentile(scores, 95)),
        }
        state.current_model_version = f"phase1_{int(time.time())}"
        logger.info("Phase 1 model saved → {}", save_path)
        return state

    def _train_phase2(self, tenant_id: str, state: PhaseState) -> PhaseState:
        """Train Phase 2: TabPFN semi-supervised model."""
        try:
            X, y = self.dataset_builder.build_supervised_dataset(tenant_id)
        except ValueError as e:
            logger.warning("Phase2 dataset error: {}. Staying in semi-supervised.", e)
            return state

        feature_names = list(X.columns)
        X_values = X.values
        y_values = y.values

        # Get cold-start model for pseudo-label generation
        cold_start_path = MODEL_DIR / tenant_id / "phase1"
        cold_start = None
        if cold_start_path.exists():
            cold_start = ColdStartEnsemble.load(cold_start_path)

        # Also try to load unlabelled data for pseudo-labels
        X_unlabelled = None
        try:
            df_unlabelled = self.dataset_builder.build_unsupervised_dataset(tenant_id)
            if len(df_unlabelled) > 0:
                X_unlabelled = df_unlabelled[feature_names].fillna(0.0).values
        except Exception:
            pass

        # Train TabPFN
        config = SemiSupervisedConfig(
            n_estimators=4,
            ignore_pretraining_limits=True,
        )
        trainer = SemiSupervisedTrainer(config)

        X_combined, y_combined, weights, pseudo_result = trainer.prepare_dataset(
            X_confirmed=X_values,
            y_confirmed=y_values,
            X_unlabelled=X_unlabelled,
            cold_start=cold_start,
        )

        wrapper, result = trainer.train(
            X=X_combined,
            y=y_combined,
            sample_weights=weights,
            feature_names=feature_names,
        )

        # Save model
        save_path = MODEL_DIR / tenant_id / "phase2"
        wrapper.save(save_path)

        state.metrics = {
            "pr_auc": result.pr_auc,
            "roc_auc": result.roc_auc,
            "calibration_error": result.calibration_error,
        }
        state.current_model_version = result.model_version
        state.pseudo_label_count = result.n_pseudo
        logger.info(
            "Phase 2 TabPFN saved → {} | metrics={}",
            save_path,
            state.metrics,
        )
        return state

    def _train_phase3(self, tenant_id: str, state: PhaseState) -> PhaseState:
        """Train Phase 3: Confidence-aware CatBoost + FT-Transformer."""
        try:
            X, y = self.dataset_builder.build_supervised_dataset(tenant_id)
        except ValueError as e:
            logger.warning("Phase3 dataset error: {}. Keeping existing model.", e)
            return state

        feature_names = list(X.columns)
        X_values = X.values
        y_values = y.values

        # Split: 60% CatBoost train, 20% FT-Transformer train, 20% meta-fusion cal
        from sklearn.model_selection import train_test_split

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X_values, y_values, test_size=0.2, random_state=42, stratify=y_values
        )
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
        )

        # 1. Train CatBoost champion
        champion = ChampionModel(
            feature_names=feature_names,
            enable_specialist=True,
            confidence_threshold=0.92,
            fusion_method="logistic_regression",
        )
        champion.fit(
            X_train,
            y_train,
            feature_names=feature_names,
            calibration_method="isotonic",
        )

        # 2. Train FT-Transformer specialist
        ft_transformer = FTTransformerPredictor(
            n_features=X_train.shape[1],
            feature_names=feature_names,
            d_token=64,
            n_heads=4,
            n_layers=2,
            dropout=0.1,
            calibration_method="isotonic",
        )

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_cal_scaled = scaler.fit_transform(X_cal)

        ft_transformer.scaler = scaler
        ft_transformer.model.train()

        import torch
        from torch.utils.data import DataLoader, TensorDataset

        X_train_scaled = scaler.transform(X_train)
        X_train_t = torch.FloatTensor(X_train_scaled)
        y_train_t = torch.FloatTensor(y_train)

        dataset = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(dataset, batch_size=256, shuffle=True)

        optimizer = torch.optim.Adam(ft_transformer.model.parameters(), lr=1e-3)
        criterion = torch.nn.BCELoss()

        ft_transformer.model.train()
        for epoch in range(30):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                output = ft_transformer.model(batch_X)
                loss = criterion(output.squeeze(), batch_y)
                loss.backward()
                optimizer.step()

        ft_transformer.model.eval()
        ft_cal_probs = ft_transformer.predict_proba(X_cal)

        # Fit FT-Transformer calibration
        from scoring.calibration import ProbabilityCalibrator

        ft_calibrator = ProbabilityCalibrator(method="isotonic")
        ft_raw_probs = (
            ft_transformer.model(torch.FloatTensor(X_cal_scaled))
            .detach()
            .numpy()
            .ravel()
        )
        ft_calibrator.fit(ft_raw_probs, y_cal)
        ft_transformer.calibrator = ft_calibrator

        # 3. Train meta-fusion on calibration set
        champion.fit_specialist(
            X_cal=X_cal,
            y_cal=y_cal,
            ft_transformer=ft_transformer,
        )

        # 4. Evaluate on test set
        cat_probs = champion.predict_proba(X_test)
        from sklearn.metrics import average_precision_score

        test_pr_auc = float(average_precision_score(y_test, cat_probs))

        # Save all models
        save_path = MODEL_DIR / tenant_id / "phase3"
        champion.save(save_path)
        ft_path = save_path / "ft_transformer"
        ft_transformer.save(ft_path)

        state.metrics = {
            "pr_auc": champion.pr_auc_,
            "f2_score": champion.f2_score_,
            "test_pr_auc": test_pr_auc,
        }
        state.current_model_version = champion.model_version
        logger.info(
            "Phase 3 models saved → {} | metrics={}",
            save_path,
            state.metrics,
        )
        return state

    def _get_experiment_id(self) -> str:
        try:
            exp = mlflow.get_experiment_by_name(settings.mlflow_experiment_name)
            if exp:
                return exp.experiment_id
            return mlflow.create_experiment(settings.mlflow_experiment_name)
        except Exception:
            return "0"


# ── Synthetic data generator (for testing / demo) ────────────────────────────


def _generate_synthetic_data(
    n: int = 10_000, fraud_rate: float = 0.015
) -> pd.DataFrame:
    """
    Generates realistic-looking transaction feature data for testing.
    Fraud rate defaults to ~1.5%.
    """
    rng = np.random.default_rng(42)
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    def make_block(size, is_fraud):
        return {
            "amount": rng.lognormal(9 if is_fraud else 8, 1.5, size),
            "amount_log": rng.normal(9 if is_fraud else 8, 1.5, size),
            "amount_zscore": rng.normal(3.0 if is_fraud else 0.0, 1.0, size),
            "hour_sin": rng.uniform(-1, 1, size),
            "hour_cos": rng.uniform(-1, 1, size),
            "is_weekend": rng.binomial(1, 0.35 if is_fraud else 0.28, size).astype(
                float
            ),
            "is_night": rng.binomial(1, 0.45 if is_fraud else 0.15, size).astype(float),
            "is_round_amount": rng.binomial(1, 0.4 if is_fraud else 0.1, size).astype(
                float
            ),
            "is_new_merchant": rng.binomial(1, 0.7 if is_fraud else 0.1, size).astype(
                float
            ),
            "is_new_device": rng.binomial(1, 0.6 if is_fraud else 0.05, size).astype(
                float
            ),
            "device_shared_flag": rng.binomial(
                1, 0.5 if is_fraud else 0.02, size
            ).astype(float),
            "device_account_count": rng.integers(1, 20 if is_fraud else 3, size).astype(
                float
            ),
            "geo_speed_kmh": rng.exponential(800 if is_fraud else 30, size),
            "impossible_travel": rng.binomial(
                1, 0.3 if is_fraud else 0.001, size
            ).astype(float),
            "cross_country_flag": rng.binomial(
                1, 0.4 if is_fraud else 0.05, size
            ).astype(float),
            "acct_v_1m_count": rng.poisson(5 if is_fraud else 1, size).astype(float),
            "acct_v_1h_count": rng.poisson(20 if is_fraud else 3, size).astype(float),
            "acct_v_24h_count": rng.poisson(50 if is_fraud else 10, size).astype(float),
            "acct_v_24h_total_amt": rng.lognormal(12 if is_fraud else 10, 1.0, size),
            "typing_zscore": rng.normal(2.5 if is_fraud else 0.0, 1.0, size),
            "channel_enc": rng.integers(0, 6, size).astype(float),
            "txn_type_enc": rng.integers(0, 6, size).astype(float),
            "label": np.full(size, 1 if is_fraud else 0),
            "transaction_timestamp": [
                (
                    datetime.now(timezone.utc)
                    - timedelta(days=int(rng.integers(1, 180)))
                ).isoformat()
                for _ in range(size)
            ],
        }

    fraud_block = make_block(n_fraud, True)
    legit_block = make_block(n_legit, False)

    df = (
        pd.concat(
            [
                pd.DataFrame(fraud_block),
                pd.DataFrame(legit_block),
            ],
            ignore_index=True,
        )
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    return df
