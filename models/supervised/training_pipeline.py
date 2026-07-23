"""
FraudTrap — Offline Training Pipeline
Trains champion and challenger models offline.
Supports the Champion-Challenger supervised learning architecture.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
import numpy as np
from loguru import logger

from models.supervised.champion import ChampionModel, train_champion
from models.supervised.challengers import (
    BaseChallenger,
    create_challenger,
    get_available_algorithms,
)
from models.supervised.registry import ModelRegistry, ModelMetadata, ModelStatus
from models.supervised.evaluator import ModelEvaluator, EvaluationMetrics
from models.supervised.promotion import ChampionPromoter, PromotionCriteria
from scoring.calibration import ProbabilityCalibrator


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""

    # Data split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Random seed
    random_seed: int = 42

    # Champion training
    champion_algorithm: str = "catboost"
    champion_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "iterations": 1000,
            "depth": 6,
            "learning_rate": 0.05,
            "calibration_method": "isotonic",
        }
    )

    # Challenger training
    challenger_algorithms: List[str] = field(
        default_factory=lambda: ["xgboost", "lightgbm", "ft_transformer", "tabnet"]
    )
    challenger_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Evaluation
    fpr_threshold: float = 0.01
    promotion_threshold: float = 0.01
    max_fpr: float = 0.01
    max_calibration_error: float = 0.05

    # Output
    output_dir: Path = Path("models/supervised/training_runs")


@dataclass
class TrainingRun:
    """Result of a training run."""

    run_id: str
    start_time: str
    end_time: Optional[str] = None
    duration_seconds: float = 0.0

    # Data info
    n_samples: int = 0
    n_features: int = 0
    fraud_rate: float = 0.0
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0

    # Champion results
    champion_id: Optional[str] = None
    champion_metrics: Optional[Dict[str, float]] = None

    # Challenger results
    challenger_ids: List[str] = field(default_factory=list)
    challenger_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Evaluation
    evaluation_report_id: Optional[str] = None
    recommended_champion: Optional[str] = None
    promotion_recommended: bool = False

    # Status
    status: str = "pending"
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "fraud_rate": self.fraud_rate,
            "train_samples": self.train_samples,
            "val_samples": self.val_samples,
            "test_samples": self.test_samples,
            "champion_id": self.champion_id,
            "champion_metrics": self.champion_metrics,
            "challenger_ids": self.challenger_ids,
            "challenger_metrics": self.challenger_metrics,
            "evaluation_report_id": self.evaluation_report_id,
            "recommended_champion": self.recommended_champion,
            "promotion_recommended": self.promotion_recommended,
            "status": self.status,
            "error": self.error,
        }


class TrainingPipeline:
    """
    Offline training pipeline for Champion-Challenger architecture.

    Features:
    - Trains champion model (CatBoost)
    - Trains multiple challenger models (XGBoost, LightGBM, FT-Transformer, TabNet)
    - Evaluates all models
    - Recommends champion promotion
    - Saves models and reports
    """

    def __init__(self, config: TrainingConfig = None):
        """
        Initialize training pipeline.

        Args:
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        self.registry = ModelRegistry()
        self.evaluator = ModelEvaluator()
        self.promoter = ChampionPromoter(
            registry=self.registry,
            criteria=PromotionCriteria(
                min_pr_auc_improvement=self.config.promotion_threshold,
                max_fpr=self.config.max_fpr,
                max_calibration_error=self.config.max_calibration_error,
            ),
        )

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        categorical_indices: Optional[List[int]] = None,
        dataset_version: str = "1.0.0",
        feature_version: str = "1.0.0",
        callbacks: Optional[Dict[str, Callable]] = None,
    ) -> TrainingRun:
        """
        Run the full training pipeline.

        Args:
            X: Feature matrix
            y: Labels (0=legit, 1=fraud)
            feature_names: List of feature names
            categorical_indices: Indices of categorical features
            dataset_version: Dataset version
            feature_version: Feature version
            callbacks: Optional callbacks (on_champion_trained, on_challenger_trained, etc.)

        Returns:
            TrainingRun object with results
        """
        run = TrainingRun(
            run_id=f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            start_time=datetime.now(timezone.utc).isoformat(),
            n_samples=len(y),
            n_features=X.shape[1],
            fraud_rate=float(y.mean()),
        )

        logger.info(
            "Starting training pipeline: {} samples, {} features, {:.3%} fraud rate",
            run.n_samples,
            run.n_features,
            run.fraud_rate,
        )

        try:
            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(X, y)
            run.train_samples = len(y_train)
            run.val_samples = len(y_val)
            run.test_samples = len(y_test)

            # Train champion
            champion_id, champion_metrics = self._train_champion(
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                feature_names,
                categorical_indices,
                dataset_version,
                feature_version,
            )
            run.champion_id = champion_id
            run.champion_metrics = champion_metrics

            if callbacks and "on_champion_trained" in callbacks:
                callbacks["on_champion_trained"](champion_id, champion_metrics)

            # Train challengers
            challenger_ids, challenger_metrics = self._train_challengers(
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                feature_names,
                categorical_indices,
                dataset_version,
                feature_version,
            )
            run.challenger_ids = challenger_ids
            run.challenger_metrics = challenger_metrics

            if callbacks and "on_challengers_trained" in callbacks:
                callbacks["on_challengers_trained"](challenger_ids, challenger_metrics)

            # Evaluate and compare
            evaluation = self._evaluate_models(
                champion_id,
                champion_metrics,
                challenger_ids,
                challenger_metrics,
            )
            run.evaluation_report_id = evaluation.get("report_id")
            run.recommended_champion = evaluation.get("recommended_champion")
            run.promotion_recommended = evaluation.get("promotion_recommended", False)

            run.status = "completed"
            run.end_time = datetime.now(timezone.utc).isoformat()
            run.duration_seconds = (
                datetime.fromisoformat(run.end_time)
                - datetime.fromisoformat(run.start_time)
            ).total_seconds()

            logger.info(
                "Training pipeline completed in {:.1f}s — champion: {}, recommended: {}",
                run.duration_seconds,
                run.champion_id,
                run.recommended_champion,
            )

        except Exception as exc:
            run.status = "failed"
            run.error = str(exc)
            run.end_time = datetime.now(timezone.utc).isoformat()
            logger.error("Training pipeline failed: {}", exc)
            raise

        finally:
            # Save run
            self._save_run(run)

        return run

    def _split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple:
        """Split data into train/val/test sets."""
        from sklearn.model_selection import train_test_split

        # First split: train+val vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_ratio,
            random_state=self.config.random_seed,
            stratify=y,
        )

        # Second split: train vs val
        val_ratio_adjusted = self.config.val_ratio / (
            self.config.train_ratio + self.config.val_ratio
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=val_ratio_adjusted,
            random_state=self.config.random_seed,
            stratify=y_train_val,
        )

        logger.info(
            "Data split: train={}, val={}, test={}",
            len(y_train),
            len(y_val),
            len(y_test),
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _train_champion(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[List[str]],
        categorical_indices: Optional[List[int]],
        dataset_version: str,
        feature_version: str,
    ) -> tuple[str, Dict[str, float]]:
        """Train the champion model."""
        logger.info("Training champion model (CatBoost)...")

        # Create and train champion
        champion = ChampionModel(
            feature_names=feature_names,
            categorical_features=categorical_indices or [],
            **self.config.champion_params,
        )

        start_time = time.time()
        champion.fit(
            X_train,
            y_train,
            feature_names=feature_names,
            categorical_indices=categorical_indices,
            calibration_method=self.config.champion_params.get(
                "calibration_method", "isotonic"
            ),
        )
        training_time = time.time() - start_time

        # Evaluate on test set
        metrics = champion.compute_metrics(y_test, champion.predict_proba(X_test))

        # Register in registry
        model_id = f"champion_{champion.model_version}"
        self.registry.register(
            model_id=model_id,
            version=champion.model_version,
            algorithm="catboost",
            training_date=champion.trained_at,
            dataset_version=dataset_version,
            feature_version=feature_version,
            metrics=metrics,
            hyperparameters=self.config.champion_params,
            calibration_method=self.config.champion_params.get(
                "calibration_method", "isotonic"
            ),
            description=f"Champion model trained on {len(y_train)} samples",
        )

        # Save model
        model_dir = self.config.output_dir / model_id
        champion.save(model_dir)

        logger.info(
            "Champion trained in {:.1f}s — PR-AUC: {:.4f}, F2: {:.4f}",
            training_time,
            metrics["pr_auc"],
            metrics["f2_score"],
        )

        return model_id, metrics

    def _train_challengers(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[List[str]],
        categorical_indices: Optional[List[int]],
        dataset_version: str,
        feature_version: str,
    ) -> tuple[List[str], Dict[str, Dict[str, float]]]:
        """Train all challenger models."""
        challenger_ids = []
        challenger_metrics = {}

        for algorithm in self.config.challenger_algorithms:
            logger.info("Training challenger: {}", algorithm)

            try:
                # Get challenger parameters
                params = self.config.challenger_params.get(algorithm, {})

                # Create and train challenger
                challenger = create_challenger(algorithm, **params)

                start_time = time.time()
                challenger.fit(
                    X_train,
                    y_train,
                    feature_names=feature_names,
                    categorical_indices=categorical_indices,
                    **params,
                )
                training_time = time.time() - start_time

                # Evaluate on test set
                metrics = challenger.compute_metrics(
                    X_test, y_test, fpr_threshold=self.config.fpr_threshold
                )

                # Register in registry
                model_id = f"challenger_{algorithm}_{challenger.model_version}"
                self.registry.register(
                    model_id=model_id,
                    version=challenger.version,
                    algorithm=algorithm,
                    training_date=challenger.trained_at,
                    dataset_version=dataset_version,
                    feature_version=feature_version,
                    metrics=metrics,
                    hyperparameters=params,
                    calibration_method=challenger.calibration_method,
                    description=f"{algorithm} challenger trained on {len(y_train)} samples",
                )

                # Save model
                model_dir = self.config.output_dir / model_id
                challenger.save(model_dir)

                challenger_ids.append(model_id)
                challenger_metrics[model_id] = metrics

                logger.info(
                    "{} challenger trained in {:.1f}s — PR-AUC: {:.4f}, F2: {:.4f}",
                    algorithm,
                    training_time,
                    metrics["pr_auc"],
                    metrics["f2_score"],
                )

            except Exception as exc:
                logger.error("Failed to train {} challenger: {}", algorithm, exc)
                continue

        return challenger_ids, challenger_metrics

    def _evaluate_models(
        self,
        champion_id: str,
        champion_metrics: Dict[str, float],
        challenger_ids: List[str],
        challenger_metrics: Dict[str, Dict[str, float]],
    ) -> Dict[str, Any]:
        """Evaluate and compare all models."""
        logger.info("Evaluating models...")

        # Find best challenger
        best_challenger_id = None
        best_pr_auc = 0.0

        for challenger_id, metrics in challenger_metrics.items():
            if metrics.get("pr_auc", 0) > best_pr_auc:
                best_pr_auc = metrics["pr_auc"]
                best_challenger_id = challenger_id

        # Check promotion criteria
        if best_challenger_id:
            should_promote, criteria_met, criteria_failed = (
                self.promoter.should_promote(
                    best_challenger_id,
                    champion_metrics,
                    challenger_metrics[best_challenger_id],
                )
            )

            return {
                "champion_id": champion_id,
                "best_challenger_id": best_challenger_id,
                "recommended_champion": (
                    best_challenger_id if should_promote else champion_id
                ),
                "promotion_recommended": should_promote,
                "criteria_met": criteria_met,
                "criteria_failed": criteria_failed,
            }

        return {
            "champion_id": champion_id,
            "best_challenger_id": None,
            "recommended_champion": champion_id,
            "promotion_recommended": False,
        }

    def _save_run(self, run: TrainingRun) -> None:
        """Save training run to disk."""
        filepath = self.config.output_dir / f"{run.run_id}.json"

        with open(filepath, "w") as f:
            json.dump(run.to_dict(), f, indent=2, default=str)

        logger.info("Training run saved to {}", filepath)

    def load_run(self, run_id: str) -> Optional[TrainingRun]:
        """Load a training run from disk."""
        filepath = self.config.output_dir / f"{run_id}.json"

        if not filepath.exists():
            return None

        with open(filepath, "r") as f:
            data = json.load(f)

        return TrainingRun(**data)

    def list_runs(self, limit: int = 50) -> List[TrainingRun]:
        """List recent training runs."""
        runs = []

        for filepath in sorted(self.config.output_dir.glob("run_*.json"), reverse=True)[
            :limit
        ]:
            with open(filepath, "r") as f:
                data = json.load(f)
            runs.append(TrainingRun(**data))

        return runs


def create_pipeline(config: TrainingConfig = None) -> TrainingPipeline:
    """
    Factory function to create a training pipeline.

    Args:
        config: Training configuration

    Returns:
        TrainingPipeline instance
    """
    return TrainingPipeline(config=config)


def run_training(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    categorical_indices: Optional[List[int]] = None,
    dataset_version: str = "1.0.0",
    feature_version: str = "1.0.0",
    output_dir: str = None,
) -> TrainingRun:
    """
    Convenience function to run training pipeline.

    Args:
        X: Feature matrix
        y: Labels
        feature_names: Feature names
        categorical_indices: Categorical feature indices
        dataset_version: Dataset version
        feature_version: Feature version
        output_dir: Output directory

    Returns:
        TrainingRun object
    """
    config = TrainingConfig()
    if output_dir:
        config.output_dir = Path(output_dir)

    pipeline = TrainingPipeline(config=config)
    return pipeline.run(
        X,
        y,
        feature_names=feature_names,
        categorical_indices=categorical_indices,
        dataset_version=dataset_version,
        feature_version=feature_version,
    )
