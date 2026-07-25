"""
RiskLens — Model Evaluation Framework
Comprehensive evaluation and comparison of challenger models.
Supports the Champion-Challenger architecture.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
from loguru import logger


@dataclass
class EvaluationMetrics:
    """Complete evaluation metrics for a model."""

    model_id: str
    algorithm: str
    version: str

    # Core metrics
    pr_auc: float = 0.0
    roc_auc: float = 0.0
    f2_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0

    # Fraud-specific metrics
    fpr: float = 0.0
    tpr: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    # Calibration metrics
    calibration_error: float = 0.0
    brier_score: float = 0.0

    # Threshold analysis
    thresholds: Dict[str, float] = field(default_factory=dict)

    # Latency (inference time)
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Metadata
    evaluation_date: str = ""
    dataset_size: int = 0
    fraud_rate: float = 0.0

    def __post_init__(self):
        if not self.evaluation_date:
            self.evaluation_date = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "algorithm": self.algorithm,
            "version": self.version,
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "f2_score": self.f2_score,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "fpr": self.fpr,
            "tpr": self.tpr,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "calibration_error": self.calibration_error,
            "brier_score": self.brier_score,
            "thresholds": self.thresholds,
            "avg_latency_ms": self.avg_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "evaluation_date": self.evaluation_date,
            "dataset_size": self.dataset_size,
            "fraud_rate": self.fraud_rate,
        }


@dataclass
class EvaluationReport:
    """Complete evaluation report comparing multiple models."""

    report_id: str
    champion_id: str
    challenger_ids: List[str]

    # Champion metrics
    champion_metrics: Optional[EvaluationMetrics] = None

    # Challenger metrics
    challenger_metrics: Dict[str, EvaluationMetrics] = field(default_factory=dict)

    # Comparison results
    comparisons: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Recommendations
    recommended_champion: Optional[str] = None
    promotion_recommended: bool = False
    promotion_reasons: List[str] = field(default_factory=list)

    # Metadata
    evaluation_date: str = ""
    evaluation_duration_seconds: float = 0.0

    def __post_init__(self):
        if not self.evaluation_date:
            self.evaluation_date = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "champion_id": self.champion_id,
            "challenger_ids": self.challenger_ids,
            "champion_metrics": (
                self.champion_metrics.to_dict() if self.champion_metrics else None
            ),
            "challenger_metrics": {
                k: v.to_dict() for k, v in self.challenger_metrics.items()
            },
            "comparisons": self.comparisons,
            "recommended_champion": self.recommended_champion,
            "promotion_recommended": self.promotion_recommended,
            "promotion_reasons": self.promotion_reasons,
            "evaluation_date": self.evaluation_date,
            "evaluation_duration_seconds": self.evaluation_duration_seconds,
        }


class ModelEvaluator:
    """
    Model evaluation framework for Champion-Challenger architecture.

    Features:
    - Comprehensive metric computation
    - Side-by-side model comparison
    - Promotion recommendation
    - Threshold analysis
    - Latency profiling
    """

    def __init__(self, output_dir: Path = None):
        """
        Initialize evaluator.

        Args:
            output_dir: Directory to save evaluation reports
        """
        self.output_dir = (
            Path(output_dir) if output_dir else Path("models/supervised/evaluations")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_model(
        self,
        model,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_id: str = "",
        algorithm: str = "",
        version: str = "",
        include_latency: bool = True,
        n_latency_samples: int = 1000,
    ) -> EvaluationMetrics:
        """
        Evaluate a single model.

        Args:
            model: Model to evaluate (must have predict_proba or score method)
            X_val: Validation features
            y_val: Validation labels
            model_id: Model identifier
            algorithm: Algorithm name
            version: Model version
            include_latency: Whether to measure inference latency
            n_latency_samples: Number of samples for latency measurement

        Returns:
            EvaluationMetrics object
        """
        from sklearn.metrics import (
            average_precision_score,
            roc_auc_score,
            precision_score,
            recall_score,
            fbeta_score,
            f1_score,
            accuracy_score,
            confusion_matrix,
            brier_score_loss,
        )
        from sklearn.calibration import calibration_curve

        logger.info("Evaluating model {} on {} samples...", model_id, len(y_val))

        # Get predictions
        if hasattr(model, "predict_proba"):
            raw = model.predict_proba(X_val)
            probs = raw[:, 1] if raw.ndim == 2 else raw
        elif hasattr(model, "score"):
            probs = model.score(X_val)
        else:
            raise ValueError("Model must have predict_proba or score method")

        # Binary predictions
        y_pred = (probs >= 0.5).astype(int)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

        # Core metrics
        pr_auc = float(average_precision_score(y_val, probs))
        roc_auc_val = float(roc_auc_score(y_val, probs))
        precision = float(precision_score(y_val, y_pred, zero_division=0))
        recall_val = float(recall_score(y_val, y_pred, zero_division=0))
        f1 = float(f1_score(y_val, y_pred, zero_division=0))
        f2 = float(fbeta_score(y_val, y_pred, beta=2, zero_division=0))
        accuracy = float(accuracy_score(y_val, y_pred))

        # Fraud-specific metrics
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        # Calibration error
        fraction_pos, mean_predicted = calibration_curve(y_val, probs, n_bins=10)
        calibration_error = float(np.mean(np.abs(fraction_pos - mean_predicted)))

        # Brier score
        brier = float(brier_score_loss(y_val, probs))

        # Threshold analysis
        thresholds = self._analyze_thresholds(y_val, probs)

        # Latency measurement
        avg_latency = 0.0
        p99_latency = 0.0
        if include_latency:
            avg_latency, p99_latency = self._measure_latency(
                model, X_val, n_latency_samples
            )

        metrics = EvaluationMetrics(
            model_id=model_id,
            algorithm=algorithm,
            version=version,
            pr_auc=pr_auc,
            roc_auc=roc_auc_val,
            f2_score=f2,
            precision=precision,
            recall=recall_val,
            f1=f1,
            accuracy=accuracy,
            fpr=fpr,
            tpr=tpr,
            true_positives=int(tp),
            false_positives=int(fp),
            true_negatives=int(tn),
            false_negatives=int(fn),
            calibration_error=calibration_error,
            brier_score=brier,
            thresholds=thresholds,
            avg_latency_ms=avg_latency,
            p99_latency_ms=p99_latency,
            dataset_size=len(y_val),
            fraud_rate=float(y_val.mean()),
        )

        logger.info(
            "Model {} evaluated — PR-AUC: {:.4f}, ROC-AUC: {:.4f}, F2: {:.4f}, FPR: {:.4f}",
            model_id,
            pr_auc,
            roc_auc_val,
            f2,
            fpr,
        )

        return metrics

    def _analyze_thresholds(
        self,
        y_true: np.ndarray,
        probs: np.ndarray,
        threshold_range: np.ndarray = None,
    ) -> Dict[str, float]:
        """Analyze performance at different thresholds."""
        if threshold_range is None:
            threshold_range = np.arange(0.1, 1.0, 0.1)

        from sklearn.metrics import precision_score, recall_score, fbeta_score

        best_f2 = 0.0
        best_f2_threshold = 0.5

        for threshold in threshold_range:
            y_pred = (probs >= threshold).astype(int)
            f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

            if f2 > best_f2:
                best_f2 = f2
                best_f2_threshold = threshold

        # FPR at threshold (for fraud detection)
        fpr_threshold = 0.1
        y_pred_fpr = (probs >= fpr_threshold).astype(int)
        from sklearn.metrics import confusion_matrix

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_fpr).ravel()
        fpr_at_threshold = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        return {
            "best_f2_threshold": float(best_f2_threshold),
            "best_f2_score": float(best_f2),
            "fpr_at_0.1_threshold": fpr_at_threshold,
        }

    def _measure_latency(
        self,
        model,
        X: np.ndarray,
        n_samples: int = 1000,
    ) -> tuple[float, float]:
        """Measure model inference latency."""
        import time

        # Sample if needed
        if len(X) > n_samples:
            idx = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X

        # Warmup
        for _ in range(min(10, len(X_sample))):
            if hasattr(model, "predict_proba"):
                _ = model.predict_proba(X_sample[:1])
            elif hasattr(model, "score"):
                _ = model.score(X_sample[:1])

        # Measure
        latencies = []
        for i in range(len(X_sample)):
            start = time.perf_counter()
            if hasattr(model, "predict_proba"):
                _ = model.predict_proba(X_sample[i : i + 1])
            elif hasattr(model, "score"):
                _ = model.score(X_sample[i : i + 1])
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        avg_latency = float(np.mean(latencies))
        p99_latency = float(np.percentile(latencies, 99))

        return avg_latency, p99_latency

    def compare_models(
        self,
        champion_metrics: EvaluationMetrics,
        challenger_metrics: List[EvaluationMetrics],
        promotion_threshold: float = 0.01,
        max_fpr: float = 0.01,
        max_calibration_error: float = 0.05,
    ) -> EvaluationReport:
        """
        Compare champion against challengers.

        Args:
            champion_metrics: Champion model metrics
            challenger_metrics: List of challenger model metrics
            promotion_threshold: Minimum PR-AUC improvement required
            max_fpr: Maximum allowed FPR
            max_calibration_error: Maximum allowed calibration error

        Returns:
            EvaluationReport with recommendations
        """
        report = EvaluationReport(
            report_id=f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            champion_id=champion_metrics.model_id,
            challenger_ids=[m.model_id for m in challenger_metrics],
            champion_metrics=champion_metrics,
        )

        # Compare each challenger against champion
        for challenger in challenger_metrics:
            comparison = self._compare_pair(
                champion_metrics,
                challenger,
                promotion_threshold,
                max_fpr,
                max_calibration_error,
            )
            report.comparisons[challenger.model_id] = comparison

            # Check if challenger is better
            if comparison["recommended_for_promotion"]:
                if (
                    report.recommended_champion is None
                    or challenger.pr_auc
                    > report.challenger_metrics.get(
                        report.recommended_champion, EvaluationMetrics("", "", "")
                    ).pr_auc
                ):
                    report.recommended_champion = challenger.model_id
                    report.promotion_recommended = True
                    report.promotion_reasons = comparison["promotion_reasons"]

            report.challenger_metrics[challenger.model_id] = challenger

        return report

    def _compare_pair(
        self,
        champion: EvaluationMetrics,
        challenger: EvaluationMetrics,
        promotion_threshold: float,
        max_fpr: float,
        max_calibration_error: float,
    ) -> Dict[str, Any]:
        """Compare a single challenger against the champion."""
        pr_auc_improvement = challenger.pr_auc - champion.pr_auc
        roc_auc_improvement = challenger.roc_auc - champion.roc_auc
        f2_improvement = challenger.f2_score - champion.f2_score

        # Check promotion criteria
        criteria_met = []
        criteria_failed = []

        # PR-AUC improvement
        if pr_auc_improvement > promotion_threshold:
            criteria_met.append("pr_auc_improvement")
        else:
            criteria_failed.append(
                f"pr_auc_improvement ({pr_auc_improvement:.4f} <= {promotion_threshold:.4f})"
            )

        # FPR check
        if challenger.fpr <= max_fpr:
            criteria_met.append("fpr_within_limit")
        else:
            criteria_failed.append(f"fpr ({challenger.fpr:.4f} > {max_fpr:.4f})")

        # Calibration check
        if challenger.calibration_error <= max_calibration_error:
            criteria_met.append("calibration_within_limit")
        else:
            criteria_failed.append(
                f"calibration_error ({challenger.calibration_error:.4f} > {max_calibration_error:.4f})"
            )

        # Latency check (challenger shouldn't be significantly slower)
        latency_ratio = challenger.avg_latency_ms / max(champion.avg_latency_ms, 0.001)
        if latency_ratio <= 2.0:  # Allow up to 2x slower
            criteria_met.append("latency_acceptable")
        else:
            criteria_failed.append(f"latency ({latency_ratio:.1f}x slower)")

        recommended = len(criteria_failed) == 0

        return {
            "champion_id": champion.model_id,
            "challenger_id": challenger.model_id,
            "pr_auc_improvement": pr_auc_improvement,
            "roc_auc_improvement": roc_auc_improvement,
            "f2_improvement": f2_improvement,
            "champion_pr_auc": champion.pr_auc,
            "challenger_pr_auc": challenger.pr_auc,
            "champion_fpr": champion.fpr,
            "challenger_fpr": challenger.fpr,
            "champion_calibration_error": champion.calibration_error,
            "challenger_calibration_error": challenger.calibration_error,
            "latency_ratio": latency_ratio,
            "criteria_met": criteria_met,
            "criteria_failed": criteria_failed,
            "recommended_for_promotion": recommended,
            "promotion_reasons": [
                f"PR-AUC improvement: {pr_auc_improvement:+.4f}",
                f"FPR: {challenger.fpr:.4f} (limit: {max_fpr:.4f})",
                f"Calibration error: {challenger.calibration_error:.4f} (limit: {max_calibration_error:.4f})",
            ],
        }

    def save_report(
        self,
        report: EvaluationReport,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Save evaluation report to disk.

        Args:
            report: EvaluationReport to save
            filename: Optional filename (auto-generated if None)

        Returns:
            Path to saved report
        """
        if filename is None:
            filename = f"{report.report_id}.json"

        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)

        logger.info("Evaluation report saved to {}", filepath)
        return filepath

    def load_report(self, filepath: Path) -> EvaluationReport:
        """Load evaluation report from disk."""
        with open(filepath, "r") as f:
            data = json.load(f)

        # Reconstruct report (simplified)
        report = EvaluationReport(
            report_id=data["report_id"],
            champion_id=data["champion_id"],
            challenger_ids=data["challenger_ids"],
            recommended_champion=data.get("recommended_champion"),
            promotion_recommended=data.get("promotion_recommended", False),
            promotion_reasons=data.get("promotion_reasons", []),
            evaluation_date=data.get("evaluation_date", ""),
            evaluation_duration_seconds=data.get("evaluation_duration_seconds", 0.0),
        )

        # Reconstruct metrics
        if data.get("champion_metrics"):
            report.champion_metrics = EvaluationMetrics(**data["champion_metrics"])

        for model_id, metrics_data in data.get("challenger_metrics", {}).items():
            report.challenger_metrics[model_id] = EvaluationMetrics(**metrics_data)

        report.comparisons = data.get("comparisons", {})

        return report

    def generate_leaderboard(
        self,
        metrics_list: List[EvaluationMetrics],
        sort_by: str = "pr_auc",
        ascending: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Generate a leaderboard of models.

        Args:
            metrics_list: List of EvaluationMetrics objects
            sort_by: Metric to sort by
            ascending: Sort order

        Returns:
            List of leaderboard entries
        """
        # Sort metrics
        sorted_metrics = sorted(
            metrics_list, key=lambda x: getattr(x, sort_by, 0), reverse=not ascending
        )

        leaderboard = []
        for rank, metrics in enumerate(sorted_metrics, 1):
            leaderboard.append(
                {
                    "rank": rank,
                    "model_id": metrics.model_id,
                    "algorithm": metrics.algorithm,
                    "version": metrics.version,
                    "pr_auc": metrics.pr_auc,
                    "roc_auc": metrics.roc_auc,
                    "f2_score": metrics.f2_score,
                    "fpr": metrics.fpr,
                    "calibration_error": metrics.calibration_error,
                    "avg_latency_ms": metrics.avg_latency_ms,
                    "dataset_size": metrics.dataset_size,
                    "fraud_rate": metrics.fraud_rate,
                }
            )

        return leaderboard
