"""
FraudTrap — MLOps Layer
Drift detection (PSI + performance), model registry wrapper,
and automated retraining trigger.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import mlflow
from loguru import logger

from config.settings import get_settings

settings = get_settings()


# ── Population Stability Index ────────────────────────────────────────────────

def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    buckets: int = 10,
    epsilon: float = 1e-8,
) -> float:
    """
    PSI < 0.10 → stable
    PSI 0.10–0.20 → moderate drift
    PSI > 0.20 → significant drift → trigger retrain
    """
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[0]  = -np.inf
    breakpoints[-1] =  np.inf

    def bucket_pcts(arr):
        counts = np.histogram(arr, bins=breakpoints)[0]
        pcts = counts / (len(arr) + epsilon)
        return np.clip(pcts, epsilon, None)

    e_pct = bucket_pcts(expected)
    a_pct = bucket_pcts(actual)
    psi = np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))
    return float(psi)


# ── Drift detector ────────────────────────────────────────────────────────────

@dataclass
class DriftReport:
    tenant_id: str
    evaluated_at: str
    feature_psi: dict[str, float] = field(default_factory=dict)
    score_psi: float = 0.0
    performance_delta: float = 0.0
    drift_detected: bool = False
    retrain_recommended: bool = False
    alert_level: str = "ok"       # ok | warning | critical


class DriftDetector:
    """
    Monitors feature distributions and model performance for drift.
    Writes reports to MLflow and optionally triggers retraining.
    """

    def __init__(self, baseline_path: Optional[Path] = None):
        self._baseline: Optional[dict] = None
        if baseline_path and baseline_path.exists():
            self._load_baseline(baseline_path)

    def _load_baseline(self, path: Path) -> None:
        with open(path) as f:
            self._baseline = json.load(f)
        logger.info("Drift baseline loaded from {}", path)

    def save_baseline(
        self,
        feature_distributions: dict[str, list],
        score_distribution: list,
        baseline_metrics: dict,
        path: Path,
    ) -> None:
        payload = {
            "feature_distributions": feature_distributions,
            "score_distribution":    score_distribution,
            "baseline_metrics":      baseline_metrics,
            "saved_at":              datetime.now(timezone.utc).isoformat(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f)
        self._baseline = payload
        logger.info("Drift baseline saved → {}", path)

    def evaluate(
        self,
        tenant_id: str,
        current_feature_distributions: dict[str, list],
        current_scores: list,
        current_metrics: Optional[dict] = None,
    ) -> DriftReport:

        report = DriftReport(
            tenant_id=tenant_id,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
        )

        if not self._baseline:
            logger.warning("No drift baseline set for tenant={}; skipping", tenant_id)
            return report

        # ── Feature PSI ───────────────────────────────────────────────────────
        for feature, current_vals in current_feature_distributions.items():
            baseline_vals = self._baseline["feature_distributions"].get(feature)
            if baseline_vals:
                psi = compute_psi(
                    np.array(baseline_vals),
                    np.array(current_vals),
                )
                report.feature_psi[feature] = round(psi, 4)

        # ── Score PSI ─────────────────────────────────────────────────────────
        report.score_psi = round(
            compute_psi(
                np.array(self._baseline["score_distribution"]),
                np.array(current_scores),
            ),
            4,
        )

        # ── Performance delta ─────────────────────────────────────────────────
        if current_metrics and "f1" in current_metrics:
            baseline_f1 = self._baseline["baseline_metrics"].get("f1", 0.0)
            report.performance_delta = round(
                baseline_f1 - current_metrics.get("f1", baseline_f1), 4
            )

        # ── Alert level ───────────────────────────────────────────────────────
        max_psi = max(report.feature_psi.values(), default=0.0)
        critical_features = sum(
            1 for psi in report.feature_psi.values()
            if psi > settings.psi_drift_threshold
        )
        perf_drop = report.performance_delta > settings.performance_drop_threshold

        if max_psi > settings.psi_drift_threshold * 2 or perf_drop:
            report.alert_level = "critical"
            report.drift_detected = True
            report.retrain_recommended = True
        elif max_psi > settings.psi_drift_threshold or critical_features > 3:
            report.alert_level = "warning"
            report.drift_detected = True
            report.retrain_recommended = max_psi > settings.psi_drift_threshold * 1.5

        logger.info(
            "Drift report tenant={}: alert={}, max_psi={:.4f}, score_psi={:.4f}, "
            "perf_delta={:.4f}, retrain={}",
            tenant_id, report.alert_level, max_psi,
            report.score_psi, report.performance_delta, report.retrain_recommended,
        )
        return report


# ── Model registry ────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Wraps MLflow model registry with champion/challenger logic.
    New models are registered as 'challenger'; promoted to 'champion'
    only after passing evaluation gates.
    """

    def __init__(self):
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        self._client = mlflow.MlflowClient()

    def register_model(
        self,
        run_id: str,
        model_name: str,
        metrics: dict,
        phase: str,
    ) -> str:
        """Register a trained model as a challenger."""
        model_uri = f"runs:/{run_id}/model"
        try:
            mv = mlflow.register_model(model_uri, model_name)
            self._client.set_model_version_tag(
                model_name, mv.version, "phase", phase
            )
            self._client.set_model_version_tag(
                model_name, mv.version, "role", "challenger"
            )
            for k, v in metrics.items():
                self._client.set_model_version_tag(model_name, mv.version, k, str(v))
            logger.info(
                "Registered model={} version={} as challenger", model_name, mv.version
            )
            return mv.version
        except Exception as exc:
            logger.warning("MLflow registration failed (non-fatal): {}", exc)
            return "local"

    def promote_to_champion(
        self,
        model_name: str,
        challenger_version: str,
        champion_metrics: dict,
        challenger_metrics: dict,
        metric_key: str = "pr_auc",
    ) -> bool:
        """
        Promotes challenger to Production if it beats the current champion.
        Returns True if promotion happened.
        """
        champ_score = champion_metrics.get(metric_key, 0.0)
        chal_score  = challenger_metrics.get(metric_key, 0.0)

        if chal_score > champ_score:
            try:
                # Archive current production model
                prod_versions = self._client.get_model_version_by_alias(
                    model_name, "champion"
                )
                if prod_versions:
                    self._client.set_model_version_tag(
                        model_name, prod_versions.version, "role", "archived"
                    )

                self._client.set_registered_model_alias(
                    model_name, "champion", challenger_version
                )
                logger.info(
                    "Promoted model={} version={} to champion "
                    "({}={:.4f} > {:.4f})",
                    model_name, challenger_version, metric_key,
                    chal_score, champ_score,
                )
                return True
            except Exception as exc:
                logger.warning("Promotion failed: {}", exc)
        else:
            logger.info(
                "Challenger ({}={:.4f}) did not beat champion ({:.4f}). "
                "Keeping champion.",
                metric_key, chal_score, champ_score,
            )
        return False


# ── Retraining scheduler ──────────────────────────────────────────────────────

class RetrainingScheduler:
    """
    Manages scheduled and drift-triggered retraining.
    In production, this is called by a Kubernetes CronJob or Airflow DAG.
    """

    def __init__(self, training_pipeline, drift_detector: DriftDetector):
        self.pipeline = training_pipeline
        self.drift_detector = drift_detector
        self.registry = ModelRegistry()

    def run_scheduled(self, tenant_id: str, state) -> None:
        logger.info("Scheduled retrain: tenant={}", tenant_id)
        self._retrain(tenant_id, state, trigger="scheduled")

    def run_drift_triggered(
        self,
        tenant_id: str,
        state,
        drift_report: DriftReport,
    ) -> None:
        if not drift_report.retrain_recommended:
            return
        logger.info(
            "Drift-triggered retrain: tenant={}, alert={}",
            tenant_id, drift_report.alert_level,
        )
        self._retrain(tenant_id, state, trigger="drift")

    def _retrain(self, tenant_id: str, state, trigger: str) -> None:
        try:
            updated_state = self.pipeline.run(tenant_id, state)
            logger.info(
                "Retrain complete: tenant={}, trigger={}, phase={}, metrics={}",
                tenant_id, trigger,
                updated_state.current_phase.value,
                updated_state.metrics,
            )
        except Exception as exc:
            logger.error("Retrain failed: tenant={}, error={}", tenant_id, exc)
