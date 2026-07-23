#!/usr/bin/env python3
"""
FraudTrap — Daily Metrics Rollup Job
Runs at 02:00 UTC to compute and persist daily model performance and drift metrics.
"""

from __future__ import annotations
import sys
import os
from datetime import datetime, timezone, timedelta, date
from typing import Optional
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clickhouse_driver import Client
from loguru import logger

from monitoring.drift import (
    compute_all_feature_drift,
    compute_embedding_drift,
    compute_concept_drift,
    DEFAULT_MONITORED_FEATURES,
)

from config.settings import get_settings

settings = get_settings()


def get_ch_client() -> Client:
    """Get ClickHouse client."""
    return Client(
        host=settings.clickhouse_host,
        port=settings.clickhouse_port,
        database=settings.clickhouse_database,
        user=settings.clickhouse_user,
        password=settings.clickhouse_password or "",
    )


def get_active_tenants() -> list[str]:
    """Get list of active tenants from recent scores."""
    ch = get_ch_client()
    # Get tenants with recent activity
    rows = ch.execute("""
        SELECT DISTINCT tenant_id 
        FROM fraudtrap:recent_scores 
        WHERE scored_at >= now() - INTERVAL 2 DAY
    """)
    return [r[0] for r in rows]


def fetch_recent_scores(ch: Client, tenant_id: str, days: int = 2) -> pd.DataFrame:
    """Fetch recent scored transactions for a tenant."""
    query = """
        SELECT 
            tenant_id,
            transaction_id,
            risk_score,
            decision,
            model_phase,
            model_version,
            latency_ms,
            scored_at,
            is_fraud,
            {}
        FROM fraudtrap:recent_scores
        WHERE tenant_id = %(tenant)s
          AND scored_at >= now() - INTERVAL %(days)s DAY
        ORDER BY scored_at
    """.format(", ".join(DEFAULT_MONITORED_FEATURES))

    rows = ch.execute(query, {"tenant": tenant_id, "days": days})
    if not rows:
        return pd.DataFrame()

    cols = [
        "tenant_id",
        "transaction_id",
        "risk_score",
        "decision",
        "model_phase",
        "model_version",
        "latency_ms",
        "scored_at",
        "is_fraud",
    ] + DEFAULT_MONITORED_FEATURES
    return pd.DataFrame(rows, columns=cols)


def fetch_labels(ch: Client, tenant_id: str, days: int = 2) -> pd.DataFrame:
    """Fetch labels for concept drift detection."""
    # In production, this would come from a labels table
    # For now, use recent_scores which has simulated labels
    query = """
        SELECT transaction_id, is_fraud, risk_score, scored_at
        FROM fraudtrap:recent_scores
        WHERE tenant_id = %(tenant)s
          AND scored_at >= now() - INTERVAL %(days)s DAY
          AND is_fraud IS NOT NULL
    """
    rows = ch.execute(query, {"tenant": tenant_id, "days": days})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(
        rows, columns=["transaction_id", "is_fraud", "risk_score", "scored_at"]
    )


def compute_daily_model_metrics(
    df: pd.DataFrame,
    drift_results: dict,
    model_version: str,
    model_phase: str,
    bucket_date: date,
) -> dict:
    """Compute daily model performance metrics."""
    if df.empty:
        return None

    # Filter rows with labels
    labeled = df[df["is_fraud"].notna()].copy()
    if len(labeled) < 50:
        logger.warning("Insufficient labeled data for metrics: {}", len(labeled))
        return None

    y_true = labeled["is_fraud"].astype(int).values
    y_score = labeled["risk_score"].values

    # Compute metrics
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
        recall_score,
        precision_score,
        fbeta_score,
    )

    # Discrimination
    auc_pr = average_precision_score(y_true, y_score)
    auc_roc = roc_auc_score(y_true, y_score)

    # Threshold-based (BLOCK at 0.85)
    y_pred_block = (y_score >= 0.85).astype(int)
    recall = recall_score(y_true, y_pred_block, zero_division=0)
    precision = precision_score(y_true, y_pred_block, zero_division=0)
    f2 = fbeta_score(y_true, y_pred_block, beta=2, zero_division=0)

    # Fraud capture rate (blocked fraud / total fraud)
    total_fraud = y_true.sum()
    blocked_fraud = ((y_pred_block == 1) & (y_true == 1)).sum()
    fraud_capture = blocked_fraud / max(total_fraud, 1)

    # Operational rates
    review_rate = (df["decision"] == "REVIEW").mean()
    fpr = ((y_pred_block == 1) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1)

    # Latency
    avg_latency = df["latency_ms"].mean()
    p95_latency = df["latency_ms"].quantile(0.95)

    # Volume
    total_scored = len(df)
    total_blocked = (df["decision"] == "BLOCK").sum()
    total_reviewed = (df["decision"] == "REVIEW").sum()
    total_approved = (df["decision"] == "APPROVE").sum()

    # Drift summary
    max_psi = max((r.psi for r in drift_results.values()), default=0.0)
    max_kl = max((r.kl_divergence for r in drift_results.values()), default=0.0)
    features_drift = sum(1 for r in drift_results.values() if r.drift_detected)

    return {
        "tenant_id": df["tenant_id"].iloc[0],
        "model_version": model_version,
        "model_phase": model_phase,
        "bucket_date": bucket_date,
        "auc_pr": float(auc_pr),
        "auc_roc": float(auc_roc),
        "recall": float(recall),
        "precision": float(precision),
        "f2_score": float(f2),
        "fraud_capture_rate": float(fraud_capture),
        "review_rate": float(review_rate),
        "fpr": float(fpr),
        "avg_latency_ms": float(avg_latency),
        "p95_latency_ms": float(p95_latency),
        "total_scored": int(total_scored),
        "total_blocked": int(total_blocked),
        "total_reviewed": int(total_reviewed),
        "total_approved": int(total_approved),
        "max_psi": float(max_psi),
        "max_kl": float(max_kl),
        "features_with_drift": int(features_drift),
    }


def insert_drift_metrics(
    ch: Client,
    drift_results: dict,
    tenant_id: str,
    bucket_date: date,
    bucket_hour: int = 0,
) -> None:
    """Insert drift metrics to ClickHouse."""
    if not drift_results:
        return

    data = []
    for feature, result in drift_results.items():
        data.append(
            {
                "tenant_id": tenant_id,
                "feature": feature,
                "metric_type": "psi",
                "value": result.psi,
                "bucket_date": bucket_date,
                "bucket_hour": bucket_hour,
                "drift_detected": 1 if result.drift_detected else 0,
            }
        )
        data.append(
            {
                "tenant_id": tenant_id,
                "feature": feature,
                "metric_type": "kl",
                "value": result.kl_divergence,
                "bucket_date": bucket_date,
                "bucket_hour": bucket_hour,
                "drift_detected": 1 if result.drift_detected else 0,
            }
        )

    ch.execute("INSERT INTO drift_metrics_hourly VALUES", data)


def insert_concept_drift(
    ch: Client, concept_result, tenant_id: str, bucket_date: date
) -> None:
    """Insert concept drift metrics."""
    if concept_result is None:
        return

    ch.execute(
        """
        INSERT INTO concept_drift_daily VALUES
        """,
        [
            {
                "tenant_id": tenant_id,
                "bucket_date": bucket_date,
                "label_rate_baseline": concept_result.label_rate_baseline,
                "label_rate_current": concept_result.label_rate_current,
                "rate_change": concept_result.rate_change,
                "drift_detected": 1 if concept_result.drift_detected else 0,
                "prediction_rate_baseline": concept_result.prediction_rate_baseline,
                "prediction_rate_current": concept_result.prediction_rate_current,
            }
        ],
    )


def insert_embedding_drift(
    ch: Client, embed_result, tenant_id: str, bucket_date: date, bucket_hour: int = 0
) -> None:
    """Insert embedding drift metrics."""
    if embed_result is None:
        return

    ch.execute(
        """
        INSERT INTO embedding_drift_hourly VALUES
        """,
        [
            {
                "tenant_id": tenant_id,
                "bucket_date": bucket_date,
                "bucket_hour": bucket_hour,
                "centroid_distance": embed_result.centroid_distance,
                "max_distance": embed_result.max_distance,
                "mean_distance": embed_result.mean_distance,
                "drift_detected": 1 if embed_result.drift_detected else 0,
            }
        ],
    )


def insert_daily_metrics(ch: Client, metrics: dict) -> None:
    """Insert daily model performance metrics."""
    if metrics is None:
        return

    ch.execute(
        """
        INSERT INTO model_performance_daily VALUES
        """,
        [metrics],
    )


def process_tenant(
    ch: Client, tenant_id: str, bucket_date: date, bucket_hour: int = 0
) -> None:
    """Process a single tenant's daily metrics."""
    logger.info(
        "Processing daily metrics for tenant={} date={}", tenant_id, bucket_date
    )

    try:
        # Fetch recent scores
        df = fetch_recent_scores(ch, tenant_id, days=2)
        if df.empty:
            logger.warning("No scores for tenant={}, skipping", tenant_id)
            return

        # Get current model info
        model_version = (
            df["model_version"].mode().iloc[0]
            if not df["model_version"].isna().all()
            else "unknown"
        )
        model_phase = (
            df["model_phase"].mode().iloc[0]
            if not df["model_phase"].isna().all()
            else "UNKNOWN"
        )

        # Split into baseline (older) and current (newer) for drift
        mid = len(df) // 2
        baseline_df = df.iloc[:mid].copy()
        current_df = df.iloc[mid:].copy()

        # Compute feature drift
        drift_results = compute_all_feature_drift(
            baseline_df=baseline_df,
            current_df=current_df,
            feature_list=DEFAULT_MONITORED_FEATURES,
            psi_threshold=0.1,
            kl_threshold=0.1,
        )

        # Compute embedding drift (if GNN embeddings available)
        # For now, skip - requires GNN embeddings to be stored
        embed_result = None

        # Compute concept drift
        labels_df = fetch_labels(ch, tenant_id, days=2)
        concept_result = None
        if not labels_df.empty:
            mid_labels = len(labels_df) // 2
            base_labels = labels_df.iloc[:mid_labels]["is_fraud"].values
            curr_labels = labels_df.iloc[mid_labels:]["is_fraud"].values
            base_preds = labels_df.iloc[:mid_labels]["risk_score"].values
            curr_preds = labels_df.iloc[mid_labels:]["risk_score"].values

            concept_result = compute_concept_drift(
                tenant_id=tenant_id,
                baseline_labels=base_labels,
                current_labels=curr_labels,
                baseline_predictions=base_preds,
                current_predictions=curr_preds,
                threshold=0.2,
            )

        # Insert drift metrics
        insert_drift_metrics(ch, drift_results, tenant_id, bucket_date, bucket_hour)
        insert_concept_drift(ch, concept_result, tenant_id, bucket_date)
        insert_embedding_drift(ch, embed_result, tenant_id, bucket_date, bucket_hour)

        # Compute and insert daily model metrics
        metrics = compute_daily_model_metrics(
            df=df,
            drift_results=drift_results,
            model_version=model_version,
            model_phase=model_phase,
            bucket_date=bucket_date,
        )
        insert_daily_metrics(ch, metrics)

        logger.info("Completed daily metrics for tenant={}", tenant_id)

    except Exception as exc:
        logger.exception("Failed processing tenant={}: {}", tenant_id, exc)


def main():
    """Main rollup job."""
    import argparse

    parser = argparse.ArgumentParser(description="Daily metrics rollup")
    parser.add_argument("--date", help="Date to process (YYYY-MM-DD)", default=None)
    parser.add_argument("--tenant", help="Specific tenant to process", default=None)
    parser.add_argument("--hour", type=int, help="Hour bucket (0-23)", default=0)
    args = parser.parse_args()

    # Determine date
    if args.date:
        bucket_date = date.fromisoformat(args.date)
    else:
        bucket_date = (datetime.now(timezone.utc) - timedelta(days=1)).date()

    bucket_hour = args.hour

    logger.info("Starting daily rollup for date={}, hour={}", bucket_date, bucket_hour)

    ch = get_ch_client()

    # Get tenants to process
    if args.tenant:
        tenants = [args.tenant]
    else:
        tenants = get_active_tenants()

    logger.info("Processing {} tenants: {}", len(tenants), tenants)

    for tenant in tenants:
        process_tenant(ch, tenant, bucket_date, bucket_hour)

    logger.info("Daily rollup completed")


if __name__ == "__main__":
    main()
