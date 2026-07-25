"""
RiskLens — Metrics Collector & ClickHouse Rollup
Collects daily model performance and drift metrics, persists to ClickHouse.
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone, timedelta, date
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from clickhouse_driver import Client

from loguru import logger

from monitoring.drift import (
    compute_all_feature_drift,
    compute_embedding_drift,
    compute_concept_drift,
    DEFAULT_MONITORED_FEATURES,
    DriftResult,
    EmbeddingDriftResult,
    ConceptDriftResult,
)

settings = None  # Will be set via init_metrics_collector()


@dataclass
class DailyModelMetrics:
    """Daily aggregated model performance metrics."""

    tenant_id: str
    model_version: str
    model_phase: str
    bucket_date: date
    # Discrimination
    auc_pr: float
    auc_roc: float
    # Threshold-based (at BLOCK threshold 0.85)
    recall: float
    precision: float
    f2_score: float
    fraud_capture_rate: float
    # Operational
    review_rate: float
    fpr: float
    # Latency
    avg_latency_ms: float
    p95_latency_ms: float
    # Volume
    total_scored: int
    total_blocked: int
    total_reviewed: int
    total_approved: int
    # Drift summary
    max_psi: float
    max_kl: float
    features_with_drift: int


def get_clickhouse_client() -> Client:
    """Get ClickHouse client from settings."""
    from config.settings import get_settings

    s = get_settings()
    return Client(
        host=s.clickhouse_host,
        port=s.clickhouse_port,
        database=s.clickhouse_database,
        user=s.clickhouse_user,
        password=s.clickhouse_password or "",
    )


def init_clickhouse_tables(client: Client) -> None:
    """Create ClickHouse tables for metrics and drift."""

    # Daily model performance metrics
    client.execute("""
        CREATE TABLE IF NOT EXISTS model_performance_daily (
            tenant_id String,
            model_version String,
            model_phase String,
            bucket_date Date,
            -- Discrimination
            auc_pr Float64,
            auc_roc Float64,
            -- Threshold-based (at 0.85)
            recall Float64,
            precision Float64,
            f2_score Float64,
            fraud_capture_rate Float64,
            -- Operational
            review_rate Float64,
            fpr Float64,
            -- Latency
            avg_latency_ms Float64,
            p95_latency_ms Float64,
            -- Volume
            total_scored UInt64,
            total_blocked UInt64,
            total_reviewed UInt64,
            total_approved UInt64,
            -- Drift summary
            max_psi Float64,
            max_kl Float64,
            features_with_drift UInt32,
            computed_at DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        PARTITION BY bucket_date
        ORDER BY (tenant_id, model_version, bucket_date)
        TTL bucket_date + INTERVAL 2 YEAR
        SETTINGS storage_policy = 'hot';
    """)

    # Hourly drift metrics per feature
    client.execute("""
        CREATE TABLE IF NOT EXISTS drift_metrics_hourly (
            tenant_id String,
            feature String,
            metric_type String,  -- 'psi', 'kl', 'embedding', 'concept'
            value Float64,
            bucket_date Date,
            bucket_hour UInt8,
            drift_detected UInt8,
            computed_at DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        PARTITION BY bucket_date
        ORDER BY (tenant_id, feature, metric_type, bucket_hour)
        TTL bucket_date + INTERVAL 90 DAY TO VOLUME 'cold',
            bucket_date + INTERVAL 2 YEAR DELETE
        SETTINGS storage_policy = 'hot';
    """)

    # Daily concept drift
    client.execute("""
        CREATE TABLE IF NOT EXISTS concept_drift_daily (
            tenant_id String,
            bucket_date Date,
            label_rate_baseline Float64,
            label_rate_current Float64,
            rate_change Float64,
            drift_detected UInt8,
            prediction_rate_baseline Nullable(Float64),
            prediction_rate_current Nullable(Float64),
            computed_at DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        PARTITION BY bucket_date
        ORDER BY (tenant_id, bucket_date)
        TTL bucket_date + INTERVAL 2 YEAR
        SETTINGS storage_policy = 'hot';
    """)

    # Embedding drift
    client.execute("""
        CREATE TABLE IF NOT EXISTS embedding_drift_hourly (
            tenant_id String,
            bucket_date Date,
            bucket_hour UInt8,
            centroid_distance Float64,
            max_distance Float64,
            mean_distance Float64,
            drift_detected UInt8,
            computed_at DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        PARTITION BY bucket_date
        ORDER BY (tenant_id, bucket_hour)
        TTL bucket_date + INTERVAL 90 DAY TO VOLUME 'cold',
            bucket_date + INTERVAL 2 YEAR DELETE
        SETTINGS storage_policy = 'hot';
    """)


def fetch_recent_scores(client: Client, tenant_id: str, days: int = 1) -> pd.DataFrame:
    """Fetch recent scores from ClickHouse or fallback."""
    try:
        query = """
            SELECT 
                tenant_id,
                transaction_id,
                risk_score,
                decision,
                model_phase,
                model_version,
                latency_ms,
                is_fraud,
                scored_at
            FROM recent_scores
            WHERE tenant_id = %(tenant)s
              AND scored_at >= now() - INTERVAL %(days)s DAY
        """
        rows = client.execute(query, {"tenant": tenant_id, "days": days})
        return pd.DataFrame(
            rows,
            columns=[
                "tenant_id",
                "transaction_id",
                "risk_score",
                "decision",
                "model_phase",
                "model_version",
                "latency_ms",
                "is_fraud",
                "scored_at",
            ],
        )
    except Exception as exc:
        logger.warning("Failed to fetch scores from ClickHouse: {}", exc)
        return pd.DataFrame()


def fetch_recent_labels(client: Client, tenant_id: str, days: int = 7) -> pd.DataFrame:
    """Fetch recent labels for concept drift."""
    try:
        query = """
            SELECT 
                transaction_id,
                label,
                labelled_at
            FROM labels
            WHERE tenant_id = %(tenant)s
              AND labelled_at >= now() - INTERVAL %(days)s DAY
        """
        rows = client.execute(query, {"tenant": tenant_id, "days": days})
        return pd.DataFrame(rows, columns=["transaction_id", "label", "labelled_at"])
    except Exception as exc:
        logger.warning("Failed to fetch labels from ClickHouse: {}", exc)
        return pd.DataFrame()


def fetch_recent_embeddings(
    client: Client, tenant_id: str, days: int = 7
) -> np.ndarray:
    """Fetch GNN embeddings for drift computation."""
    try:
        query = """
            SELECT embedding
            FROM gnn_embeddings
            WHERE tenant_id = %(tenant)s
              AND created_at >= now() - INTERVAL %(days)s DAY
        """
        rows = client.execute(query, {"tenant": tenant_id, "days": days})
        if not rows:
            return np.array([])
        return np.array([r[0] for r in rows])
    except Exception:
        return np.array([])


def compute_daily_metrics(
    tenant_id: str,
    scores_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    drift_results: dict,
    model_version: str,
    model_phase: str,
    bucket_date: date,
) -> Optional[DailyModelMetrics]:
    """Compute all daily metrics from scores and labels."""
    if scores_df.empty:
        return None

    # Filter to tenant
    scores = scores_df[scores_df["tenant_id"] == tenant_id].copy()
    if scores.empty:
        return None

    # Get model version from most common
    model_version = (
        scores["model_version"].mode()[0]
        if not scores["model_version"].empty
        else model_version
    )
    model_phase = (
        scores["model_phase"].mode()[0]
        if not scores["model_phase"].empty
        else model_phase
    )

    # Join with labels if available
    if not labels_df.empty:
        labels = labels_df[labels_df["transaction_id"].isin(scores["transaction_id"])]
        scores = scores.merge(
            labels, on="transaction_id", how="left", suffixes=("", "_label")
        )
    else:
        scores["label"] = np.nan

    # Extract arrays
    y_true = scores["label"].values
    y_score = scores["risk_score"].values
    decisions = scores["decision"].values
    latencies = scores["latency_ms"].values

    # Discrimination metrics (only where labels exist)
    labeled_mask = ~np.isnan(y_true)
    if labeled_mask.sum() > 10:
        y_true_l = y_true[labeled_mask].astype(int)
        y_score_l = y_score[labeled_mask]
        y_pred_l = (y_score_l >= 0.85).astype(int)

        from sklearn.metrics import (
            average_precision_score,
            roc_auc_score,
            recall_score,
            precision_score,
            fbeta_score,
        )

        auc_pr = average_precision_score(y_true_l, y_score_l)
        auc_roc = roc_auc_score(y_true_l, y_score_l)
        recall = recall_score(y_true_l, y_pred_l)
        precision = precision_score(y_true_l, y_pred_l, zero_division=0)
        f2 = fbeta_score(y_true_l, y_pred_l, beta=2, zero_division=0)
        fraud_capture = (y_true_l & y_pred_l).sum() / max(y_true_l.sum(), 1)
    else:
        auc_pr = auc_roc = recall = precision = f2 = fraud_capture = 0.0

    # Operational metrics
    total = len(scores)
    blocked = (decisions == "BLOCK").sum()
    reviewed = (decisions == "REVIEW").sum()
    approved = (decisions == "APPROVE").sum()

    review_rate = reviewed / max(total, 1)
    fpr = 0.0
    if labeled_mask.sum() > 0:
        neg_mask = ~y_true[labeled_mask].astype(bool)
        if neg_mask.sum() > 0:
            fpr = ((y_pred_l == 1) & neg_mask).sum() / max(neg_mask.sum(), 1)

    # Latency
    avg_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))

    # Drift summary
    max_psi = max((r.psi for r in drift_results.values()), default=0.0)
    max_kl = max((r.kl_divergence for r in drift_results.values()), default=0.0)
    features_with_drift = sum(1 for r in drift_results.values() if r.drift_detected)

    return DailyModelMetrics(
        tenant_id=tenant_id,
        model_version=model_version,
        model_phase=model_phase,
        bucket_date=bucket_date,
        auc_pr=auc_pr,
        auc_roc=auc_roc,
        recall=recall,
        precision=precision,
        f2_score=f2,
        fraud_capture_rate=fraud_capture,
        review_rate=review_rate,
        fpr=fpr,
        avg_latency_ms=avg_latency,
        p95_latency_ms=p95_latency,
        total_scored=total,
        total_blocked=blocked,
        total_reviewed=reviewed,
        total_approved=approved,
        max_psi=max_psi,
        max_kl=max_kl,
        features_with_drift=features_with_drift,
    )


def persist_daily_metrics(client: Client, metrics: DailyModelMetrics) -> None:
    """Persist daily model metrics to ClickHouse."""
    client.execute(
        """
        INSERT INTO model_performance_daily (
            tenant_id, model_version, model_phase, bucket_date,
            auc_pr, auc_roc, recall, precision, f2_score, fraud_capture_rate,
            review_rate, fpr, avg_latency_ms, p95_latency_ms,
            total_scored, total_blocked, total_reviewed, total_approved,
            max_psi, max_kl, features_with_drift
        ) VALUES
    """,
        [
            [
                metrics.tenant_id,
                metrics.model_version,
                metrics.model_phase,
                metrics.bucket_date,
                metrics.auc_pr,
                metrics.auc_roc,
                metrics.recall,
                metrics.precision,
                metrics.f2_score,
                metrics.fraud_capture_rate,
                metrics.review_rate,
                metrics.fpr,
                metrics.avg_latency_ms,
                metrics.p95_latency_ms,
                metrics.total_scored,
                metrics.total_blocked,
                metrics.total_reviewed,
                metrics.total_approved,
                metrics.max_psi,
                metrics.max_kl,
                metrics.features_with_drift,
            ]
        ],
    )


def persist_drift_metrics(
    client: Client,
    tenant_id: str,
    drift_results: dict[str, DriftResult],
    bucket_date: date,
    bucket_hour: int,
) -> None:
    """Persist hourly drift metrics to ClickHouse."""
    rows = []
    for feature, result in drift_results.items():
        rows.append(
            [
                tenant_id,
                feature,
                "psi",
                result.psi,
                bucket_date,
                bucket_hour,
                1 if result.drift_detected else 0,
            ]
        )
        rows.append(
            [
                tenant_id,
                feature,
                "kl",
                result.kl_divergence,
                bucket_date,
                bucket_hour,
                1 if result.drift_detected else 0,
            ]
        )

    if rows:
        client.execute(
            """
            INSERT INTO drift_metrics_hourly (
                tenant_id, feature, metric_type, value, bucket_date, bucket_hour, drift_detected
            ) VALUES
        """,
            rows,
        )


def persist_embedding_drift(
    client: Client, result: EmbeddingDriftResult, bucket_date: date, bucket_hour: int
) -> None:
    """Persist embedding drift metrics."""
    client.execute(
        """
        INSERT INTO embedding_drift_hourly (
            tenant_id, bucket_date, bucket_hour,
            centroid_distance, max_distance, mean_distance, drift_detected
        ) VALUES
    """,
        [
            [
                result.tenant_id,
                bucket_date,
                bucket_hour,
                result.centroid_distance,
                result.max_distance,
                result.mean_distance,
                1 if result.drift_detected else 0,
            ]
        ],
    )


def persist_concept_drift(
    client: Client, result: ConceptDriftResult, bucket_date: date
) -> None:
    """Persist concept drift metrics."""
    client.execute(
        """
        INSERT INTO concept_drift_daily (
            tenant_id, bucket_date, label_rate_baseline, label_rate_current,
            rate_change, drift_detected,
            prediction_rate_baseline, prediction_rate_current
        ) VALUES
    """,
        [
            [
                result.tenant_id,
                bucket_date,
                result.label_rate_baseline,
                result.label_rate_current,
                result.rate_change,
                1 if result.drift_detected else 0,
                result.prediction_rate_baseline,
                result.prediction_rate_current,
            ]
        ],
    )


def run_daily_rollup(tenant_ids: list[str], bucket_date: Optional[date] = None) -> None:
    """
    Main rollup job - computes and persists all daily metrics.

    Args:
        tenant_ids: List of tenant IDs to process
        bucket_date: Date to process (defaults to yesterday)
    """
    from config.settings import get_settings

    global settings
    settings = get_settings()

    if bucket_date is None:
        bucket_date = date.today() - timedelta(days=1)

    bucket_hour = datetime.now().hour
    client = get_clickhouse_client()

    # Ensure tables exist
    init_clickhouse_tables(client)

    for tenant_id in tenant_ids:
        logger.info(
            "Running daily rollup for tenant={} date={}", tenant_id, bucket_date
        )

        try:
            # 1. Fetch data
            scores_df = fetch_recent_scores(client, tenant_id, days=1)
            labels_df = fetch_recent_labels(client, tenant_id, days=7)

            if scores_df.empty:
                logger.warning("No scores found for tenant={}", tenant_id)
                continue

            # 2. Compute feature drift (baseline = 7d ago, current = last 1d)
            # For simplicity, split scores_df by date
            scores_df["scored_date"] = pd.to_datetime(scores_df["scored_at"]).dt.date
            current_day = bucket_date
            baseline_start = current_day - timedelta(days=7)
            baseline_end = current_day - timedelta(days=1)

            baseline_df = scores_df[
                (scores_df["scored_date"] >= baseline_start)
                & (scores_df["scored_date"] <= baseline_end)
            ]
            current_df = scores_df[scores_df["scored_date"] == current_day]

            feature_list = [
                f for f in DEFAULT_MONITORED_FEATURES if f in scores_df.columns
            ]
            drift_results = compute_all_feature_drift(
                baseline_df, current_df, feature_list
            )

            # 3. Compute embedding drift (if GNN active)
            baseline_embeddings = fetch_recent_embeddings(client, tenant_id, days=7)
            current_embeddings = fetch_recent_embeddings(client, tenant_id, days=1)
            embedding_drift = None
            if len(baseline_embeddings) > 0 and len(current_embeddings) > 0:
                embedding_drift = compute_embedding_drift(
                    tenant_id, baseline_embeddings, current_embeddings
                )
                persist_embedding_drift(
                    client, embedding_drift, bucket_date, bucket_hour
                )

            # 4. Compute concept drift
            concept_drift = None
            if not labels_df.empty:
                # Split labels by period
                labels_df["labelled_date"] = pd.to_datetime(
                    labels_df["labelled_at"]
                ).dt.date
                base_labels = labels_df[
                    (labels_df["labelled_date"] >= baseline_start)
                    & (labels_df["labelled_date"] <= baseline_end)
                ]["label"].values
                curr_labels = labels_df[labels_df["labelled_date"] == current_day][
                    "label"
                ].values

                # Get predictions for same transactions
                base_scores = scores_df[
                    scores_df["scored_date"].isin(
                        pd.date_range(baseline_start, baseline_end)
                    )
                ]["risk_score"].values
                curr_scores = scores_df[scores_df["scored_date"] == current_day][
                    "risk_score"
                ].values

                concept_drift = compute_concept_drift(
                    tenant_id,
                    base_labels,
                    curr_labels,
                    baseline_predictions=base_scores,
                    current_predictions=curr_scores,
                )
                persist_concept_drift(client, concept_drift, bucket_date)

            # 5. Compute daily model metrics
            metrics = compute_daily_metrics(
                tenant_id=tenant_id,
                scores_df=scores_df,
                labels_df=labels_df,
                drift_results=drift_results,
                model_version=(
                    scores_df["model_version"].mode()[0]
                    if not scores_df["model_version"].empty
                    else "unknown"
                ),
                model_phase=(
                    scores_df["model_phase"].mode()[0]
                    if not scores_df["model_phase"].empty
                    else "unknown"
                ),
                bucket_date=bucket_date,
            )

            if metrics:
                persist_daily_metrics(client, metrics)

            # 6. Persist drift metrics
            persist_drift_metrics(
                client, tenant_id, drift_results, bucket_date, bucket_hour
            )

            logger.info(
                "Daily rollup complete for tenant={}: scored={}, drift_features={}",
                tenant_id,
                metrics.total_scored if metrics else 0,
                metrics.features_with_drift if metrics else 0,
            )

        except Exception as exc:
            logger.exception("Daily rollup failed for tenant={}: {}", tenant_id, exc)


# Allow running as script
if __name__ == "__main__":
    import sys
    from config.settings import get_settings

    settings = get_settings()
    tenant_ids = ["bank_ng_gtb", "bank_ke_equity", "fintech_za_yoco"]

    # Allow override via CLI
    if len(sys.argv) > 1:
        tenant_ids = sys.argv[1].split(",")

    run_daily_rollup(tenant_ids)
