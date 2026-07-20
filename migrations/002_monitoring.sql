-- FraudTrap — ClickHouse Tables for Monitoring & Drift
-- Version: 002
-- Description: Creates tables for model performance, drift metrics, and alerting

-- Enable required settings
SET allow_experimental_object_type = 1;

-- ============================================================
-- 1. Daily Model Performance Metrics
-- ============================================================
CREATE TABLE IF NOT EXISTS model_performance_daily (
    tenant_id           String,
    model_version       String,
    model_phase         String,           -- 'UNSUPERVISED', 'SEMI_SUPERVISED', 'SUPERVISED'
    bucket_date         Date,
    -- Discrimination metrics
    auc_pr              Float64,
    auc_roc             Float64,
    -- Threshold-based metrics (at BLOCK threshold 0.85)
    recall              Float64,
    precision           Float64,
    f2_score            Float64,
    fraud_capture_rate  Float64,
    -- Operational metrics
    review_rate         Float64,
    fpr                 Float64,          -- False positive rate
    -- Latency metrics
    avg_latency_ms      Float64,
    p95_latency_ms      Float64,
    -- Volume metrics
    total_scored        UInt64,
    total_blocked       UInt64,
    total_reviewed      UInt64,
    total_approved      UInt64,
    -- Drift summary
    max_psi             Float64,
    max_kl              Float64,
    features_with_drift UInt32,
    computed_at         DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY bucket_date
ORDER BY (tenant_id, model_version, bucket_date)
TTL bucket_date + INTERVAL 2 YEAR
SETTINGS storage_policy = 'hot';

-- ============================================================
-- 2. Hourly Drift Metrics per Feature
-- ============================================================
CREATE TABLE IF NOT EXISTS drift_metrics_hourly (
    tenant_id           String,
    feature             String,
    metric_type         String,           -- 'psi', 'kl', 'embedding', 'concept'
    value               Float64,
    bucket_date         Date,
    bucket_hour         UInt8,
    drift_detected      UInt8,
    computed_at         DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY bucket_date
ORDER BY (tenant_id, feature, metric_type, bucket_hour)
TTL bucket_date + INTERVAL 90 DAY TO VOLUME 'cold',
    bucket_date + INTERVAL 2 YEAR DELETE
SETTINGS storage_policy = 'hot';

-- ============================================================
-- 3. Daily Concept Drift
-- ============================================================
CREATE TABLE IF NOT EXISTS concept_drift_daily (
    tenant_id                   String,
    bucket_date                 Date,
    label_rate_baseline         Float64,
    label_rate_current          Float64,
    rate_change                 Float64,
    drift_detected              UInt8,
    prediction_rate_baseline    Nullable(Float64),
    prediction_rate_current     Nullable(Float64),
    computed_at                 DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY bucket_date
ORDER BY (tenant_id, bucket_date)
TTL bucket_date + INTERVAL 2 YEAR
SETTINGS storage_policy = 'hot';

-- ============================================================
-- 4. Hourly Embedding Drift (for GNN models)
-- ============================================================
CREATE TABLE IF NOT EXISTS embedding_drift_hourly (
    tenant_id           String,
    bucket_date         Date,
    bucket_hour         UInt8,
    centroid_distance   Float64,
    max_distance        Float64,
    mean_distance       Float64,
    drift_detected      UInt8,
    computed_at         DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY bucket_date
ORDER BY (tenant_id, bucket_date, bucket_hour)
TTL bucket_date + INTERVAL 90 DAY TO VOLUME 'cold',
    bucket_date + INTERVAL 2 YEAR DELETE
SETTINGS storage_policy = 'hot';

-- ============================================================
-- 5. Hourly Alerts Log
-- ============================================================
CREATE TABLE IF NOT EXISTS alerts_log (
    alert_id            String,
    title               String,
    message             String,
    severity            String,           -- 'info', 'warning', 'critical'
    category            String,           -- 'sla_breach', 'drift_spike', etc.
    tenant_id           String,
    timestamp           DateTime,
    metadata            String,           -- JSON
    runbook_url         Nullable(String),
    acknowledged        UInt8 DEFAULT 0,
    acknowledged_by     Nullable(String),
    acknowledged_at     Nullable(DateTime)
) ENGINE = MergeTree()
PARTITION BY toDate(timestamp)
ORDER BY (tenant_id, timestamp)
TTL timestamp + INTERVAL 1 YEAR
SETTINGS storage_policy = 'hot';

-- ============================================================
-- Materialized Views for Dashboards
-- ============================================================

-- Daily model health summary
CREATE MATERIALIZED VIEW IF NOT EXISTS model_health_daily
ENGINE = SummingMergeTree()
PARTITION BY bucket_date
ORDER BY (tenant_id, bucket_date)
AS SELECT
    tenant_id,
    bucket_date,
    count() as model_count,
    avg(auc_pr) as avg_auc_pr,
    min(auc_pr) as min_auc_pr,
    max(p95_latency_ms) as max_p95_latency_ms,
    sum(total_scored) as total_scored,
    sum(total_blocked) as total_blocked,
    sum(total_reviewed) as total_reviewed,
    sum(total_approved) as total_approved,
    max(max_psi) as max_psi_any_feature,
    max(max_kl) as max_kl_any_feature,
    sum(features_with_drift) as total_features_with_drift
FROM model_performance_daily
GROUP BY tenant_id, bucket_date;

-- Drift alert summary (features with drift > threshold)
CREATE MATERIALIZED VIEW IF NOT EXISTS drift_alerts_daily
ENGINE = SummingMergeTree()
PARTITION BY bucket_date
ORDER BY (tenant_id, bucket_date)
AS SELECT
    tenant_id,
    bucket_date,
    countIf(metric_type = 'psi' AND drift_detected = 1) as psi_alerts,
    countIf(metric_type = 'kl' AND drift_detected = 1) as kl_alerts,
    countIf(metric_type = 'embedding' AND drift_detected = 1) as embedding_alerts,
    countIf(metric_type = 'concept' AND drift_detected = 1) as concept_alerts
FROM drift_metrics_hourly
GROUP BY tenant_id, bucket_date;

-- ============================================================
-- Grant permissions (adjust role as needed)
-- ============================================================
-- GRANT SELECT, INSERT ON model_performance_daily TO fraudtrap_app;
-- GRANT SELECT, INSERT ON drift_metrics_hourly TO fraudtrap_app;
-- GRANT SELECT, INSERT ON concept_drift_daily TO fraudtrap_app;
-- GRANT SELECT, INSERT ON embedding_drift_hourly TO fraudtrap_app;
-- GRANT SELECT, INSERT ON alerts_log TO fraudtrap_app;
-- GRANT SELECT ON model_health_daily TO fraudtrap_app;
-- GRANT SELECT ON drift_alerts_daily TO fraudtrap_app;

-- ============================================================
-- Comments
-- ============================================================
COMMENT ON TABLE model_performance_daily IS 
'Daily aggregated model performance metrics per tenant/model version.';

COMMENT ON TABLE drift_metrics_hourly IS 
'Hourly drift metrics (PSI, KL divergence) per feature per tenant.';

COMMENT ON TABLE concept_drift_daily IS 
'Daily concept drift metrics comparing label/prediction distributions.';

COMMENT ON TABLE embedding_drift_hourly IS 
'Hourly embedding drift metrics for GNN models.';

COMMENT ON TABLE alerts_log IS 
'Log of all alerts sent via PagerDuty/Slack.';

COMMENT ON TABLE model_health_daily IS 
'Daily model health summary for dashboard.';

COMMENT ON TABLE drift_alerts_daily IS 
'Daily count of drift alerts per tenant.';