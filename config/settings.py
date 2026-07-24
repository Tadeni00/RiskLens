"""
FraudTrap — Central Configuration
All environment-driven settings with safe defaults for local development.
"""

from __future__ import annotations
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        protected_namespaces=("settings_",),
    )

    # ── App ───────────────────────────────────────────────────────────────────
    app_name: str = "FraudTrap"
    app_version: str = "1.0.0"
    environment: str = Field("development", env="ENVIRONMENT")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_workers: int = Field(4, env="API_WORKERS")
    scoring_timeout_ms: int = Field(90, env="SCORING_TIMEOUT_MS")  # hard wall
    model_dir: str = Field("artifacts/models", env="MODEL_DIR")
    model_reload_interval_seconds: float = Field(300.0, env="MODEL_RELOAD_INTERVAL_SECONDS")

    # ── Kafka ─────────────────────────────────────────────────────────────────
    kafka_brokers: str = Field("localhost:9092", env="KAFKA_BROKERS")
    kafka_topic_transactions: str = "fraudtrap.transactions.raw"
    kafka_topic_scored: str = "fraudtrap.transactions.scored"
    kafka_topic_labels: str = "fraudtrap.labels.incoming"
    kafka_topic_audit: str = "fraudtrap.audit.decisions"
    kafka_consumer_group: str = "fraudtrap-scoring-group"
    kafka_schema_registry_url: str = Field("http://localhost:8081", env="KAFKA_SCHEMA_REGISTRY_URL")

    # ── Redis (online feature store) ──────────────────────────────────────────
    redis_host: str = Field("localhost", env="REDIS_HOST")
    redis_port: int = Field(6379, env="REDIS_PORT")
    redis_password: str = Field("", env="REDIS_PASSWORD")
    redis_db: int = Field(0, env="REDIS_DB")
    redis_feature_ttl_seconds: int = Field(86_400, env="REDIS_FEATURE_TTL")  # 24h

    # ── ClickHouse (offline feature store + analytics) ────────────────────────
    clickhouse_host: str = Field("localhost", env="CLICKHOUSE_HOST")
    clickhouse_port: int = Field(9000, env="CLICKHOUSE_PORT")
    clickhouse_database: str = Field("fraudtrap", env="CLICKHOUSE_DATABASE")
    clickhouse_user: str = Field("default", env="CLICKHOUSE_USER")
    clickhouse_password: str = Field("", env="CLICKHOUSE_PASSWORD")

    # ── Postgres (metadata, model registry mirror) ────────────────────────────
    postgres_url: str = Field(
        "postgresql://fraudtrap:fraudtrap@localhost:5432/fraudtrap",
        env="POSTGRES_URL",
    )

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = Field("http://localhost:5000", env="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = "fraudtrap-production"

    # ── Model lifecycle thresholds ────────────────────────────────────────────
    # Cold-start → adaptive learning gate
    phase1_min_fraud_labels: int = 500
    phase1_min_transactions: int = 500_000
    phase1_min_weeks: int = 8
    phase1_min_pr_auc: float = 0.65

    # Adaptive learning → supervised gate
    phase2_min_fraud_labels: int = 5_000
    phase2_min_pr_auc: float = 0.78

    # Drift alerts
    psi_drift_threshold: float = 0.20  # Population Stability Index
    performance_drop_threshold: float = 0.05  # absolute F1 drop triggers retrain

    # ── Scoring thresholds ────────────────────────────────────────────────────
    score_block_threshold: float = 0.85  # auto-block above this
    score_review_low: float = 0.40  # review band lower
    score_review_high: float = 0.85  # review band upper (below block)
    # < 0.40 → approve automatically

    # ── Privacy / DP ─────────────────────────────────────────────────────────
    dp_epsilon: float = 10.0  # privacy budget
    dp_delta: float = 1e-5
    dp_max_grad_norm: float = 1.0

    # ── Label pipeline ────────────────────────────────────────────────────────
    label_lag_days: int = 70  # buffer for chargeback arrival
    training_window_days: int = 180
    retrain_schedule_cron: str = "0 2 * * 1"  # every Monday 02:00 UTC

    # ── TabPFN (Adaptive Learning) ─────────────────────────────────────────
    tabpfn_token: str = Field("", env="TABPFN_TOKEN")

    # ── Feature windows ───────────────────────────────────────────────────────
    velocity_windows: list[int] = [1, 5, 60, 1_440, 10_080]  # minutes


@lru_cache
def get_settings() -> Settings:
    return Settings()
