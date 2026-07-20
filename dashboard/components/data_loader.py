"""
FraudTrap Dashboard — Data Loader
Provides realistic synthetic data for all dashboard pages.
In production, replace _load_*() methods with ClickHouse queries.
"""
from __future__ import annotations
import os
import logging
import uuid
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _get_rng() -> np.random.Generator:
    """Return a new RNG seeded from OS entropy (unique per call)."""
    return np.random.default_rng()


def make_transactions(n: int = 5000, fraud_rate: float = 0.015) -> pd.DataFrame:
    rng = _get_rng()
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud
    now = datetime.now(timezone.utc)

    rows = []
    for i in range(n):
        is_fraud = i < n_fraud
        delta_days = rng.integers(0, 90)
        ts = now - timedelta(days=int(delta_days), hours=int(rng.integers(0, 24)))
        rows.append({
            "transaction_id":   f"txn_{i:06d}",
            "trace_id":         f"trace_{uuid.uuid4().hex[:12]}",
            "tenant_id":        rng.choice(["bank_ng_gtb", "bank_ke_equity", "fintech_za_yoco"]),
            "amount":           float(rng.lognormal(9.5 if is_fraud else 8.5, 1.2)),
            "currency":         rng.choice(["NGN", "KES", "ZAR"]),
            "timestamp":        ts,
            "transaction_type": rng.choice(["PAYMENT", "TRANSFER", "WITHDRAWAL", "TOP_UP"]),
            "channel":          rng.choice(["MOBILE", "WEB", "POS", "ATM"]),
            "country_code":     rng.choice(["NG", "KE", "ZA", "GB", "US"]),
            "is_fraud":         int(is_fraud),
            "risk_score":       float(rng.beta(8, 2) if is_fraud else rng.beta(1, 8)),
            "decision":         ("BLOCK" if is_fraud and rng.random() > 0.2
                                 else "REVIEW" if rng.random() > 0.85
                                 else "APPROVE"),
            "amount_zscore":    float(rng.normal(3.0 if is_fraud else 0.0, 1.0)),
            "acct_v_1h_count":  float(rng.poisson(20 if is_fraud else 3)),
            "is_new_device":    float(rng.binomial(1, 0.6 if is_fraud else 0.05)),
            "impossible_travel":float(rng.binomial(1, 0.3 if is_fraud else 0.001)),
            "geo_speed_kmh":    float(rng.exponential(800 if is_fraud else 30)),
            "typing_zscore":    float(rng.normal(2.5 if is_fraud else 0.0, 1.0)),
            "latency_ms":       float(rng.normal(72, 12)),
            "model_phase":      rng.choice(["UNSUPERVISED", "SEMI_SUPERVISED", "SUPERVISED"],
                                           p=[0.15, 0.25, 0.60]),
        })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def make_feature_importance() -> pd.DataFrame:
    rng = _get_rng()
    features = [
        "acct_v_1h_count", "amount_zscore", "is_new_device",
        "impossible_travel", "geo_speed_kmh", "typing_zscore",
        "acct_v_24h_total_amt", "device_account_count", "is_new_merchant",
        "cross_country_flag", "is_night", "is_round_amount",
        "channel_enc", "hour_sin", "amount_log",
    ]
    importance = np.array([
        0.18, 0.15, 0.12, 0.10, 0.09, 0.08,
        0.07, 0.06, 0.05, 0.04, 0.03, 0.02,
        0.015, 0.01, 0.005,
    ])
    return pd.DataFrame({"feature": features, "importance": importance})


def make_shap_values(n_samples: int = 200) -> pd.DataFrame:
    rng = _get_rng()
    features = [
        "acct_v_1h_count", "amount_zscore", "is_new_device",
        "impossible_travel", "geo_speed_kmh", "typing_zscore",
        "acct_v_24h_total_amt", "device_account_count",
    ]
    rows = []
    for _ in range(n_samples):
        is_fraud = rng.random() < 0.3
        for feat in features:
            base = {"acct_v_1h_count": 0.18, "amount_zscore": 0.15,
                    "is_new_device": 0.12, "impossible_travel": 0.10,
                    "geo_speed_kmh": 0.09, "typing_zscore": 0.08,
                    "acct_v_24h_total_amt": 0.07, "device_account_count": 0.06}
            sign = 1 if is_fraud else -1
            rows.append({
                "feature": feat,
                "shap_value": sign * base[feat] * rng.uniform(0.5, 1.5),
                "is_fraud": is_fraud,
            })
    return pd.DataFrame(rows)


def make_drift_data() -> dict:
    rng = _get_rng()
    features = [
        "amount", "acct_v_1h_count", "geo_speed_kmh",
        "typing_zscore", "device_account_count", "amount_zscore",
    ]
    return {
        feat: {
            "psi": float(rng.uniform(0.02, 0.35)),
            "baseline_mean": float(rng.normal(0, 1)),
            "current_mean": float(rng.normal(0.2 if feat == "amount" else 0, 1)),
        }
        for feat in features
    }


def make_pr_curve() -> pd.DataFrame:
    thresholds = np.linspace(0, 1, 100)
    precision  = 0.05 + 0.90 * thresholds ** 0.4
    recall     = 1.0  - thresholds ** 0.6
    return pd.DataFrame({"threshold": thresholds, "precision": precision, "recall": recall})


def make_confusion_matrix() -> dict:
    return {"TP": 423, "FP": 87, "TN": 47_231, "FN": 52}


def make_score_distribution(n: int = 2000) -> pd.DataFrame:
    rng = _get_rng()
    fraud_scores = rng.beta(8, 2, int(n * 0.015))
    legit_scores = rng.beta(1, 8, int(n * 0.985))
    return pd.DataFrame({
        "score": np.concatenate([fraud_scores, legit_scores]),
        "label": np.concatenate([
            np.ones(len(fraud_scores)), np.zeros(len(legit_scores))
        ]),
    })


def _fetch_recent(tenant_id: str = "all_tenants", limit: int = 1000) -> tuple[pd.DataFrame, bool]:
    """Fetch recent transactions from ClickHouse. Returns (df, is_live)."""
    try:
        from clickhouse_driver import Client

        host = os.getenv("CLICKHOUSE_HOST", "clickhouse")
        port = int(os.getenv("CLICKHOUSE_PORT", "9000"))
        client = Client(host=host, port=port)

        query = "SELECT * FROM fraudtrap.transactions"
        params = {}

        if tenant_id != "all_tenants":
            query += " WHERE tenant_id = %(tenant_id)s"
            params["tenant_id"] = tenant_id

        query += " ORDER BY timestamp DESC LIMIT %(limit)s"
        params["limit"] = limit

        data = client.execute(query, params, with_column_types=True)
        if not data or not data[0]:
            return pd.DataFrame(), False

        columns = [col[0] for col in data[1]]
        df = pd.DataFrame(data[0], columns=columns)

        df["is_fraud"] = df["is_fraud"].astype(int)

        defaults = {
            "amount_zscore": 0.0,
            "acct_v_1h_count": 0.0,
            "acct_v_24h_total_amt": 0.0,
            "geo_speed_kmh": 0.0,
            "typing_zscore": 0.0,
            "is_new_device": 0.0,
            "impossible_travel": 0.0,
            "device_account_count": 0.0,
            "is_new_merchant": 0.0,
            "cross_country_flag": 0.0,
            "is_night": 0.0,
            "is_round_amount": 0.0,
            "channel_enc": 0.0,
            "hour_sin": 0.0,
            "amount_log": 0.0,
        }
        for col, default in defaults.items():
            if col not in df.columns:
                df[col] = default
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

        return df, True
    except Exception as exc:
        logger.debug("ClickHouse fetch failed (using synthetic): %s", exc)
        return pd.DataFrame(), False


def make_live_timeseries(hours: int = 24, tenant_id: str = "all_tenants") -> tuple[pd.DataFrame, bool]:
    """Returns (df, is_live)."""
    recent, is_live = _fetch_recent(tenant_id, limit=2000)
    if is_live and not recent.empty:
        recent["bucket"] = recent["timestamp"].dt.floor("h")
        grouped = recent.groupby("bucket").agg(
            txn_count=("transaction_id", "count"),
            fraud_rate=("is_fraud", "mean"),
            latency_p95=("latency_ms", lambda s: float(np.percentile(s, 95))),
        ).reset_index()
        grouped["txn_per_min"] = grouped["txn_count"] / 60.0
        grouped["fp_rate"] = 0.0
        grouped = grouped.rename(columns={"bucket": "timestamp"})
        return grouped.tail(hours), True

    rng = _get_rng()
    now = datetime.now(timezone.utc)
    rows = []
    for h in range(hours):
        ts = now - timedelta(hours=hours - h)
        rows.append({
            "timestamp":   ts,
            "txn_per_min": float(rng.poisson(800 + 200 * np.sin(h * np.pi / 12))),
            "fraud_rate":  float(rng.beta(2, 150)),
            "latency_p95": float(rng.normal(78, 8)),
            "fp_rate":     float(rng.beta(1, 50)),
        })
    return pd.DataFrame(rows), False


def load_data(tenant_id: str) -> tuple[pd.DataFrame, bool]:
    """Returns (df, is_live)."""
    recent, is_live = _fetch_recent(tenant_id, limit=5000)
    if is_live and not recent.empty:
        return recent, True
    df = make_transactions(5000)
    if tenant_id != "all_tenants":
        df = df[df["tenant_id"] == tenant_id]
    return df, False


def compute_kpis(df: pd.DataFrame) -> dict:
    """Compute KPIs from a DataFrame (works for both live and synthetic data)."""
    if df.empty:
        return {"total": 0, "n_block": 0, "n_review": 0, "fraud_rate": 0.0, "avg_lat": 0.0}

    total = len(df)
    n_block = int((df["decision"] == "BLOCK").sum()) if "decision" in df.columns else 0
    n_review = int((df["decision"] == "REVIEW").sum()) if "decision" in df.columns else 0
    fraud_rate = float(df["is_fraud"].mean() * 100) if "is_fraud" in df.columns else 0.0
    avg_lat = float(df["latency_ms"].mean()) if "latency_ms" in df.columns else 0.0

    return {
        "total": total,
        "n_block": n_block,
        "n_review": n_review,
        "fraud_rate": fraud_rate,
        "avg_lat": avg_lat,
    }
