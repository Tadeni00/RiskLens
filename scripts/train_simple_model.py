from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scoring.simple_model import SimpleFraudModel

FEATURES = [
    "amount",
    "amount_log",
    "amount_zscore",
    "hour_sin",
    "hour_cos",
    "is_weekend",
    "is_night",
    "is_round_amount",
    "is_new_merchant",
    "is_new_device",
    "device_shared_flag",
    "device_account_count",
    "geo_speed_kmh",
    "impossible_travel",
    "cross_country_flag",
    "acct_v_1m_count",
    "acct_v_1h_count",
    "acct_v_24h_count",
    "acct_v_24h_total_amt",
    "typing_zscore",
    "channel_enc",
    "txn_type_enc",
]


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def train_logistic(
    X: np.ndarray, y: np.ndarray, epochs: int, lr: float
) -> tuple[np.ndarray, float]:
    weights = np.zeros(X.shape[1], dtype=np.float32)
    bias = 0.0
    pos_weight = max(float((y == 0).sum()) / max(float((y == 1).sum()), 1.0), 1.0)
    sample_weight = np.where(y == 1, pos_weight, 1.0).astype(np.float32)

    for _ in range(epochs):
        pred = sigmoid(X @ weights + bias)
        err = (pred - y) * sample_weight
        weights -= lr * ((X.T @ err) / len(X) + 0.001 * weights)
        bias -= lr * float(err.mean())
    return weights, bias


def build_calibration(raw_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map learned score rank into practical fraud-operation bands."""
    percentiles = np.array([0, 50, 75, 90, 95, 97.5, 99, 99.5, 100], dtype=np.float32)
    raw = np.percentile(raw_scores, percentiles).astype(np.float32)
    raw = np.maximum.accumulate(raw)
    for i in range(1, len(raw)):
        if raw[i] <= raw[i - 1]:
            raw[i] = np.nextafter(raw[i - 1], np.float32(np.inf))
    calibrated = np.array(
        [0.01, 0.08, 0.18, 0.35, 0.50, 0.68, 0.85, 0.93, 0.99], dtype=np.float32
    )
    return raw, calibrated


def train_tenant(
    tenant: str, data_dir: Path, model_dir: Path, epochs: int, lr: float
) -> Path:
    data_path = data_dir / tenant / "features.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing training data: {data_path}")

    df = pd.read_parquet(data_path)
    if "label" not in df.columns:
        raise ValueError(f"{data_path} has no label column")

    for feature in FEATURES:
        if feature not in df.columns:
            df[feature] = 0.0

    X_raw = df[FEATURES].fillna(0.0).astype("float32").values
    y = df["label"].astype("float32").values
    mean = X_raw.mean(axis=0)
    scale = X_raw.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    X = (X_raw - mean) / scale

    weights, bias = train_logistic(X, y, epochs=epochs, lr=lr)
    raw_model = SimpleFraudModel(FEATURES, weights, bias, mean, scale, "raw")
    raw_scores = raw_model.score(X_raw)
    calibration_raw, calibration_score = build_calibration(raw_scores)
    version = f"simple_{tenant}_{int(time.time())}"
    model = SimpleFraudModel(
        FEATURES,
        weights,
        bias,
        mean,
        scale,
        version,
        calibration_raw=calibration_raw,
        calibration_score=calibration_score,
    )

    out = model_dir / tenant / "simple_model.pkl"
    model.save(out)
    (model_dir / tenant / "version.txt").write_text(version, encoding="utf-8")

    scores = model.score(X_raw)
    print(
        f"{tenant}: trained {len(df):,} rows, fraud={y.mean()*100:.2f}%, "
        f"score_mean={scores.mean():.3f}, score_p95={np.percentile(scores, 95):.3f}, "
        f"score_p99={np.percentile(scores, 99):.3f} -> {out}"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train lightweight FraudTrap serving models"
    )
    parser.add_argument("--tenant", action="append", help="Tenant to train; repeatable")
    parser.add_argument("--all-tenants", action="store_true")
    parser.add_argument("--data-dir", default="artifacts/data")
    parser.add_argument("--model-dir", default="artifacts/models")
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--lr", type=float, default=0.08)
    args = parser.parse_args()

    tenants = args.tenant or []
    if args.all_tenants or not tenants:
        tenants = ["bank_ng_gtb", "bank_ke_equity", "fintech_za_yoco"]

    for tenant in tenants:
        train_tenant(
            tenant, Path(args.data_dir), Path(args.model_dir), args.epochs, args.lr
        )


if __name__ == "__main__":
    main()
