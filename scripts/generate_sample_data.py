"""
RiskLens — Sample data generator (standalone script).
Generates realistic synthetic transaction datasets for testing
the full pipeline without real client data.

Usage:
    python scripts/generate_sample_data.py
    python scripts/generate_sample_data.py --rows 50000 --fraud-rate 0.02
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path("./artifacts/data")
DEMO_TENANTS = ["bank_ng_gtb", "bank_ke_equity", "fintech_za_yoco"]


def _generate_synthetic_data(
    n: int = 10_000, fraud_rate: float = 0.015
) -> pd.DataFrame:
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

    return (
        pd.concat(
            [
                pd.DataFrame(make_block(n_fraud, True)),
                pd.DataFrame(make_block(n_legit, False)),
            ],
            ignore_index=True,
        )
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic RiskLens data")
    parser.add_argument("--rows", type=int, default=10_000, help="Rows per tenant")
    parser.add_argument(
        "--fraud-rate", type=float, default=0.015, help="Fraud rate (0–1)"
    )
    parser.add_argument("--tenants", nargs="+", default=DEMO_TENANTS)
    args = parser.parse_args()

    for tenant in args.tenants:
        out = DATA_DIR / tenant
        out.mkdir(parents=True, exist_ok=True)
        df = _generate_synthetic_data(n=args.rows, fraud_rate=args.fraud_rate)
        path = out / "features.parquet"
        df.to_parquet(path, index=False)
        fraud = int(df["label"].sum())
        print(
            f"✓ {tenant}: {len(df):,} rows  |  {fraud:,} fraud ({fraud/len(df)*100:.2f}%)  →  {path}"
        )

    print(f"\nDone. Run training with: python scripts/run_training.py --all-tenants")


if __name__ == "__main__":
    main()
