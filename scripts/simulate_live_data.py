"""
FraudTrap live traffic simulator.

Continuously posts realistic transaction payloads to /v1/score and sends
ground-truth labels for sampled simulated fraud cases.
"""

from __future__ import annotations

import argparse
import random
import time
import uuid
from datetime import datetime, timezone

import httpx

TENANTS = ["bank_ng_gtb", "bank_ke_equity", "fintech_za_yoco"]
CURRENCIES = {"bank_ng_gtb": "NGN", "bank_ke_equity": "KES", "fintech_za_yoco": "ZAR"}
COUNTRIES = {"bank_ng_gtb": "NG", "bank_ke_equity": "KE", "fintech_za_yoco": "ZA"}
CHANNELS = ["MOBILE", "WEB", "POS", "ATM", "API", "USSD"]
TXN_TYPES = ["PAYMENT", "TRANSFER", "WITHDRAWAL", "TOP_UP"]


def build_transaction(fraud_rate: float) -> tuple[dict, bool]:
    tenant_id = random.choice(TENANTS)
    is_fraud = random.random() < fraud_rate
    account_seed = random.randint(1, 2500)

    if is_fraud:
        amount = random.choice([50_000, 120_000, 250_000, 500_000, 999_000])
        channel = random.choice(["API", "MOBILE", "USSD"])
        country = random.choice([COUNTRIES[tenant_id], "US", "GB"])
        device_id = f"tok_dev_hot_{random.randint(1, 50)}"
    else:
        amount = round(random.lognormvariate(8.6, 0.8), 2)
        channel = random.choice(CHANNELS)
        country = COUNTRIES[tenant_id]
        device_id = f"tok_dev_{random.randint(1, 2000)}"

    txn_id = f"sim_{uuid.uuid4()}"
    payload = {
        "transaction_id": txn_id,
        "tenant_id": tenant_id,
        "account_id": f"tok_acct_{account_seed}",
        "amount": float(max(100.0, amount)),
        "currency": CURRENCIES[tenant_id],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "transaction_type": random.choice(TXN_TYPES),
        "channel": channel,
        "merchant_id": f"tok_merch_{random.randint(1, 400)}",
        "merchant_category_code": random.choice(
            ["5411", "5812", "6011", "5732", "7995"]
        ),
        "device_id": device_id,
        "country_code": country,
        "typing_cadence_ms": float(random.normalvariate(190 if is_fraud else 260, 45)),
        "session_duration_seconds": float(random.uniform(8, 80 if is_fraud else 420)),
        "field_visit_count": random.randint(1, 12),
        "extra_fields": {
            "simulated": True,
            "simulated_label": int(is_fraud),
        },
    }
    return payload, is_fraud


def emit_label(
    client: httpx.Client, api_url: str, txn: dict, confidence: float
) -> None:
    label = {
        "transaction_id": txn["transaction_id"],
        "tenant_id": txn["tenant_id"],
        "label": int(txn["extra_fields"]["simulated_label"]),
        "label_source": "SIMULATOR",
        "labelled_at": datetime.now(timezone.utc).isoformat(),
        "confidence": confidence,
    }
    client.post(f"{api_url}/v1/labels", json=label, timeout=10.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate live FraudTrap traffic")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument(
        "--rate", type=float, default=1.0, help="Transactions per second"
    )
    parser.add_argument("--fraud-rate", type=float, default=0.04)
    parser.add_argument("--label-sample-rate", type=float, default=0.75)
    parser.add_argument("--max-events", type=int, default=0, help="0 means run forever")
    args = parser.parse_args()

    delay = 1.0 / max(args.rate, 0.1)
    sent = 0

    with httpx.Client() as client:
        while args.max_events <= 0 or sent < args.max_events:
            txn, is_fraud = build_transaction(args.fraud_rate)
            try:
                response = client.post(
                    f"{args.api_url}/v1/score", json=txn, timeout=10.0
                )
                response.raise_for_status()
                scored = response.json()
                print(
                    f"{datetime.now(timezone.utc).isoformat()} "
                    f"{txn['tenant_id']} {txn['transaction_id']} "
                    f"label={int(is_fraud)} score={scored['risk_score']:.3f} "
                    f"decision={scored['decision']} latency={scored['latency_ms']}ms",
                    flush=True,
                )

                if is_fraud and random.random() < args.label_sample_rate:
                    emit_label(client, args.api_url, txn, confidence=0.98)
            except Exception as exc:
                print(f"simulator error: {exc}", flush=True)

            sent += 1
            time.sleep(delay)


if __name__ == "__main__":
    main()
