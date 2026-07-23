"""
FraudTrap — Training Runner
Called by the CronJob / Airflow DAG for scheduled retraining.
Also used for manual retraining during development.

Usage:
  python scripts/run_training.py --tenant bank_ng_gtb
  python scripts/run_training.py --all-tenants
  python scripts/run_training.py --generate-sample-data
"""

from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import redis
from loguru import logger

from config.settings import get_settings
from training.pipeline import (
    TrainingPipeline,
    PhaseState,
    ModelPhase,
    _generate_synthetic_data,
)

settings = get_settings()
DATA_DIR = Path("./artifacts/data")
MODEL_DIR = Path("./artifacts/models")


def get_redis_client() -> redis.Redis | None:
    try:
        r = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password or None,
            socket_timeout=2.0,
        )
        r.ping()
        return r
    except Exception:
        logger.warning("Redis unavailable — phase states will not be persisted")
        return None


def load_phase_state(tenant_id: str, r: redis.Redis | None) -> PhaseState:
    if r:
        raw = r.get(f"fraudtrap:phase:{tenant_id}")
        if raw:
            return PhaseState.from_json(raw)
    return PhaseState(
        tenant_id=tenant_id,
        first_transaction_at=datetime.now(timezone.utc).isoformat(),
    )


def save_phase_state(state: PhaseState, r: redis.Redis | None) -> None:
    if r:
        r.set(
            f"fraudtrap:phase:{state.tenant_id}",
            state.to_json(),
            ex=86_400 * 90,
        )


def generate_sample_data(tenant_ids: list[str]) -> None:
    """Generates synthetic Parquet data for testing without real client data."""
    logger.info("Generating synthetic sample data …")
    for tenant in tenant_ids:
        path = DATA_DIR / tenant
        path.mkdir(parents=True, exist_ok=True)
        df = _generate_synthetic_data(n=10_000)
        df.to_parquet(path / "features.parquet", index=False)
        fraud_count = int(df["label"].sum())
        logger.info(
            "Generated {} transactions ({} fraud, {:.2f}%) → {}",
            len(df),
            fraud_count,
            100 * fraud_count / len(df),
            path / "features.parquet",
        )


def run_training_for_tenant(tenant_id: str, r: redis.Redis | None) -> None:
    pipeline = TrainingPipeline()
    state = load_phase_state(tenant_id, r)

    logger.info(
        "Starting training: tenant={} phase={} fraud_labels={}",
        tenant_id,
        state.current_phase.value,
        state.confirmed_fraud_labels,
    )

    updated_state = pipeline.run(tenant_id, state)
    save_phase_state(updated_state, r)

    logger.info(
        "Training complete: tenant={} phase={} → {} metrics={}",
        tenant_id,
        state.current_phase.value,
        updated_state.current_phase.value,
        updated_state.metrics,
    )


def main():
    parser = argparse.ArgumentParser(description="FraudTrap Training Runner")
    parser.add_argument("--tenant", type=str, help="Train a specific tenant")
    parser.add_argument(
        "--all-tenants", action="store_true", help="Train all known tenants"
    )
    parser.add_argument(
        "--generate-sample-data",
        action="store_true",
        help="Generate synthetic training data for demo tenants",
    )
    parser.add_argument(
        "--force-phase",
        type=str,
        choices=["UNSUPERVISED", "SEMI_SUPERVISED", "SUPERVISED"],
        help="Force a specific phase (for testing)",
    )
    args = parser.parse_args()

    demo_tenants = ["bank_ng_gtb", "bank_ke_equity", "fintech_za_yoco"]

    if args.generate_sample_data:
        generate_sample_data(demo_tenants)
        return

    r = get_redis_client()

    if args.all_tenants:
        tenants = demo_tenants
    elif args.tenant:
        tenants = [args.tenant]
    else:
        # Default: process retrain queue from Redis
        if r:
            tenants_queued = set()
            while True:
                item = r.rpop("fraudtrap:retrain:queue")
                if not item:
                    break
                req = json.loads(item)
                tenants_queued.add(req["tenant_id"])
                logger.info(
                    "Dequeued retrain for tenant={} trigger={}",
                    req["tenant_id"],
                    req["trigger"],
                )
            tenants = list(tenants_queued)
        else:
            tenants = demo_tenants
            logger.info("No Redis queue; defaulting to demo tenants: {}", tenants)

    if not tenants:
        logger.info("No tenants to retrain. Exiting.")
        return

    for tenant_id in tenants:
        try:
            if args.force_phase and r:
                state = load_phase_state(tenant_id, r)
                state.current_phase = ModelPhase(args.force_phase)
                # Seed labels so gates pass
                if args.force_phase == "SEMI_SUPERVISED":
                    state.confirmed_fraud_labels = settings.phase1_min_fraud_labels + 1
                elif args.force_phase == "SUPERVISED":
                    state.confirmed_fraud_labels = settings.phase2_min_fraud_labels + 1
                save_phase_state(state, r)
                logger.info(
                    "Forced phase={} for tenant={}", args.force_phase, tenant_id
                )

            run_training_for_tenant(tenant_id, r)
        except Exception as exc:
            logger.exception("Training failed for tenant={}: {}", tenant_id, exc)
            continue

    logger.info("All training runs complete.")


if __name__ == "__main__":
    main()
