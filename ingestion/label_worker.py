"""
FraudTrap — Label Ingestion Worker
Consumes the labels Kafka topic, validates label quality,
updates phase state counters, and writes to ClickHouse for training.
Runs as a long-lived background process alongside the API.
"""

from __future__ import annotations
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import redis
from loguru import logger

from config.settings import get_settings
from ingestion.kafka_client import FraudTrapConsumer
from ingestion.schema import LabelPayload
from training.pipeline import PhaseState, ModelPhase, TrainingPipeline

settings = get_settings()

# Chargeback reason codes that indicate actual fraud (not merchant disputes)
FRAUD_REASON_CODES = {
    "4863",  # Visa: Cardholder Does Not Recognize
    "10.4",  # Visa: Other Fraud - Card Absent
    "10.5",  # Visa: Visa Fraud Monitoring Program
    "37",  # Mastercard: No Cardholder Authorization
    "4853",  # Mastercard: Cardholder Dispute (sometimes fraud)
    "UA01",  # Discover: Fraud - Card Present
    "UA02",  # Discover: Fraud - Card Not Present
    "4807",  # Visa: Warning Bulletin File
    "4808",  # Visa: Authorization-Related Chargeback
}

# Parquet / ClickHouse writer placeholder
LABEL_STORE_PATH = Path("./artifacts/labels")


class LabelWorker:
    """
    Stateful worker that processes incoming labels and updates tenant phase states.
    Phase states are persisted in Redis and checked by the training pipeline.
    """

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self._redis = redis_client
        self._consumer = FraudTrapConsumer(
            topics=[settings.kafka_topic_labels],
            group_id="fraudtrap-label-worker",
            auto_offset="earliest",
        )
        self._phase_states: dict[str, PhaseState] = {}
        LABEL_STORE_PATH.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        """Start consuming labels. Blocking call — run in a dedicated process/thread."""
        logger.info("LabelWorker starting …")
        self._consumer.connect()
        self._consumer.consume(handler=self._handle_label)

    def _handle_label(self, payload: dict) -> None:
        try:
            label = LabelPayload(**payload)
        except Exception as exc:
            logger.warning("Invalid label payload: {} — {}", payload, exc)
            return

        # ── Quality gate: filter non-fraud chargebacks ────────────────────────
        if label.label_source == "CHARGEBACK":
            if (
                label.chargeback_reason_code
                and label.chargeback_reason_code not in FRAUD_REASON_CODES
            ):
                logger.debug(
                    "Skipping non-fraud chargeback reason_code={} txn={}",
                    label.chargeback_reason_code,
                    label.transaction_id,
                )
                return

        # ── Update phase state ────────────────────────────────────────────────
        state = self._get_or_load_state(label.tenant_id)
        if label.label == 1:
            state.confirmed_fraud_labels += 1
        self._save_state(state)

        # ── Write label to store ──────────────────────────────────────────────
        self._write_label(label)

        logger.debug(
            "Label processed: tenant={} txn={} label={} source={} "
            "total_fraud_labels={}",
            label.tenant_id,
            label.transaction_id,
            label.label,
            label.label_source,
            state.confirmed_fraud_labels,
        )

        # ── Check retraining trigger ──────────────────────────────────────────
        self._maybe_trigger_retrain(state)

    def _get_or_load_state(self, tenant_id: str) -> PhaseState:
        if tenant_id in self._phase_states:
            return self._phase_states[tenant_id]

        if self._redis:
            raw = self._redis.get(f"fraudtrap:phase:{tenant_id}")
            if raw:
                state = PhaseState.from_json(raw)
                self._phase_states[tenant_id] = state
                return state

        state = PhaseState(
            tenant_id=tenant_id,
            first_transaction_at=datetime.now(timezone.utc).isoformat(),
        )
        self._phase_states[tenant_id] = state
        return state

    def _save_state(self, state: PhaseState) -> None:
        self._phase_states[state.tenant_id] = state
        if self._redis:
            try:
                self._redis.set(
                    f"fraudtrap:phase:{state.tenant_id}",
                    state.to_json(),
                    ex=86_400 * 90,  # 90 day TTL
                )
            except Exception as exc:
                logger.warning("Failed to persist phase state to Redis: {}", exc)

    def _write_label(self, label: LabelPayload) -> None:
        """
        Append label to the parquet store for this tenant.
        In production: write to ClickHouse via INSERT INTO fraudtrap.labels.
        """
        tenant_path = LABEL_STORE_PATH / label.tenant_id
        tenant_path.mkdir(parents=True, exist_ok=True)

        record = {
            "transaction_id": label.transaction_id,
            "tenant_id": label.tenant_id,
            "label": label.label,
            "label_source": label.label_source,
            "chargeback_reason_code": label.chargeback_reason_code,
            "labelled_at": label.labelled_at.isoformat(),
            "confidence": label.confidence,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }

        # Append to JSONL (parquet writer would replace this in production)
        with open(tenant_path / "labels.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

    def _maybe_trigger_retrain(self, state: PhaseState) -> None:
        """
        Trigger an immediate retrain if a phase transition threshold is just crossed.
        This catches the moment when Phase 1 → 2 or Phase 2 → 3 gates open.
        """
        phase1_gate = settings.phase1_min_fraud_labels
        phase2_gate = settings.phase2_min_fraud_labels

        if state.current_phase == ModelPhase.UNSUPERVISED:
            if state.confirmed_fraud_labels == phase1_gate:
                logger.info(
                    "Phase 1 fraud label threshold reached for tenant={}. "
                    "Queueing retrain.",
                    state.tenant_id,
                )
                self._queue_retrain(state.tenant_id, "phase_gate")

        elif state.current_phase == ModelPhase.SEMI_SUPERVISED:
            if state.confirmed_fraud_labels == phase2_gate:
                logger.info(
                    "Phase 2 fraud label threshold reached for tenant={}. "
                    "Queueing retrain.",
                    state.tenant_id,
                )
                self._queue_retrain(state.tenant_id, "phase_gate")

    def _queue_retrain(self, tenant_id: str, trigger: str) -> None:
        """Write a retrain request to Redis for the training worker to pick up."""
        if self._redis:
            self._redis.lpush(
                "fraudtrap:retrain:queue",
                json.dumps(
                    {
                        "tenant_id": tenant_id,
                        "trigger": trigger,
                        "queued_at": datetime.now(timezone.utc).isoformat(),
                    }
                ),
            )


if __name__ == "__main__":
    import redis as redis_lib
    from config.settings import get_settings

    s = get_settings()
    try:
        r = redis_lib.Redis(host=s.redis_host, port=s.redis_port)
        r.ping()
    except Exception:
        r = None
        logger.warning(
            "Redis unavailable — running label worker without state persistence"
        )
    worker = LabelWorker(redis_client=r)
    worker.start()
