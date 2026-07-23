"""
FraudTrap — Kafka Ingestion Layer
Handles producing scored events and consuming raw transactions.
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from typing import Callable, Optional
from loguru import logger
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import get_settings

settings = get_settings()


def _serialise(obj: dict) -> bytes:
    """JSON serialiser that handles datetime objects."""

    def _default(o):
        if isinstance(o, datetime):
            return o.isoformat()
        raise TypeError(f"Not serialisable: {type(o)}")

    return json.dumps(obj, default=_default).encode("utf-8")


# ── Producer ─────────────────────────────────────────────────────────────────


class FraudTrapProducer:
    """
    Thin wrapper around KafkaProducer.
    Used by the scoring API to emit decisions and audit events.
    """

    def __init__(self):
        self._producer: Optional[KafkaProducer] = None

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def connect(self) -> None:
        self._producer = KafkaProducer(
            bootstrap_servers=settings.kafka_brokers.split(","),
            value_serializer=_serialise,
            acks="all",  # wait for all ISR acknowledgement
            retries=3,
            compression_type="gzip",
            linger_ms=5,  # micro-batching for throughput
            batch_size=65_536,
        )
        logger.info("Kafka producer connected to {}", settings.kafka_brokers)

    def emit(self, topic: str, payload: dict, key: Optional[str] = None) -> None:
        if self._producer is None:
            self.connect()
        key_bytes = key.encode("utf-8") if key else None
        future = self._producer.send(topic, value=payload, key=key_bytes)
        try:
            future.get(timeout=5)
        except KafkaError as exc:
            logger.error("Kafka emit failed on topic={}: {}", topic, exc)
            raise

    def emit_scored_transaction(self, response: dict) -> None:
        self.emit(
            topic=settings.kafka_topic_scored,
            payload=response,
            key=response.get("transaction_id"),
        )

    def emit_audit_event(self, event: dict) -> None:
        self.emit(
            topic=settings.kafka_topic_audit,
            payload={**event, "emitted_at": datetime.now(timezone.utc).isoformat()},
            key=event.get("transaction_id"),
        )

    def close(self) -> None:
        if self._producer:
            self._producer.flush()
            self._producer.close()
            logger.info("Kafka producer closed")


# ── Consumer ─────────────────────────────────────────────────────────────────


class FraudTrapConsumer:
    """
    Generic Kafka consumer. Used by the label ingestion worker
    and the feature computation pipeline.
    """

    def __init__(self, topics: list[str], group_id: str, auto_offset: str = "earliest"):
        self.topics = topics
        self.group_id = group_id
        self.auto_offset = auto_offset
        self._consumer: Optional[KafkaConsumer] = None

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def connect(self) -> None:
        self._consumer = KafkaConsumer(
            *self.topics,
            bootstrap_servers=settings.kafka_brokers.split(","),
            group_id=self.group_id,
            auto_offset_reset=self.auto_offset,
            enable_auto_commit=False,
            value_deserializer=lambda b: json.loads(b.decode("utf-8")),
            max_poll_records=500,
            session_timeout_ms=30_000,
        )
        logger.info("Kafka consumer connected — topics={}", self.topics)

    def consume(self, handler: Callable[[dict], None], batch_size: int = 100) -> None:
        """
        Consume messages in batches, calling `handler` for each.
        Commits only after successful handler execution (at-least-once semantics).
        """
        if self._consumer is None:
            self.connect()

        logger.info("Starting consumption loop")
        for message in self._consumer:
            try:
                payload = message.value
                handler(payload)
                self._consumer.commit()
            except Exception as exc:
                logger.error(
                    "Handler failed for topic={} partition={} offset={}: {}",
                    message.topic,
                    message.partition,
                    message.offset,
                    exc,
                )
                # Do NOT commit — message will be reprocessed on restart

    def close(self) -> None:
        if self._consumer:
            self._consumer.close()
            logger.info("Kafka consumer closed")
