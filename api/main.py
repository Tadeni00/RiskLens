"""
FraudTrap — FastAPI Scoring API
Primary integration point for banks and fintechs.
Enforces 90ms scoring SLA. Emits audit events to Kafka.

Run locally:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from loguru import logger

from config.settings import get_settings
from ingestion.schema import TransactionRequest, ScoringResponse, LabelPayload
from scoring.orchestrator import ScoringOrchestrator
from api.admin import router as admin_router

settings = get_settings()

# ── Prometheus metrics ─────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "fraudtrap_requests_total", "Total scoring requests", ["tenant_id", "decision"]
)
LATENCY_HIST = Histogram(
    "fraudtrap_latency_ms",
    "Scoring latency (ms)",
    buckets=[10, 25, 50, 75, 90, 100, 150, 200],
)
LABEL_COUNT = Counter("fraudtrap_labels_received_total", "Labels received", ["source"])

# ── Global singleton ──────────────────────────────────────────────────────────
_orchestrator: Optional[ScoringOrchestrator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator
    logger.info("FraudTrap API starting up …")
    _orchestrator = ScoringOrchestrator()
    _orchestrator.registry.load_from_disk(Path(settings.model_dir))
    _orchestrator._get_redis()  # warm up Redis connection
    logger.info("ScoringOrchestrator ready")
    yield
    logger.info("FraudTrap API shutting down …")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FraudTrap API",
    description="Real-time fraud detection for banks and fintechs.",
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Include admin router
app.include_router(admin_router, prefix="/v1/admin", tags=["admin"])


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["ops"])
async def health():
    return {
        "status": "ok",
        "version": settings.app_version,
        "environment": settings.environment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/metrics", tags=["ops"])
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/recent", tags=["ops"])
async def recent_scores(limit: int = 500, tenant_id: str | None = None):
    rows = _orchestrator.recent_scores(limit=limit)
    if tenant_id and tenant_id != "all_tenants":
        rows = [row for row in rows if row.get("tenant_id") == tenant_id]
    return {"count": len(rows), "items": rows}


# ── Core scoring endpoint ──────────────────────────────────────────────────────
@app.post(
    "/v1/score",
    response_model=ScoringResponse,
    tags=["scoring"],
    summary="Score a transaction for fraud risk",
    description=(
        "Submit a transaction payload. Returns a fraud risk score (0–1), "
        "a decision (APPROVE / REVIEW / BLOCK), and a SHAP explanation. "
        "P95 latency target: <100ms."
    ),
)
async def score_transaction(txn: TransactionRequest) -> ScoringResponse:
    try:
        response = _orchestrator.score(txn)

        REQUEST_COUNT.labels(tenant_id=txn.tenant_id, decision=response.decision).inc()
        LATENCY_HIST.observe(response.latency_ms)

        return response

    except Exception as exc:
        logger.exception("Scoring error for txn={}: {}", txn.transaction_id, exc)
        raise HTTPException(status_code=500, detail="Scoring service error")


# ── Batch scoring ─────────────────────────────────────────────────────────────
@app.post("/v1/score/batch", tags=["scoring"])
async def score_batch(transactions: list[TransactionRequest]) -> list[ScoringResponse]:
    if len(transactions) > 1000:
        raise HTTPException(status_code=400, detail="Batch size exceeds 1000")
    results = []
    for txn in transactions:
        try:
            results.append(_orchestrator.score(txn))
        except Exception as exc:
            logger.error("Batch scoring error txn={}: {}", txn.transaction_id, exc)
    return results


# ── Label ingestion ───────────────────────────────────────────────────────────
@app.post("/v1/labels", tags=["labels"], status_code=202)
async def ingest_label(label: LabelPayload):
    """
    Receive ground-truth labels from chargeback systems or manual review.
    Labels are written to Kafka for the training pipeline.
    """
    LABEL_COUNT.labels(source=label.label_source).inc()

    try:
        from ingestion.kafka_client import FraudTrapProducer

        producer = FraudTrapProducer()
        producer.connect()
        producer.emit(
            topic=settings.kafka_topic_labels,
            payload=label.model_dump(mode="json"),
            key=label.transaction_id,
        )
        producer.close()
    except Exception as exc:
        logger.warning("Label Kafka emit failed (non-fatal): {}", exc)

    logger.info(
        "Label received: txn={} tenant={} label={} source={}",
        label.transaction_id,
        label.tenant_id,
        label.label,
        label.label_source,
    )
    return {"status": "accepted", "transaction_id": label.transaction_id}


# ── Phase status ──────────────────────────────────────────────────────────────
@app.get("/v1/phase/{tenant_id}", tags=["ops"])
async def get_phase_status(tenant_id: str):
    """Returns the current model phase and version for a tenant."""
    registry = _orchestrator.registry
    tenant_phase = registry.active_phase
    if tenant_id in registry.champion_models:
        tenant_phase = "SUPERVISED"
    elif tenant_id in registry.adaptive_learner_models:
        tenant_phase = "ADAPTIVE_LEARNING"
    elif tenant_id in registry.cold_start_models:
        tenant_phase = "UNSUPERVISED"
    elif tenant_id in registry.simple_models:
        tenant_phase = "SUPERVISED"

    return {
        "tenant_id": tenant_id,
        "current_phase": tenant_phase,
        "model_version": registry.get_model_version(tenant_id),
        "loaded_models": {
            "cold_start": tenant_id in registry.cold_start_models
            or registry.cold_start is not None,
            "adaptive_learning": tenant_id in registry.adaptive_learner_models
            or registry.adaptive_learner is not None,
            "supervised": tenant_id in registry.champion_models or registry.champion is not None,
            "simple_model": tenant_id in registry.simple_models,
        },
        "available_model_tenants": {
            "cold_start": sorted(registry.cold_start_models),
            "adaptive_learning": sorted(registry.adaptive_learner_models),
            "supervised": sorted(registry.champion_models),
            "simple_model": sorted(registry.simple_models),
        },
    }


# ── Explanation lookup ────────────────────────────────────────────────────────
@app.get("/v1/explain/{trace_id}", tags=["explainability"])
async def get_explanation(trace_id: str):
    """
    Retrieve the SHAP explanation for any scored transaction by trace_id.
    """
    all_scores = _orchestrator.recent_scores(limit=1000)
    for score in all_scores:
        if score.get("trace_id") == trace_id:
            # Check if explanation exists
            explanation = score.get("explanation")
            if explanation:
                return {
                    "trace_id": trace_id,
                    "explanation": explanation,
                    "transaction_id": score.get("transaction_id"),
                    "risk_score": score.get("risk_score"),
                    "decision": score.get("decision"),
                }
            else:
                return {
                    "trace_id": trace_id,
                    "message": "Transaction found, but no explanation was stored.",
                }

    return {
        "trace_id": trace_id,
        "message": "Trace ID not found in recent scores cache.",
    }


# ── Drift metrics ─────────────────────────────────────────────────────────────
@app.get("/v1/drift/{tenant_id}", tags=["ops"])
async def get_drift(tenant_id: str):
    """
    Returns real-time drift metrics (PSI) by comparing the oldest half of
    recent scores (baseline) against the newest half (current).
    """
    import numpy as np

    all_scores = _orchestrator.recent_scores(limit=5000)

    if tenant_id and tenant_id != "all_tenants":
        scores = [s for s in all_scores if s.get("tenant_id") == tenant_id]
    else:
        scores = all_scores

    if len(scores) < 100:
        return {
            "status": "insufficient_data",
            "message": "Need at least 100 transactions to compute drift",
        }

    # Sort scores by time (oldest first)
    # The cache might be reverse chronological, so let's parse timestamp and sort
    def parse_ts(s):
        ts = s.get("scored_at", s.get("timestamp"))
        if not ts:
            return ""
        return str(ts)

    scores = sorted(scores, key=parse_ts)

    midpoint = len(scores) // 2
    baseline = scores[:midpoint]
    current = scores[midpoint:]

    features_to_monitor = [
        "amount",
        "acct_v_1h_count",
        "acct_v_24h_total_amt",
        "geo_speed_kmh",
        "typing_zscore",
        "device_account_count",
    ]

    drift_data = {}

    for feat in features_to_monitor:
        base_vals = [float(s.get(feat, 0.0)) for s in baseline if s.get(feat) is not None]
        curr_vals = [float(s.get(feat, 0.0)) for s in current if s.get(feat) is not None]

        if not base_vals or not curr_vals:
            continue

        base_mean = float(np.mean(base_vals))
        curr_mean = float(np.mean(curr_vals))

        # Simple PSI approximation
        # 1. Create decile bins based on baseline
        try:
            bins = np.percentile(base_vals, np.linspace(0, 100, 11))
            bins[0] -= 0.001  # avoid out of bounds
            bins[-1] += 0.001

            base_counts, _ = np.histogram(base_vals, bins=bins)
            curr_counts, _ = np.histogram(curr_vals, bins=bins)

            base_pct = base_counts / len(base_vals)
            curr_pct = curr_counts / len(curr_vals)

            # Avoid division by zero and log(0)
            base_pct = np.clip(base_pct, 0.001, 1.0)
            curr_pct = np.clip(curr_pct, 0.001, 1.0)

            psi = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))

            drift_data[feat] = {
                "psi": float(psi),
                "baseline_mean": base_mean,
                "current_mean": curr_mean,
            }
        except Exception:
            pass

    return {
        "tenant_id": tenant_id,
        "n_baseline": len(baseline),
        "n_current": len(current),
        "metrics": drift_data,
    }


# ── Lifecycle status ──────────────────────────────────────────────────────────
@app.get("/v1/lifecycle/{tenant_id}", tags=["ops"])
async def get_lifecycle(tenant_id: str):
    """
    Returns real-time lifecycle metrics for the dashboard:
    label counts, phase progression, scoring distribution, and
    transition readiness computed from the live stream.
    """
    import numpy as np

    registry = _orchestrator.registry
    all_scores = _orchestrator.recent_scores(limit=5000)

    # Filter to tenant if not "all_tenants"
    if tenant_id and tenant_id != "all_tenants":
        scores = [s for s in all_scores if s.get("tenant_id") == tenant_id]
    else:
        scores = all_scores

    total_scored = len(scores)
    fraud_labels = sum(1 for s in scores if int(s.get("is_fraud", 0)) == 1)
    legit_labels = total_scored - fraud_labels

    # Decision breakdown
    decisions = {}
    for s in scores:
        d = s.get("decision", "UNKNOWN")
        decisions[d] = decisions.get(d, 0) + 1

    # Phase breakdown from actual stream
    phase_counts = {}
    for s in scores:
        p = s.get("model_phase", "UNKNOWN")
        phase_counts[p] = phase_counts.get(p, 0) + 1

    # Compute a live PR-AUC estimate from scored transactions
    pr_auc = 0.0
    if fraud_labels > 0 and legit_labels > 0:
        try:
            y_true = [int(s.get("is_fraud", 0)) for s in scores]
            y_score = [float(s.get("risk_score", 0.0)) for s in scores]
            # Sort by descending score
            paired = sorted(zip(y_score, y_true), reverse=True)
            tp, fp, total_pos = 0, 0, sum(y_true)
            precisions, recalls = [1.0], [0.0]
            for score_val, label in paired:
                if label == 1:
                    tp += 1
                else:
                    fp += 1
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / total_pos if total_pos > 0 else 0.0
                precisions.append(prec)
                recalls.append(rec)
            # Trapezoidal AUC
            pr_auc = 0.0
            for i in range(1, len(recalls)):
                pr_auc += (recalls[i] - recalls[i - 1]) * (precisions[i] + precisions[i - 1]) / 2
            pr_auc = max(0.0, min(1.0, pr_auc))
        except Exception:
            pr_auc = 0.0

    # Latency stats
    latencies = [float(s.get("latency_ms", 0)) for s in scores if s.get("latency_ms")]
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0

    # Determine current phase
    has_simple = (
        tenant_id in registry.simple_models
        if tenant_id != "all_tenants"
        else bool(registry.simple_models)
    )
    current_phase = "SUPERVISED" if has_simple else registry.active_phase

    # Scoring history — compute from time buckets in the stream
    scoring_history = []
    if scores:
        from collections import defaultdict

        buckets = defaultdict(lambda: {"count": 0, "fraud": 0, "total_score": 0.0})
        for s in scores:
            ts = s.get("scored_at", s.get("timestamp", ""))
            if isinstance(ts, str) and len(ts) >= 10:
                day = ts[:10]
            else:
                day = "unknown"
            buckets[day]["count"] += 1
            buckets[day]["fraud"] += int(s.get("is_fraud", 0))
            buckets[day]["total_score"] += float(s.get("risk_score", 0.0))

        for day in sorted(buckets.keys()):
            b = buckets[day]
            scoring_history.append(
                {
                    "date": day,
                    "transactions": b["count"],
                    "fraud_labels": b["fraud"],
                    "avg_score": (round(b["total_score"] / b["count"], 4) if b["count"] else 0.0),
                }
            )

    # Transition readiness gates
    phase2_label_target = settings.phase2_min_fraud_labels  # 5000
    phase2_pr_auc_target = settings.phase2_min_pr_auc  # 0.78
    label_pct = min(fraud_labels / phase2_label_target * 100, 100) if phase2_label_target else 0
    pr_auc_pct = min(pr_auc / phase2_pr_auc_target * 100, 100) if phase2_pr_auc_target else 0
    champion_pct = 100.0 if current_phase == "SUPERVISED" and has_simple else 0.0

    return {
        "tenant_id": tenant_id,
        "current_phase": current_phase,
        "model_version": (
            registry.get_model_version(tenant_id)
            if tenant_id != "all_tenants"
            else registry.model_version
        ),
        "total_scored": total_scored,
        "fraud_labels": fraud_labels,
        "legit_labels": legit_labels,
        "decisions": decisions,
        "phase_counts": phase_counts,
        "pr_auc": round(pr_auc, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "scoring_history": scoring_history,
        "transition_readiness": {
            "fraud_labels": {
                "current": fraud_labels,
                "target": phase2_label_target,
                "pct": round(label_pct, 1),
            },
            "pr_auc": {
                "current": round(pr_auc, 4),
                "target": phase2_pr_auc_target,
                "pct": round(pr_auc_pct, 1),
            },
            "champion_challenger": {
                "current": "deployed" if champion_pct > 0 else "pending",
                "pct": round(champion_pct, 1),
            },
        },
        "loaded_models": {
            "cold_start": registry.cold_start is not None,
            "adaptive_learning": registry.adaptive_learner is not None,
            "supervised": registry.supervised is not None,
            "simple_model": has_simple,
        },
        "available_tenants": sorted(registry.simple_models.keys()),
    }
