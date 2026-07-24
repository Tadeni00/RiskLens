"""
FraudTrap — Admin API Endpoints
Management endpoints for rules, blocklists, and model operations.
"""

from __future__ import annotations
from typing import Optional, Annotated
from fastapi import APIRouter, Depends, HTTPException, Body, Query
from pydantic import BaseModel, Field
from datetime import datetime, timezone

from scoring.rules_engine import (
    RulesEngine,
    BlocklistManager,
    BlocklistEntry,
    HeuristicConfig,
    create_rules_engine,
    create_blocklist_manager,
)
from scoring.rules_config import RuleType
from scoring.orchestrator import get_registry, ScoringOrchestrator
from scoring.version_manager import get_version_manager, VersionManager
from config.settings import get_settings

router = APIRouter(prefix="/v1/admin", tags=["admin"])

settings = get_settings()


# ─── Dependency Injection ────────────────────────────────────────────────────


def get_rules_engine() -> RulesEngine:
    """Get rules engine instance."""
    return create_rules_engine()


def get_blocklist_manager() -> BlocklistManager:
    """Get blocklist manager."""
    # In production, inject Redis client from app state
    import redis
    from config.settings import get_settings

    s = get_settings()
    r = redis.Redis(
        host=s.redis_host,
        port=s.redis_port,
        password=s.redis_password or None,
        db=s.redis_db,
        decode_responses=True,
    )
    return create_blocklist_manager(r)


def get_version_manager() -> VersionManager:
    return get_version_manager()


# ─── Rules Management ────────────────────────────────────────────────────────


class RuleCreateRequest(BaseModel):
    """Request to create/update a rule."""

    id: str
    description: str
    type: str
    action: str
    severity: str = "high"
    boost: float = 0.05
    max_boost: float = 0.20
    score_override: Optional[float] = None
    enabled: bool = True
    tags: list[str] = []
    metadata: dict = {}

    # Type-specific configs
    blocklist: Optional[dict] = None
    threshold: Optional[dict] = None
    expression: Optional[dict] = None
    velocity: Optional[dict] = None
    geo: Optional[dict] = None


class RuleResponse(BaseModel):
    id: str
    description: str
    type: str
    action: str
    severity: str
    boost: float
    max_boost: float
    enabled: bool
    tags: list[str]
    metadata: dict


@router.get("/rules", response_model=list[RuleResponse])
async def list_rules(
    engine: RulesEngine = Depends(get_rules_engine),
):
    """List all active rules."""
    rules = []
    for rule in engine._rules:
        rules.append(
            RuleResponse(
                id=rule.id,
                description=rule.description,
                type=rule.type.value,
                action=rule.action.value,
                severity=rule.severity.value,
                boost=rule.boost,
                max_boost=rule.max_boost,
                enabled=rule.enabled,
                tags=rule.tags,
                metadata=rule.metadata,
            )
        )
    return rules


@router.post("/rules", response_model=RuleResponse)
async def create_rule(
    request: RuleCreateRequest,
    engine: RulesEngine = Depends(get_rules_engine),
):
    """Create or update a rule."""
    from scoring.rules_config import RuleDefinition

    # Check if rule exists
    existing = next((r for r in engine._rules if r.id == request.id), None)

    # Build rule
    rule = RuleDefinition(
        id=request.id,
        description=request.description,
        type=request.type,
        action=request.action,
        severity=request.severity,
        boost=request.boost,
        max_boost=request.max_boost,
        score_override=request.score_override,
        enabled=request.enabled,
        tags=request.tags,
        metadata=request.metadata,
    )

    # Add type-specific config
    if request.blocklist:
        from scoring.rules_config import BlocklistConfig

        rule.blocklist = BlocklistConfig(**request.blocklist)
    if request.threshold:
        from scoring.rules_config import ThresholdConfig

        rule.threshold = ThresholdConfig(**request.threshold)
    if request.expression:
        from scoring.rules_config import ExpressionConfig

        rule.expression = ExpressionConfig(**request.expression)
    if request.velocity:
        from scoring.rules_config import VelocityConfig

        rule.velocity = VelocityConfig(**request.velocity)
    if request.geo:
        from scoring.rules_config import GeoConfig

        rule.geo = GeoConfig(**request.geo)

    if existing:
        # Update existing
        engine._rules = [r for r in engine._rules if r.id != request.id]

    engine._rules.append(rule)
    engine.save_rules(engine._rules)

    return RuleResponse(
        id=rule.id,
        description=rule.description,
        type=rule.type.value,
        action=rule.action.value,
        severity=rule.severity.value,
        boost=rule.boost,
        max_boost=rule.max_boost,
        enabled=rule.enabled,
        tags=rule.tags,
        metadata=rule.metadata,
    )


@router.delete("/rules/{rule_id}")
async def delete_rule(
    rule_id: str,
    engine: RulesEngine = Depends(get_rules_engine),
):
    """Delete a rule."""
    engine._rules = [r for r in engine._rules if r.id != rule_id]
    engine.save_rules(engine._rules)
    return {"status": "deleted", "rule_id": rule_id}


@router.post("/rules/reload")
async def reload_rules(
    engine: RulesEngine = Depends(get_rules_engine),
):
    """Force reload rules from file/Redis."""
    engine.reload_rules()
    return {"status": "reloaded", "count": len(engine._rules)}


# ─── Blocklist Management ────────────────────────────────────────────────────


class BlocklistAddRequest(BaseModel):
    list_name: str
    value: str
    tenant_id: str
    added_by: str
    reason: str
    ttl_days: Optional[int] = None


class BlocklistResponse(BaseModel):
    value: str
    list_name: str
    tenant_id: str
    added_by: str
    reason: str
    added_at: datetime
    expires_at: Optional[datetime]
    is_active: bool


@router.post("/blocklist", response_model=BlocklistResponse)
async def add_blocklist_entry(
    request: BlocklistAddRequest,
    manager: BlocklistManager = Depends(get_blocklist_manager),
):
    """Add entry to blocklist with TTL and audit."""
    entry = manager.add(
        list_name=request.list_name,
        value=request.value,
        tenant_id=request.tenant_id,
        added_by=request.added_by,
        reason=request.reason,
        ttl_days=request.ttl_days,
    )
    return BlocklistResponse(
        value=entry.value,
        list_name=entry.list_name,
        tenant_id=entry.tenant_id,
        added_by=entry.added_by,
        reason=entry.reason,
        added_at=entry.added_at,
        expires_at=entry.expires_at,
        is_active=entry.is_active,
    )


@router.delete("/blocklist/{list_name}/{value}")
async def remove_blocklist_entry(
    list_name: str,
    value: str,
    tenant_id: str = Query(...),
    removed_by: str = Query(...),
    reason: str = Query(...),
    manager: BlocklistManager = Depends(get_blocklist_manager),
):
    """Remove entry from blocklist."""
    removed = manager.remove(list_name, value, tenant_id, removed_by, reason)
    if not removed:
        raise HTTPException(status_code=404, detail="Entry not found")
    return {"status": "removed", "list_name": list_name, "value": value}


@router.get("/blocklist/{list_name}")
async def list_blocklist_entries(
    list_name: str,
    tenant_id: str = Query(...),
    active_only: bool = True,
    manager: BlocklistManager = Depends(get_blocklist_manager),
):
    """List all entries in a blocklist."""
    entries = manager.list_entries(list_name, tenant_id, active_only)
    return [
        BlocklistResponse(
            value=e.value,
            list_name=e.list_name,
            tenant_id=e.tenant_id,
            added_by=e.added_by,
            reason=e.reason,
            added_at=e.added_at,
            expires_at=e.expires_at,
            is_active=e.is_active,
        )
        for e in entries
    ]


@router.get("/blocklist/{list_name}/{value}")
async def get_blocklist_entry(
    list_name: str,
    value: str,
    tenant_id: str = Query(...),
    manager: BlocklistManager = Depends(get_blocklist_manager),
):
    """Get blocklist entry metadata."""
    entry = manager.get_entry(list_name, value, tenant_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    return BlocklistResponse(
        value=entry.value,
        list_name=entry.list_name,
        tenant_id=entry.tenant_id,
        added_by=entry.added_by,
        reason=entry.reason,
        added_at=entry.added_at,
        expires_at=entry.expires_at,
        is_active=entry.is_active,
    )


@router.get("/blocklist/audit/{tenant_id}")
async def get_blocklist_audit(
    tenant_id: str,
    limit: int = 100,
    manager: BlocklistManager = Depends(get_blocklist_manager),
):
    """Get blocklist audit log."""
    return manager.get_audit_log(tenant_id, limit)


# ─── Heuristic Configuration ─────────────────────────────────────────────────


class HeuristicConfigResponse(BaseModel):
    base_score: float
    components: dict
    max_score: float


@router.get("/heuristic", response_model=HeuristicConfigResponse)
async def get_heuristic_config():
    """Get current heuristic configuration."""
    return HeuristicConfigResponse(
        base_score=DEFAULT_HEURISTIC.base_score,
        components=DEFAULT_HEURISTIC.components,
        max_score=DEFAULT_HEURISTIC.max_score,
    )


class HeuristicUpdateRequest(BaseModel):
    base_score: Optional[float] = None
    components: Optional[dict] = None
    max_score: Optional[float] = None


@router.post("/heuristic")
async def update_heuristic_config(
    request: HeuristicUpdateRequest,
):
    """Update heuristic configuration."""
    global DEFAULT_HEURISTIC

    if request.base_score is not None:
        DEFAULT_HEURISTIC.base_score = request.base_score
    if request.components is not None:
        DEFAULT_HEURISTIC.components = request.components
    if request.max_score is not None:
        DEFAULT_HEURISTIC.max_score = request.max_score

    return HeuristicConfigResponse(
        base_score=DEFAULT_HEURISTIC.base_score,
        components=DEFAULT_HEURISTIC.components,
        max_score=DEFAULT_HEURISTIC.max_score,
    )


# ─── Model Management ────────────────────────────────────────────────────────


@router.post("/models/reload")
async def reload_models(
    orchestrator: ScoringOrchestrator = Depends(lambda: ScoringOrchestrator()),
):
    """Force reload all models from disk."""
    registry = orchestrator.registry
    registry.load_from_disk(registry.model_dir)
    return {"status": "reloaded", "active_phase": registry.active_phase}


@router.get("/models/status")
async def get_model_status(
    orchestrator: ScoringOrchestrator = Depends(lambda: ScoringOrchestrator()),
    vm: VersionManager = Depends(get_version_manager),
):
    """Get model loading status and versions."""
    registry = orchestrator.registry
    vm_summary = vm.get_version_summary()

    return {
        "active_phase": registry.active_phase,
        "model_version": registry.model_version,
        "feature_names": registry.feature_names[:20],  # Truncated
        "tenant_versions": vm_summary,
        "loaded_models": {
            "simple": list(registry.simple_models.keys()),
            "cold_start": list(registry.cold_start_models.keys()),
            "adaptive_learning": list(registry.adaptive_learner_models.keys()),
            "supervised": list(registry.champion_models.keys()),
            "gnn": "loaded" if registry.gnn_scorer else "not_loaded",
        },
    }


@router.post("/models/validate")
async def validate_model_features(
    tenant_id: str,
    model_type: str,
    features: list[str],
    vm: VersionManager = Depends(get_version_manager),
):
    """Validate feature compatibility for a model."""
    feature_hash = hashlib.sha256("|".join(sorted(features)).encode()).hexdigest()[:16]
    is_compatible, message = vm.validate_feature_compatibility(tenant_id, model_type, feature_hash)

    return {
        "tenant_id": tenant_id,
        "model_type": model_type,
        "live_hash": feature_hash,
        "compatible": is_compatible,
        "message": message,
    }


# ─── Health & Diagnostics ────────────────────────────────────────────────────


@router.get("/health")
async def admin_health():
    """Admin health check with component status."""
    import redis
    from config.settings import get_settings
    from clickhouse_driver import Client

    s = get_settings()

    # Check Redis
    redis_ok = False
    try:
        r = redis.Redis(
            host=s.redis_host,
            port=s.redis_port,
            password=s.redis_password or None,
            db=s.redis_db,
            socket_timeout=1.0,
            socket_connect_timeout=1.0,
        )
        r.ping()
        redis_ok = True
    except Exception:
        pass

    # Check ClickHouse
    ch_ok = False
    try:
        ch = Client(
            host=s.clickhouse_host,
            port=s.clickhouse_port,
            database=s.clickhouse_database,
            user=s.clickhouse_user,
            password=s.clickhouse_password or "",
        )
        ch.execute("SELECT 1")
        ch_ok = True
    except Exception:
        pass

    # Check PostgreSQL
    pg_ok = False
    try:
        import psycopg2

        conn = psycopg2.connect(s.postgres_url, connect_timeout=2)
        conn.close()
        pg_ok = True
    except Exception:
        pass

    return {
        "status": "healthy" if all([redis_ok, ch_ok, pg_ok]) else "degraded",
        "components": {
            "redis": "healthy" if redis_ok else "unhealthy",
            "clickhouse": "healthy" if ch_ok else "unhealthy",
            "postgresql": "healthy" if pg_ok else "unhealthy",
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/stats")
async def admin_stats(
    engine: RulesEngine = Depends(get_rules_engine),
    bm: BlocklistManager = Depends(get_blocklist_manager),
):
    """Get admin statistics."""
    return {
        "rules": {
            "total": len(engine._rules),
            "enabled": sum(1 for r in engine._rules if r.enabled),
            "by_type": {
                t.value: sum(1 for r in engine._rules if r.type.value == t.value)
                for t in [
                    RuleType.BLOCKLIST,
                    RuleType.THRESHOLD,
                    RuleType.EXPRESSION,
                    RuleType.VELOCITY,
                    RuleType.GEO,
                ]
            },
        },
        "blocklists": {
            "total_tenants": "N/A",  # Would need SCAN
        },
        "heuristic": {
            "base_score": DEFAULT_HEURISTIC.base_score,
            "components": len(DEFAULT_HEURISTIC.components),
        },
    }
