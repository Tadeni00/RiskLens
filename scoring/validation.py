"""
RiskLens — Feature Schema Validation
Validates incoming feature vectors against registered schemas.
"""

from __future__ import annotations
import logging
from typing import Optional

from loguru import logger

from features.schema_registry import FeatureSchemaRegistry, ValidationResult

# Global registry instance
_schema_registry: Optional[FeatureSchemaRegistry] = None


def get_schema_registry() -> FeatureSchemaRegistry:
    """Get or create the global schema registry."""
    global _schema_registry
    if _schema_registry is None:
        _schema_registry = FeatureSchemaRegistry()
    return _schema_registry


def _compute_feature_hash(feature_names: list[str]) -> str:
    """Compute SHA256 hash of ordered feature names."""
    import hashlib

    hash_input = "|".join(sorted(feature_names)).encode()
    return hashlib.sha256(hash_input).hexdigest()[:16]


def validate_feature_compatibility(tenant_id: str, features: dict[str, float]) -> bool:
    """
    Validate live features against registered schema.

    Logs warnings for mismatches but does NOT block scoring.
    Non-blocking: returns True on any error (DB unavailable, etc.)
    to avoid blocking the scoring path.
    """
    try:
        registry = get_schema_registry()
        live_features = {k: v for k, v in features.items() if v is not None}

        # Compute live feature hash
        live_hash = _compute_feature_hash(list(live_features.keys()))

        # Get registered schema
        schema = registry.get_active_schema(tenant_id)

        if schema is None:
            # No schema registered yet — log and return permissive result
            logger.debug(
                "No schema registered for tenant={}, allowing request", tenant_id
            )
            return True

        # Compare
        live_feature_names = set(live_features.keys())
        registered_feature_names = set(schema.feature_names)

        missing = list(registered_feature_names - live_feature_names)
        extra = list(live_feature_names - registered_feature_names)

        has_mismatch = (
            (live_hash != schema.feature_hash) or bool(missing) or bool(extra)
        )

        if has_mismatch:
            logger.warning(
                "Feature schema mismatch for tenant={}: live_hash={} registered_hash={} "
                "missing={} extra={}",
                tenant_id,
                live_hash,
                schema.feature_hash,
                missing[:10],
                extra[:10],
            )

        return True  # Never block scoring on validation
    except Exception as exc:
        logger.debug("Schema validation skipped (DB unavailable): {}", exc)
        return True  # Fail open - don't block scoring


def auto_register_schema_if_missing(
    tenant_id: str, features: dict[str, float], created_by: str = "auto"
) -> None:
    """
    Auto-register schema if tenant has no registered schema.

    Useful during model training to capture the feature set.
    Non-blocking: logs error but doesn't raise.
    """
    try:
        registry = get_schema_registry()
        schema = registry.get_active_schema(tenant_id)

        if schema is None:
            logger.info("Auto-registering feature schema for tenant={}", tenant_id)
            registry.register_schema(
                tenant_id=tenant_id,
                feature_names=sorted(features.keys()),
                feature_types={k: type(v).__name__ for k, v in features.items()},
                created_by=created_by,
            )
            # Invalidate cache
            invalidate_schema_cache(tenant_id)
    except Exception as exc:
        logger.debug("Auto schema registration skipped: {}", exc)


def invalidate_schema_cache(tenant_id: str) -> None:
    """Invalidate cached schema for tenant."""
    from features.schema_registry import _schema_cache, _cache_time

    _schema_cache.pop(tenant_id, None)
    _cache_time.pop(tenant_id, None)
