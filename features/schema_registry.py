"""
FraudTrap — Feature Schema Registry
Manages feature schemas per tenant with versioning and validation.
"""

from __future__ import annotations
import hashlib
import json
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class FeatureSchema:
    """Represents a registered feature schema for a tenant."""

    tenant_id: str
    version: int
    feature_hash: str
    feature_names: list[str]
    feature_types: dict[str, str]
    created_at: datetime
    created_by: str
    is_active: bool = True


@dataclass
class ValidationResult:
    """Result of feature compatibility validation."""

    has_mismatch: bool
    live_hash: str
    registered_hash: Optional[str]
    missing_features: list[str]
    extra_features: list[str]
    type_mismatches: dict[str, tuple[str, str]]  # feature -> (expected, actual)


class FeatureSchemaRegistry:
    """
    Registry for feature schemas with versioning and validation.

    Stores feature schemas in PostgreSQL with:
    - Per-tenant versioned schemas
    - Feature hash for quick compatibility checks
    - Full feature names and types for validation
    """

    def __init__(self, dsn: str | None = None):
        self.dsn = dsn or settings.postgres_url
        self._cache: dict[str, FeatureSchema] = {}
        self._cache_ttl = 60  # seconds
        self._cache_time: dict[str, float] = {}

    @contextmanager
    def _conn(self):
        """Get a database connection."""
        conn = psycopg2.connect(self.dsn, cursor_factory=RealDictCursor)
        try:
            yield conn
        finally:
            conn.close()

    def _compute_feature_hash(self, feature_names: list[str]) -> str:
        """Compute SHA256 hash of ordered feature names."""
        hash_input = "|".join(sorted(feature_names)).encode()
        return hashlib.sha256(hash_input).hexdigest()[:16]

    def _compute_type_hash(self, feature_types: dict[str, str]) -> str:
        """Compute hash of feature types."""
        sorted_types = json.dumps(feature_types, sort_keys=True).encode()
        return hashlib.sha256(sorted_types).hexdigest()[:16]

    def register_schema(
        self,
        tenant_id: str,
        feature_names: list[str],
        feature_types: dict[str, str],
        created_by: str = "system",
    ) -> int:
        """
        Register a new feature schema for a tenant.

        Deactivates previous schema and inserts new one.
        Returns the new version number.
        """
        feature_hash = self._compute_feature_hash(feature_names)
        type_hash = self._compute_type_hash(feature_types)

        with self._conn() as conn:
            with conn.cursor() as cur:
                # Deactivate previous schema
                cur.execute(
                    """
                    UPDATE feature_schemas 
                    SET is_active = FALSE 
                    WHERE tenant_id = %s AND is_active = TRUE
                    """,
                    (tenant_id,),
                )

                # Insert new schema
                cur.execute(
                    """
                    INSERT INTO feature_schemas 
                        (tenant_id, feature_hash, type_hash, feature_names, feature_types, created_by)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING version
                    """,
                    (
                        tenant_id,
                        feature_hash,
                        type_hash,
                        json.dumps(feature_names),
                        json.dumps(feature_types),
                        created_by,
                    ),
                )
                version = cur.fetchone()["version"]
                conn.commit()

                # Invalidate cache
                self._invalidate_cache(tenant_id)

                logger.info(
                    "Registered feature schema v{} for tenant={}, features={}, hash={}",
                    version,
                    tenant_id,
                    len(feature_names),
                    feature_hash,
                )
                return version

    def get_active_schema(self, tenant_id: str) -> Optional[FeatureSchema]:
        """Get the currently active schema for a tenant (with caching)."""
        # Check cache
        now = time.time()
        if (
            tenant_id in self._cache
            and tenant_id in self._cache_time
            and now - self._cache_time[tenant_id] < self._cache_ttl
        ):
            return self._cache[tenant_id]

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT tenant_id, version, feature_hash, feature_names, 
                           feature_types, created_at, created_by, is_active
                    FROM feature_schemas
                    WHERE tenant_id = %s AND is_active = TRUE
                    ORDER BY version DESC
                    LIMIT 1
                    """,
                    (tenant_id,),
                )
                row = cur.fetchone()

                if row:
                    schema = FeatureSchema(
                        tenant_id=row["tenant_id"],
                        version=row["version"],
                        feature_hash=row["feature_hash"],
                        feature_names=row["feature_names"],
                        feature_types=row["feature_types"],
                        created_at=row["created_at"],
                        created_by=row["created_by"],
                        is_active=row["is_active"],
                    )
                    self._cache[tenant_id] = schema
                    self._cache_time[tenant_id] = now
                    return schema
                return None

    def get_schema_version(
        self, tenant_id: str, version: int
    ) -> Optional[FeatureSchema]:
        """Get a specific schema version."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT tenant_id, version, feature_hash, feature_names, 
                           feature_types, created_at, created_by, is_active
                    FROM feature_schemas
                    WHERE tenant_id = %s AND version = %s
                    """,
                    (tenant_id, version),
                )
                row = cur.fetchone()
                if row:
                    return FeatureSchema(
                        tenant_id=row["tenant_id"],
                        version=row["version"],
                        feature_hash=row["feature_hash"],
                        feature_names=row["feature_names"],
                        feature_types=row["feature_types"],
                        created_at=row["created_at"],
                        created_by=row["created_by"],
                        is_active=row["is_active"],
                    )
                return None

    def validate_compatibility(
        self, tenant_id: str, live_features: dict[str, float]
    ) -> ValidationResult:
        """
        Validate live features against registered schema.

        Checks:
        - Feature hash match (order + names)
        - Missing features (in schema but not in live)
        - Extra features (in live but not in schema)
        - Type mismatches (if type info available)
        """
        schema = self.get_active_schema(tenant_id)

        if schema is None:
            # No schema registered - permissive mode
            return ValidationResult(
                has_mismatch=False,
                live_hash=self._compute_feature_hash(list(live_features.keys())),
                registered_hash=None,
                missing_features=[],
                extra_features=[],
                type_mismatches={},
            )

        live_names = sorted(live_features.keys())
        live_hash = self._compute_feature_hash(live_names)

        registered_names = set(schema.feature_names)
        live_names_set = set(live_names)

        missing = sorted(registered_names - live_names_set)
        extra = sorted(live_names_set - registered_names)

        # Type validation (if live features have type info)
        type_mismatches = {}
        # Could be extended if live features carry type info

        has_mismatch = (
            live_hash != schema.feature_hash or missing or extra or type_mismatches
        )

        if has_mismatch:
            logger.warning(
                "Feature schema mismatch for tenant={}: "
                "live_hash={} registered_hash={} missing={} extra={}",
                tenant_id,
                live_hash,
                schema.feature_hash,
                len(missing),
                len(extra),
            )

        return ValidationResult(
            has_mismatch=has_mismatch,
            live_hash=live_hash,
            registered_hash=schema.feature_hash,
            missing_features=missing,
            extra_features=extra,
            type_mismatches=type_mismatches,
        )

    def _invalidate_cache(self, tenant_id: str) -> None:
        """Invalidate cached schema for tenant."""
        self._cache.pop(tenant_id, None)
        self._cache_time.pop(tenant_id, None)

    def list_schemas(self, tenant_id: str | None = None) -> list[FeatureSchema]:
        """List all schemas, optionally filtered by tenant."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                if tenant_id:
                    cur.execute(
                        """
                        SELECT tenant_id, version, feature_hash, feature_names, 
                               feature_types, created_at, created_by, is_active
                        FROM feature_schemas
                        WHERE tenant_id = %s
                        ORDER BY version DESC
                        """,
                        (tenant_id,),
                    )
                else:
                    cur.execute("""
                        SELECT tenant_id, version, feature_hash, feature_names, 
                               feature_types, created_at, created_by, is_active
                        FROM feature_schemas
                        ORDER BY tenant_id, version DESC
                        """)
                return [
                    FeatureSchema(
                        tenant_id=r["tenant_id"],
                        version=r["version"],
                        feature_hash=r["feature_hash"],
                        feature_names=r["feature_names"],
                        feature_types=r["feature_types"],
                        created_at=r["created_at"],
                        created_by=r["created_by"],
                        is_active=r["is_active"],
                    )
                    for r in cur.fetchall()
                ]


# Global registry instance
_schema_registry: Optional[FeatureSchemaRegistry] = None


def get_schema_registry() -> FeatureSchemaRegistry:
    global _schema_registry
    if _schema_registry is None:
        _schema_registry = FeatureSchemaRegistry()
    return _schema_registry


# Auto-registration helper
def register_schema_from_features(
    tenant_id: str, features: dict[str, float], created_by: str = "auto"
) -> int:
    """
    Automatically infer schema from feature dict and register.

    Useful during model training to capture the exact feature set.
    """
    registry = get_schema_registry()

    # Infer types from sample values
    feature_types = {}
    for name, value in features.items():
        if isinstance(value, bool):
            feature_types[name] = "boolean"
        elif isinstance(value, int):
            feature_types[name] = "integer"
        elif isinstance(value, float):
            feature_types[name] = "float"
        else:
            feature_types[name] = "unknown"

    return registry.register_schema(
        tenant_id=tenant_id,
        feature_names=sorted(features.keys()),
        feature_types=feature_types,
        created_by=created_by,
    )
