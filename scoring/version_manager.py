"""
RiskLens — Version Manager
Manages model versioning, feature hashes, and artifact metadata.
Single Responsibility: Version tracking and validation.
"""

from __future__ import annotations
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from threading import RLock

from loguru import logger


@dataclass
class ModelVersionInfo:
    """Complete version information for a model."""

    model_version: str = "unloaded"
    training_hash: Optional[str] = None
    feature_hash: Optional[str] = None
    dataset_hash: Optional[str] = None
    trained_at: Optional[str] = None
    active_phase: str = "UNSUPERVISED"
    feature_names: list[str] = field(default_factory=list)
    tenant_id: Optional[str] = None
    model_type: str = "unknown"
    load_time_ms: float = 0.0
    load_status: str = "unloaded"  # "loaded", "failed", "unloaded"


class VersionManager:
    """
    Manages model versions, hashes, and metadata.

    Single Responsibility: Version tracking, validation, and compatibility checks.
    """

    def __init__(self):
        self._versions: dict[str, ModelVersionInfo] = {}
        self._lock = RLock()
        self._last_reload = 0.0

    def register_version(
        self,
        tenant_id: str,
        model_type: str,
        version_info: ModelVersionInfo,
    ) -> None:
        """Register a model version."""
        key = f"{tenant_id}:{model_type}"
        with self._lock:
            self._versions[key] = version_info
            logger.debug("Registered version: {} v{}", key, version_info.model_version)

    def register_simple_model(self, tenant_id: str, model) -> None:
        """Register a simple model version."""
        info = ModelVersionInfo(
            tenant_id=tenant_id,
            model_version=model.model_version,
            feature_names=model.feature_names,
            load_status="loaded",
        )
        self.register_version(tenant_id, "simple", info)

    def register_cold_start(self, tenant_id: str, model) -> None:
        """Register a cold-start model version."""
        info = ModelVersionInfo(
            tenant_id=tenant_id,
            model_version=model.model_version,
            training_hash=model.training_hash,
            feature_hash=model.feature_hash,
            dataset_hash=model.dataset_hash,
            trained_at=model.trained_at,
            feature_names=model.feature_names,
            active_phase="UNSUPERVISED",
            load_status="loaded",
        )
        self.register_version(tenant_id, "cold_start", info)

    def register_adaptive_learner(self, tenant_id: str, model) -> None:
        """Register an adaptive learner (TabPFN) model version."""
        info = ModelVersionInfo(
            tenant_id=tenant_id,
            model_version=(model.model_version if hasattr(model, "model_version") else "unknown"),
            training_hash=getattr(model, "training_hash", None),
            feature_hash=getattr(model, "feature_hash", None),
            feature_names=getattr(model, "feature_names", []),
            active_phase="ADAPTIVE_LEARNING",
            load_status="loaded",
        )
        self.register_version(tenant_id, "adaptive_learning", info)

    # Backwards-compatible alias
    register_semi_supervised = register_adaptive_learner

    def register_supervised(self, tenant_id: str, model) -> None:
        """Register a supervised model version."""
        info = ModelVersionInfo(
            tenant_id=tenant_id,
            model_version=model.model_version,
            training_hash=model.training_hash,
            feature_hash=model.feature_hash,
            dataset_hash=model.dataset_hash,
            trained_at=model.trained_at,
            feature_names=model.feature_names,
            active_phase="SUPERVISED",
            load_status="loaded",
        )
        self.register_version(tenant_id, "supervised", info)

    def register_shared_model(self, model_type: str, model) -> None:
        """Register a shared (cross-tenant) model."""
        if model_type == "cold_start" and model:
            info = ModelVersionInfo(
                model_version=model.model_version,
                training_hash=model.training_hash,
                feature_hash=model.feature_hash,
                dataset_hash=model.dataset_hash,
                trained_at=model.trained_at,
                feature_names=model.feature_names,
                active_phase="UNSUPERVISED",
                load_status="loaded",
            )
            self.register_version("_shared", "cold_start", info)

        elif model_type in ("semi_supervised", "adaptive_learning") and model:
            info = ModelVersionInfo(
                model_version=(
                    model.model_version if hasattr(model, "model_version") else "unknown"
                ),
                feature_hash=getattr(model, "feature_hash", None),
                feature_names=getattr(model, "feature_names", []),
                active_phase="ADAPTIVE_LEARNING",
                load_status="loaded",
            )
            self.register_version("_shared", "adaptive_learning", info)

        elif model_type == "supervised" and model:
            info = ModelVersionInfo(
                model_version=model.model_version,
                training_hash=model.training_hash,
                feature_hash=model.feature_hash,
                dataset_hash=model.dataset_hash,
                trained_at=model.trained_at,
                feature_names=model.feature_names,
                active_phase="SUPERVISED",
                load_status="loaded",
            )
            self.register_version("_shared", "supervised", info)

        elif model_type == "gnn" and model:
            info = ModelVersionInfo(
                model_version=(
                    model.model_version if hasattr(model, "model_version") else "unknown"
                ),
                load_status="loaded",
            )
            self.register_version("_shared", "gnn", info)

    def get_version(self, tenant_id: str, model_type: str) -> Optional[ModelVersionInfo]:
        """Get version info for a tenant/model combination."""
        key = f"{tenant_id}:{model_type}"
        with self._lock:
            return self._versions.get(key)

    def get_tenant_versions(self, tenant_id: str) -> dict[str, ModelVersionInfo]:
        """Get all versions for a tenant."""
        with self._lock:
            return {
                k.split(":")[1]: v
                for k, v in self._versions.items()
                if k.startswith(f"{tenant_id}:")
            }

    def get_all_versions(self) -> dict[str, dict[str, ModelVersionInfo]]:
        """Get all versions grouped by tenant."""
        with self._lock:
            result = {}
            for key, info in self._versions.items():
                tenant, mtype = key.split(":", 1)
                if tenant not in result:
                    result[tenant] = {}
                result[tenant][mtype] = info
            return result

    def get_active_phase(self, tenant_id: str) -> str:
        """Determine active phase for a tenant based on loaded models."""
        versions = self.get_tenant_versions(tenant_id)

        if "simple" in versions:
            return "SUPERVISED"
        if "supervised" in versions:
            return "SUPERVISED"
        if "adaptive_learning" in versions:
            return "ADAPTIVE_LEARNING"
        if "cold_start" in versions:
            return "UNSUPERVISED"

        # Check shared
        shared = self.get_tenant_versions("_shared")
        if "supervised" in shared:
            return "SUPERVISED"
        if "adaptive_learning" in shared:
            return "ADAPTIVE_LEARNING"
        if "cold_start" in shared:
            return "UNSUPERVISED"

        return "UNSUPERVISED"

    def validate_feature_compatibility(
        self,
        tenant_id: str,
        model_type: str,
        live_feature_hash: str,
    ) -> tuple[bool, str]:
        """
        Validate feature hash compatibility.

        Returns:
            (is_compatible, message)
        """
        version = self.get_version(tenant_id, model_type)

        if version is None:
            return True, "No registered version, skipping validation"

        if version.feature_hash is None:
            return True, "No feature hash registered, skipping validation"

        if live_feature_hash != version.feature_hash:
            msg = (
                f"Feature hash mismatch for {tenant_id}:{model_type}. "
                f"Live: {live_feature_hash}, Registered: {version.feature_hash}. "
                f"Model may produce unreliable scores."
            )
            return False, msg

        return True, "Feature hash matches"

    def get_feature_hash(self, feature_names: list[str]) -> str:
        """Compute feature hash from feature names."""
        hash_input = "|".join(sorted(feature_names)).encode()
        return hashlib.sha256(hash_input).hexdigest()[:16]

    def get_model_hash(self, model) -> str:
        """Get hash of model hyperparameters."""
        import pickle

        config = {}
        for attr in [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "reg_alpha",
            "reg_lambda",
        ]:
            if hasattr(model, attr):
                config[attr] = getattr(model, attr)
        return hashlib.sha256(pickle.dumps(config, protocol=4)).hexdigest()[:16]

    def get_dataset_hash(self, X: np.ndarray, y: np.ndarray) -> str:
        """Compute statistical fingerprint of dataset."""
        if len(X) > 10000:
            idx = np.random.choice(len(X), 10000, replace=False)
            X, y = X[idx], y[idx]

        stats = np.concatenate(
            [
                X.mean(axis=0),
                X.std(axis=0),
                [y.mean(), len(y)],
            ]
        )
        return hashlib.sha256(stats.tobytes()).hexdigest()[:16]

    def get_version_summary(self) -> dict:
        """Get summary of all registered versions."""
        with self._lock:
            summary = {}
            for key, info in self._versions.items():
                tenant, mtype = key.split(":", 1)
                if tenant not in summary:
                    summary[tenant] = {}
                summary[tenant][mtype] = {
                    "version": info.model_version,
                    "phase": info.active_phase,
                    "status": info.load_status,
                    "feature_count": len(info.feature_names),
                    "trained_at": info.trained_at,
                }
            return summary

    def clear_tenant(self, tenant_id: str) -> None:
        """Clear all versions for a tenant."""
        with self._lock:
            keys_to_remove = [k for k in self._versions if k.startswith(f"{tenant_id}:")]
            for k in keys_to_remove:
                del self._versions[k]

    def clear_all(self) -> None:
        """Clear all versions."""
        with self._lock:
            self._versions.clear()


# Global instance
_version_manager: Optional[VersionManager] = None


def get_version_manager() -> VersionManager:
    global _version_manager
    if _version_manager is None:
        _version_manager = VersionManager()
    return _version_manager
