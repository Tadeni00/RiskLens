"""
RiskLens — Model Router
Routes tenants to appropriate models based on phase, availability, and experiments.
Single Responsibility: Model selection and routing logic.
"""

from __future__ import annotations
import hashlib
import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np

from loguru import logger

from models.cold_start.ensemble import ColdStartEnsemble
from models.adaptive_learning.tabpfn_learner import TabPFNAdaptiveLearner
from models.supervised.champion import ChampionModel
from scoring.simple_model import SimpleFraudModel


@dataclass
class RoutingDecision:
    """Result of model routing."""

    model: object
    model_type: str  # "simple", "cold_start", "adaptive_learning", "supervised", "gnn"
    model_version: str
    experiment_variant: str = "champion"
    tenant_phase: str = "UNSUPERVISED"


class ModelRouter:
    """
    Routes tenants to appropriate models based on phase, availability, and experiments.

    Single Responsibility: Model selection and traffic routing.
    """

    def __init__(
        self,
        registry_models: dict,
        experiment_config: Optional[dict] = None,
    ):
        """
        Initialize router.

        Args:
            registry_models: Dict from ModelLoader with all loaded models
            experiment_config: Optional A/B test configuration
        """
        self.simple_models = registry_models.get("simple_models", {})
        self.cold_start_models = registry_models.get("cold_start_models", {})
        self.adaptive_learner_models = registry_models.get("adaptive_learner_models", {})
        self.champion_models = registry_models.get("champion_models", {})
        self.shared_cold_start = registry_models.get("cold_start")
        self.shared_adaptive_learner = registry_models.get("adaptive_learner")
        self.shared_champion = registry_models.get("champion")

        self.active_phase = registry_models.get("active_phase", "UNSUPERVISED")
        self.model_version = registry_models.get("model_version", "unloaded")
        self.feature_names = registry_models.get("feature_names", [])

        self.experiment_config = experiment_config or {}
        self._experiments = self.experiment_config.get("experiments", [])

        self.logger = logger.bind(component="ModelRouter")

    def get_model(self, tenant_id: str, transaction_id: Optional[str] = None) -> RoutingDecision:
        """
        Get the appropriate model for a tenant.

        Priority:
        1. Tenant-specific simple model (production)
        2. Tenant-specific champion model (CatBoost + specialist)
        3. Tenant-specific adaptive learner (TabPFN)
        4. Tenant-specific cold-start
        5. Shared champion
        6. Shared adaptive learner
        7. Shared cold-start
        8. Heuristic fallback
        """
        # 1. Tenant simple model (highest priority for production)
        if tenant_id in self.simple_models:
            return RoutingDecision(
                model=self.simple_models[tenant_id],
                model_type="simple",
                model_version=self.simple_models[tenant_id].model_version,
                tenant_phase="SUPERVISED",
            )

        # 2. Tenant champion model
        if tenant_id in self.champion_models:
            variant = self._check_experiment(tenant_id, "supervised")
            return RoutingDecision(
                model=self.champion_models[tenant_id],
                model_type="supervised",
                model_version=self.champion_models[tenant_id].model_version,
                experiment_variant=variant,
                tenant_phase="SUPERVISED",
            )

        # 3. Tenant adaptive learner (TabPFN)
        if tenant_id in self.adaptive_learner_models:
            variant = self._check_experiment(tenant_id, "adaptive_learning")
            return RoutingDecision(
                model=self.adaptive_learner_models[tenant_id],
                model_type="adaptive_learning",
                model_version=self.adaptive_learner_models[tenant_id].model_version,
                experiment_variant=variant,
                tenant_phase="ADAPTIVE_LEARNING",
            )

        # 4. Tenant cold-start
        if tenant_id in self.cold_start_models:
            return RoutingDecision(
                model=self.cold_start_models[tenant_id],
                model_type="cold_start",
                model_version=self.cold_start_models[tenant_id].model_version,
                tenant_phase="UNSUPERVISED",
            )

        # 5. Shared champion
        if self.shared_champion:
            return RoutingDecision(
                model=self.shared_champion,
                model_type="supervised",
                model_version=self.model_version,
                tenant_phase="SUPERVISED",
            )

        # 6. Shared adaptive learner
        if self.shared_adaptive_learner:
            return RoutingDecision(
                model=self.shared_adaptive_learner,
                model_type="adaptive_learning",
                model_version=self.model_version,
                tenant_phase="ADAPTIVE_LEARNING",
            )

        # 7. Shared cold-start
        if self.shared_cold_start:
            return RoutingDecision(
                model=self.shared_cold_start,
                model_type="cold_start",
                model_version=self.model_version,
                tenant_phase="UNSUPERVISED",
            )

        # 8. Fallback - no model available
        self.logger.warning("No model available for tenant={}, using heuristic", tenant_id)
        return RoutingDecision(
            model=None,
            model_type="heuristic",
            model_version="heuristic",
            tenant_phase="UNSUPERVISED",
        )

    def _check_experiment(self, tenant_id: str, model_type: str) -> str:
        """Check if tenant should be routed to experiment variant."""
        if not self._experiments:
            return "champion"

        for exp in self._experiments:
            if exp.get("tenant") == tenant_id and exp.get("model_type") == model_type:
                if not exp.get("active", False):
                    continue

                # Consistent hashing for stable assignment
                hash_input = f"{transaction_id or tenant_id}:{exp['name']}"
                hash_val = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16)
                pct = (hash_val % 100) + 1

                challenger_pct = exp.get("challenger_traffic_pct", 10)
                if pct <= challenger_pct:
                    return "challenger"

        return "champion"

    def get_available_tenants(self) -> dict[str, list[str]]:
        """Get list of tenants with loaded models by type."""
        return {
            "simple": sorted(self.simple_models.keys()),
            "champion": sorted(self.champion_models.keys()),
            "adaptive_learning": sorted(self.adaptive_learner_models.keys()),
            "cold_start": sorted(self.cold_start_models.keys()),
        }

    def get_tenant_phase(self, tenant_id: str) -> str:
        """Get current phase for tenant."""
        if tenant_id in self.simple_models or tenant_id in self.champion_models:
            return "SUPERVISED"
        if tenant_id in self.adaptive_learner_models:
            return "ADAPTIVE_LEARNING"
        if tenant_id in self.cold_start_models:
            return "UNSUPERVISED"
        return self.active_phase

    def is_experiment_active(self, experiment_name: str) -> bool:
        """Check if an experiment is active."""
        for exp in self._experiments:
            if exp.get("name") == experiment_name:
                return exp.get("active", False)
        return False

    def record_experiment_exposure(
        self, tenant_id: str, experiment_name: str, variant: str
    ) -> None:
        """Record experiment exposure for analytics."""
        self.logger.info(
            "Experiment exposure: tenant={} exp={} variant={}",
            tenant_id,
            experiment_name,
            variant,
        )


def create_router(registry_models: dict, experiment_config: Optional[dict] = None) -> ModelRouter:
    """Factory function to create ModelRouter."""
    return ModelRouter(registry_models, experiment_config)
