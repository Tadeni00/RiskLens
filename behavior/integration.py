"""
FraudTrap Behavioral Intelligence Layer - Integration with Existing Components
"""

from __future__ import annotations
from typing import Optional, Dict, Any
import logging

from behavior.services.behavior_engine import BehaviorEngine, BehaviorEngineConfig
from behavior.storage.redis_store import get_feature_store, FeatureStoreClient
from behavior.profiles.customer import CustomerBehaviorProfile
from behavior.profiles.merchant import MerchantBehaviorProfile
from behavior.profiles.device import DeviceBehaviorProfile
from behavior.profiles.beneficiary import BeneficiaryBehaviorProfile
from behavior.profiles.payment_instrument import PaymentInstrumentProfile
from behavior.profiles.device import DeviceBehaviorProfile
from behavior.profiles.beneficiary import BeneficiaryBehaviorProfile
from behavior.profiles.payment_instrument import PaymentInstrumentProfile

from behavior.feature_generation import generate_behavioral_features
from behavior.storage.redis_store import get_feature_store, FeatureStoreClient

logger = logging.getLogger(__name__)


class BehavioralIntelligenceIntegration:
    """
    Integration layer between Behavioral Intelligence Layer and existing FraudTrap components.

    This module handles the integration points between the Behavioral Intelligence Layer
    and the existing FraudTrap components:
    - Cold Start (Phase 1)
    - Semi-Supervised (Phase 2)
    - Supervised (Phase 3)
    - Rules Engine
    - Scoring Orchestrator
    """

    def __init__(self):
        self._behavior_engine = None
        self._initialized = False

    def initialize(self, feature_store=None):
        """Initialize the behavioral intelligence integration."""
        from behavior.services.behavior_engine import (
            BehaviorEngine,
            BehaviorEngineConfig,
        )
        from behavior.storage.redis_store import get_feature_store

        if self._initialized:
            return

        feature_store = get_feature_store()

        config = BehaviorEngineConfig(
            feature_store=get_feature_store(),
            enable_behavioral_features=True,
            enable_velocity_features=True,
            enable_trust_scoring=True,
            enable_novelty_detection=True,
            enable_similarity_features=True,
        )

        from behavior.services.behavior_engine import (
            BehaviorEngine,
            BehaviorEngineConfig,
        )

        self._behavior_engine = BehaviorEngine(
            BehaviorEngineConfig(
                feature_store=None,  # Will use get_feature_store()
                enable_behavioral_features=True,
                enable_velocity_features=True,
                enable_trust_scoring=True,
                enable_novelty_detection=True,
                enable_similarity_features=True,
            )
        )
        self._initialized = True

    def enhance_cold_start_features(self, transaction, features: dict) -> dict:
        """
        Enhance Phase 1 (Cold Start) features with behavioral features.

        The VAE, Isolation Forest, and Tail Detector now receive:
        - Original engineered features
        + Behavioral features (velocity, trust scores, similarity, novelty)
        """
        from behavior.feature_generation.generator import generate_behavioral_features
        from behavior.storage.redis_store import get_feature_store

        feature_store = get_feature_store()
        if feature_store is None:
            return {}

        try:
            # This would need the transaction object and profiles
            # For now, return empty dict as placeholder
            return {}
        except Exception as e:
            logger.warning("Cold-start behavioral feature enhancement failed: %s", e)
            return {}

    def enhance_adaptive_learning_features(self, transaction, features: dict) -> dict:
        """
        Enhance Layer 2 (Adaptive Learning) features with behavioral features.

        Uses pseudo-labels from Cold Start + behavioral features to train
        the TabPFN adaptive learner.
        """
        # Generate behavioral features
        behavioral_features = self._generate_behavioral_features_for_transaction({})

        # Merge with existing features
        return {**features, **behavioral_features}

    def enhance_supervised_features(self, transaction, features: dict) -> dict:
        """
        Enhance Phase 3 (Supervised) features with behavioral features.

        The supervised model learns interactions like:
        - High amount + New device + High velocity + New beneficiary = Very high fraud probability
        """
        # Generate behavioral features
        behavioral_features = {}

        # Merge with supervised features
        return {**features, **behavioral_features}

    def _generate_behavioral_features(self, transaction):
        """Generate behavioral features for a transaction."""
        # Placeholder - actual implementation would use feature store
        return {}

    def get_behavioral_features_for_scoring(self, transaction, feature_store):
        """
        Generate behavioral features for a transaction during scoring.

        This is called from the scoring orchestrator to enhance features
        with behavioral intelligence before model scoring.
        """
        try:
            from behavior.feature_generation.generator import (
                generate_behavioral_features,
            )
            from behavior.storage.redis_store import get_feature_store

            feature_store = get_feature_store()
            return generate_behavioral_features(
                transaction=transaction,
                customer_profile=None,  # Would be fetched from feature store
                merchant_profile=None,
                device_profile=None,
                beneficiary_profile=None,
                instrument_profile=None,
            )
        except Exception as e:
            logger.warning("Behavioral feature generation failed: %s", e)
            return {}

    # Backwards-compatible alias
    enhance_semi_supervised_features = enhance_adaptive_learning_features


# Global instance
_behavioral_integration = None


def get_behavioral_integration() -> "BehavioralIntelligenceIntegration":
    """Get or create the global behavioral intelligence integration instance."""
    global _behavioral_integration
    if _behavioral_integration is None:
        _behavioral_integration = BehavioralIntelligenceIntegration()
    return _behavioral_integration


_behavioral_integration = None


def get_behavioral_integration() -> "BehavioralIntelligenceIntegration":
    """Get or create the global behavioral intelligence integration instance."""
    global _behavioral_integration
    if _behavioral_integration is None:
        _behavioral_integration = BehavioralIntelligenceIntegration()
    return _behavioral_integration
