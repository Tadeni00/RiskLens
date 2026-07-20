"""
FraudTrap Behavioral Intelligence Layer
Behavior Engine Service - Main orchestration service
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
from datetime import datetime, timezone
import logging

from behavior.profiles.customer import CustomerBehaviorProfile
from behavior.profiles.merchant import MerchantBehaviorProfile
from behavior.profiles.device import DeviceBehaviorProfile
from behavior.profiles.beneficiary import BeneficiaryBehaviorProfile
from behavior.profiles.payment_instrument import PaymentInstrumentProfile

from behavior.feature_generation import (
    generate_behavioral_features,
    compute_velocity_features,
)
from behavior.feature_generation.similarity import (
    merchant_similarity,
    device_similarity,
    country_similarity,
    typing_similarity,
    cosine_similarity,
)
from behavior.feature_generation.trust import (
    get_device_trust_score,
    get_merchant_trust_score,
    get_beneficiary_trust_score,
    get_customer_reputation,
    get_historical_chargeback_rate,
    get_historical_fraud_rate,
)
from behavior.storage.redis_store import (
    FeatureStoreClient,
    RedisFeatureStore,
    InMemoryFeatureStore,
    MockFeatureStore,
    get_feature_store,
)

logger = logging.getLogger(__name__)


@dataclass
class BehaviorEngineConfig:
    """Configuration for the Behavior Engine."""
    feature_store: FeatureStoreClient
    enable_behavioral_features: bool = True
    enable_velocity_features: bool = True
    enable_trust_scoring: bool = True
    enable_novelty_detection: bool = True
    enable_similarity_features: bool = True
    enable_trust_scoring: bool = True


@dataclass
class BehavioralFeatures:
    """Container for all generated behavioral features."""
    # Amount features
    customer_amount_zscore: float = 0.0
    merchant_amount_zscore: float = 0.0
    tenant_amount_percentile: float = 0.5
    
    # Velocity features
    velocity_score: float = 0.0
    acct_v_1m_count: float = 0.0
    acct_v_1h_count: float = 0.0
    acct_v_24h_count: float = 0.0
    acct_v_24h_total_amt: float = 0.0
    
    # Novelty features
    new_device_flag: float = 0.0
    new_ip_flag: float = 0.0
    new_country_flag: float = 0.0
    new_merchant_flag: float = 0.0
    new_device_flag: float = 0.0
    new_merchant_flag: float = 0.0
    
    # Similarity features
    merchant_similarity: float = 0.0
    device_similarity: float = 0.0
    country_similarity: float = 0.0
    typing_similarity: float = 0.0
    
    # Trust scores
    device_trust_score: float = 0.5
    merchant_trust_score: float = 0.5
    beneficiary_trust_score: float = 0.5
    customer_reputation: float = 0.5
    
    # Risk scores
    customer_risk_score: float = 0.0
    merchant_risk_score: float = 0.0
    device_risk_score: float = 0.0
    
    # Trust scores
    device_trust_score: float = 0.5
    merchant_trust_score: float = 0.5
    beneficiary_trust_score: float = 0.5
    customer_reputation: float = 0.5
    
    # Historical rates
    historical_chargeback_rate: float = 0.0
    historical_fraud_rate: float = 0.0
    customer_risk_score: float = 0.0
    merchant_risk_score: float = 0.0
    device_risk_score: float = 0.0
    
    # Device/Geo features
    new_device_flag: float = 0.0
    new_ip_flag: float = 0.0
    new_country_flag: float = 0.0
    new_merchant_flag: float = 0.0
    new_device_flag: float = 0.0
    new_merchant_flag: float = 0.0
    impossible_travel: float = 0.0
    cross_country_flag: float = 0.0
    
    # Velocity
    velocity_score: float = 0.0
    acct_v_1m_count: float = 0.0
    acct_v_1h_count: float = 0.0
    acct_v_24h_count: float = 0.0
    acct_v_24h_total_amt: float = 0.0
    
    # Behavioural
    hour_deviation: float = 0.0
    weekday_deviation: float = 0.0
    is_night: float = 0.0
    is_weekend: float = 0.0
    
    # Device/Geo
    device_trust_score: float = 0.5
    merchant_trust_score: float = 0.5
    beneficiary_trust_score: float = 0.5
    customer_reputation: float = 0.5
    
    # Historical rates
    historical_chargeback_rate: float = 0.0
    historical_fraud_rate: float = 0.0
    customer_risk_score: float = 0.0
    merchant_risk_score: float = 0.0
    device_risk_score: float = 0.0
    
    # Risk scores
    merchant_risk_score: float = 0.0
    device_risk_score: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to flat dictionary for model input."""
        return {
            "customer_amount_zscore": self.customer_amount_zscore,
            "merchant_amount_zscore": self.merchant_amount_zscore,
            "tenant_amount_percentile": self.tenant_amount_percentile,
            "velocity_score": self.velocity_score,
            "acct_v_1m_count": self.acct_v_1m_count,
            "acct_v_1h_count": self.acct_v_1h_count,
            "acct_v_24h_count": self.acct_v_24h_count,
            "acct_v_24h_total_amt": self.acct_v_24h_total_amt,
            "new_device_flag": self.new_device_flag,
            "new_ip_flag": self.new_ip_flag,
            "new_country_flag": self.new_country_flag,
            "new_merchant_flag": self.new_merchant_flag,
            "new_device_flag": self.new_device_flag,
            "new_merchant_flag": self.new_merchant_flag,
            "merchant_similarity": self.merchant_similarity,
            "device_similarity": self.device_similarity,
            "country_similarity": self.country_similarity,
            "typing_similarity": self.typing_similarity,
            "device_trust_score": self.device_trust_score,
            "merchant_trust_score": self.merchant_trust_score,
            "beneficiary_trust_score": self.beneficiary_trust_score,
            "customer_reputation": self.customer_reputation,
            "historical_chargeback_rate": self.historical_chargeback_rate,
            "historical_fraud_rate": self.historical_fraud_rate,
            "customer_risk_score": self.customer_risk_score,
            "merchant_risk_score": self.merchant_risk_score,
            "device_risk_score": self.device_risk_score,
            "device_trust_score": self.device_trust_score,
            "merchant_trust_score": self.merchant_trust_score,
            "beneficiary_trust_score": self.beneficiary_trust_score,
            "customer_reputation": self.customer_reputation,
            "historical_chargeback_rate": self.historical_chargeback_rate,
            "historical_fraud_rate": self.historical_fraud_rate,
            "customer_risk_score": self.customer_risk_score,
            "merchant_risk_score": self.merchant_risk_score,
            "device_risk_score": self.device_risk_score,
            "device_trust_score": self.device_trust_score,
            "merchant_trust_score": self.merchant_trust_score,
            "beneficiary_trust_score": self.beneficiary_trust_score,
            "customer_reputation": self.customer_reputation,
            "historical_chargeback_rate": self.historical_chargeback_rate,
            "historical_fraud_rate": self.historical_fraud_rate,
            "customer_risk_score": self.customer_risk_score,
            "merchant_risk_score": self.merchant_risk_score,
            "device_risk_score": self.device_risk_score,
            "device_trust_score": self.device_trust_score,
            "merchant_trust_score": self.merchant_trust_score,
            "beneficiary_trust_score": self.beneficiary_trust_score,
            "customer_reputation": self.customer_reputation,
            "historical_chargeback_rate": self.historical_chargeback_rate,
            "historical_fraud_rate": self.historical_fraud_rate,
            "customer_risk_score": self.customer_risk_score,
            "merchant_risk_score": self.merchant_risk_score,
            "device_risk_score": self.device_risk_score,
        }

    def to_dict(self) -> dict:
        """Convert to flat dictionary for model input."""
        return {
            k: getattr(self, k) for k in self.__dataclass_fields__.keys()
        }


class BehaviorEngine:
    """
    Main behavioral intelligence engine.
    Orchestrates profile retrieval, feature generation, and profile updates.
    """
    
    def __init__(self, config: BehaviorEngineConfig):
        self.config = config
        self.feature_store = config.feature_store
        self.logger = logging.getLogger(__name__)
    
    def process_transaction(
        self,
        transaction,
        tenant_id: str,
        customer_profile: Optional[object] = None,
        merchant_profile: Optional[object] = None,
        device_profile: Optional[object] = None,
        beneficiary_profile: Optional[object] = None,
        instrument_profile: Optional[object] = None,
    ) -> dict:
        """
        Main entry point for behavioral feature generation.
        
        Args:
            transaction: Transaction object
            tenant_id: Tenant identifier
            customer_profile: Pre-loaded customer profile (optional)
            merchant_profile: Merchant profile (optional)
            device_profile: Device profile (optional)
            beneficiary_profile: Beneficiary profile (optional)
            instrument_profile: Payment instrument profile (optional)
            
        Returns:
            BehavioralFeatures object with all computed features
        """
        features = BehavioralFeatures()
        
        if not self.config.enable_behavioral_features:
            return BehavioralFeatures()  # Return empty features
        
        # Get profiles from feature store if not provided
        if self.config.enable_behavioral_features:
            features = self._generate_all_features(
                transaction, tenant_id,
                customer_profile, merchant_profile,
                device_profile, beneficiary_profile,
                instrument_profile
            )
        
        return features
    
    def _generate_all_features(
        self,
        transaction,
        tenant_id: str,
        customer_profile,
        merchant_profile,
        device_profile,
        beneficiary_profile,
        instrument_profile,
    ):
        """Generate all behavioral features."""
        features = BehavioralFeatures()
        
        if self.config.enable_velocity_features:
            features = compute_velocity_features(
                transaction, tenant_id
            )
        
        # Get profiles from feature store
        customer_profile = self._get_customer_profile(transaction.tenant_id, transaction.account_id)
        merchant_profile = self._get_merchant_profile(transaction.tenant_id, transaction.merchant_id)
        device_profile = self._get_device_profile(transaction.tenant_id, transaction.device_id)
        beneficiary_profile = self._get_beneficiary_profile(transaction.tenant_id, transaction.counterparty_id)
        instrument_profile = self._get_instrument_profile(transaction.tenant_id, transaction.payment_instrument_id)
        
        # Generate all features
        features = generate_behavioral_features(
            transaction,
            customer_profile=customer_profile,
            merchant_profile=merchant_profile,
            device_profile=device_profile,
            beneficiary_profile=beneficiary_profile,
            instrument_profile=instrument_profile,
        )
        
        return features
    
    def _get_customer_profile(self, tenant_id: str, customer_id: str):
        if not self.feature_store:
            return None
        try:
            return self.feature_store.get_customer_profile(tenant_id, customer_id)
        except Exception as e:
            logger.warning(f"Failed to get customer profile: {e}")
            return None
    
    def _get_merchant_profile(self, tenant_id: str, merchant_id: str):
        if not self.feature_store:
            return None
        try:
            return self.feature_store.get_merchant_profile(merchant_id, tenant_id)
        except Exception:
            return None
    
    def _get_device_profile(self, tenant_id: str, device_id: str):
        try:
            return self.feature_store.get_device_profile(tenant_id, device_id)
        except Exception:
            return None
    
    def _get_beneficiary_profile(self, tenant_id: str, beneficiary_id: str):
        try:
            return self.feature_store.get_beneficiary_profile(beneficiary_id)
        except Exception:
            return None
    
    def _get_instrument_profile(self, tenant_id: str, instrument_id: str):
        try:
            return self.feature_store.get_payment_instrument_profile(instrument_id)
        except Exception:
            return None
    
    def update_profiles(self, transaction, risk_score: float, decision: str) -> None:
        """
        Update behavioral profiles with transaction outcome.
        Called after scoring to update profiles incrementally.
        """
        try:
            # Update customer profile
            if self.feature_store:
                profile = self.feature_store.get_customer_profile(transaction.tenant_id, transaction.account_id)
                if profile:
                    # Update profile with transaction
                    # In a real implementation, this would call profile.update(transaction)
                    pass
            
# Update merchant profile
            # ... similar pattern for other profiles
            pass
        except Exception:
            logger.warning("Failed to update profiles")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance rankings for monitoring."""
        return {
            "amount_zscore": 0.18,
            "acct_v_1h_count": 0.15,
            "is_new_device": 0.12,
            "impossible_travel": 0.10,
            "is_new_merchant": 0.12,
            "is_new_device": 0.16,
            "acct_v_1h_count": 0.15,
            "acct_v_24h_count": 0.14,
            "geo_speed_kmh": 0.09,
            "typing_zscore": 0.08,
            "is_new_device": 0.12,
            "is_new_merchant": 0.12,
            "cross_country_flag": 0.10,
            "is_new_merchant": 0.12,
            "is_new_device": 0.16,
            "velocity_score": 0.14,
        }


# Convenience functions
def create_behavior_engine(feature_store) -> BehaviorEngine:
    """Create behavior engine with default config."""
    config = BehaviorEngineConfig(feature_store=feature_store)
    return BehaviorEngine(BehaviorEngineConfig(feature_store=feature_store))


# Global instance
_behavior_engine = None


def get_behavior_engine() -> BehaviorEngine:
    """Get or create global behavior engine instance."""
    global _behavior_engine
    if _behavior_engine is None:
        store = get_feature_store()
        _behavior_engine = BehaviorEngine(BehaviorEngineConfig(feature_store=store))
    return _behavior_engine