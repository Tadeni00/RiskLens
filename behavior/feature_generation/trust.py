"""
FraudTrap Behavioral Intelligence Layer
Trust Scoring Features
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Set, Optional
from datetime import datetime, timezone
from collections import defaultdict

from behavior.profiles.customer import CustomerBehaviorProfile
from behavior.profiles.merchant import MerchantBehaviorProfile
from behavior.profiles.device import DeviceBehaviorProfile
from behavior.profiles.beneficiary import BeneficiaryBehaviorProfile
from behavior.profiles.payment_instrument import PaymentInstrumentProfile
from behavior.profiles.device import DeviceBehaviorProfile
from behavior.profiles.beneficiary import BeneficiaryBehaviorProfile


@dataclass
class TrustScorer:
    """
    Computes trust scores for various entities based on behavioral profiles.
    """
    
    @staticmethod
    def get_device_trust_score(customer_profile, device_id: str) -> float:
        """Get trust score for a device (0-1)."""
        if not device_id:
            return 0.5
        if device_id in customer_profile.trusted_devices:
            return 1.0
        freq = customer_profile.device_fingerprint_frequency.get(device_id, 0)
        total = sum(customer_profile.device_fingerprint_frequency.values()) or 1
        return 1.0 - min(1.0, freq / max(1, total / 10))
    
    @staticmethod
    def get_merchant_trust_score(customer_profile, merchant_id: str) -> float:
        """Get trust score for a merchant (0-1)."""
        if not merchant_id:
            return 0.5
        freq = customer_profile.merchant_frequency.get(merchant_id, 0)
        total = sum(customer_profile.merchant_frequency.values()) or 1
        return freq / max(1, total)
    
    @staticmethod
    def get_beneficiary_trust_score(customer_profile, beneficiary_id: str) -> float:
        """Get trust score for a beneficiary (0-1)."""
        # Would need beneficiary tracking - simplified for now
        return 0.5
    
    @staticmethod
    def get_customer_reputation(customer_profile) -> float:
        """Get customer reputation score (0-1)."""
        return 1.0 - TrustScorer.get_customer_risk_score(customer_profile)
    
    @staticmethod
    def get_historical_chargeback_rate(customer_profile) -> float:
        if customer_profile.total_transactions == 0:
            return 0.0
        return customer_profile.chargeback_count / max(1, customer_profile.total_transactions)
    
    @staticmethod
    def get_historical_fraud_rate(customer_profile) -> float:
        if customer_profile.total_transactions == 0:
            return 0.0
        return len(customer_profile.fraud_history) / customer_profile.total_transactions
    
    @staticmethod
    def get_customer_risk_score(customer_profile) -> float:
        """Overall customer risk score (0-1)."""
        score = 0.0
        
        # Fraud history
        if customer_profile.fraud_history:
            score += min(0.5, len(customer_profile.fraud_history) * 0.1)
        
        # Chargeback rate
        if customer_profile.chargeback_count > 0:
            score += min(0.3, customer_profile.chargeback_count * 0.1)
        
        # Velocity anomalies
        if hasattr(customer_profile, 'velocity_stats') and customer_profile.velocity_stats.count > 10:
            zscore = customer_profile.velocity_stats.get_zscore(customer_profile.velocity_stats.mean)
            if zscore > 3:
                score += 0.2
        
        # Device trust
        if not customer_profile.trusted_devices and customer_profile.device_frequency:
            score += 0.1
        
        return min(1.0, score)
    
    @staticmethod
    def get_merchant_risk_score(customer_profile, merchant_id: str) -> float:
        return 1.0 - customer_profile.get_merchant_trust_score(merchant_id)
    
    @staticmethod
    def get_device_risk_score(customer_profile, device_id: str) -> float:
        return 1.0 - customer_profile.get_device_trust_score(device_id)


# Convenience functions
def get_device_trust_score(customer_profile, device_id: str) -> float:
    return customer_profile.get_device_trust_score(device_id)

def get_merchant_trust_score(customer_profile, merchant_id: str) -> float:
    return customer_profile.get_merchant_trust_score(merchant_id)

def get_beneficiary_trust_score(customer_profile, beneficiary_id: str) -> float:
    return 0.5  # Placeholder

def get_customer_reputation(customer_profile) -> float:
    return 1.0 - customer_profile.get_customer_risk_score()

def get_historical_chargeback_rate(customer_profile) -> float:
    if customer_profile.total_transactions == 0:
        return 0.0
    return customer_profile.chargeback_count / max(1, customer_profile.total_transactions)

def get_historical_fraud_rate(customer_profile) -> float:
    if customer_profile.total_transactions == 0:
        return 0.0
    return len(customer_profile.fraud_history) / customer_profile.total_transactions

def get_customer_risk_score(customer_profile) -> float:
    return customer_profile.get_customer_risk_score()

def get_merchant_risk_score(customer_profile, merchant_id: str) -> float:
    return 1.0 - customer_profile.get_merchant_trust_score(merchant_id)

def get_device_risk_score(customer_profile, device_id: str) -> float:
    return 1.0 - customer_profile.get_device_trust_score(device_id)


# Convenience functions for feature generation
def get_device_trust_score(customer_profile, device_id: str) -> float:
    return customer_profile.get_device_trust_score(device_id)

def get_merchant_trust_score(customer_profile, merchant_id: str) -> float:
    return customer_profile.get_merchant_trust_score(merchant_id)

def get_beneficiary_trust_score(customer_profile, beneficiary_id: str) -> float:
    return 0.5  # Placeholder

def get_customer_reputation(customer_profile) -> float:
    return 1.0 - customer_profile.get_customer_risk_score()

def get_historical_chargeback_rate(customer_profile) -> float:
    if customer_profile.total_transactions == 0:
        return 0.0
    return customer_profile.chargeback_count / max(1, customer_profile.total_transactions)

def get_historical_fraud_rate(customer_profile) -> float:
    if customer_profile.total_transactions == 0:
        return 0.0
    return len(customer_profile.fraud_history) / customer_profile.total_transactions

def get_customer_risk_score(customer_profile) -> float:
    return customer_profile.get_customer_risk_score()

def get_merchant_risk_score(customer_profile, merchant_id: str) -> float:
    return 1.0 - customer_profile.get_merchant_trust_score(merchant_id)

def get_device_risk_score(customer_profile, device_id: str) -> float:
    return 1.0 - customer_profile.get_device_trust_score(device_id)