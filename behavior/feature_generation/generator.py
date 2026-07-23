"""
FraudTrap Behavioral Intelligence Layer
Feature Generation - Behavioral Features
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
from datetime import datetime, timezone
import math

from behavior.profiles.customer import CustomerBehaviorProfile
from behavior.profiles.merchant import MerchantBehaviorProfile
from behavior.profiles.device import DeviceBehaviorProfile
from behavior.profiles.beneficiary import BeneficiaryBehaviorProfile
from behavior.profiles.payment_instrument import PaymentInstrumentProfile
from behavior.profiles.device import DeviceBehaviorProfile
from behavior.profiles.beneficiary import BeneficiaryBehaviorProfile
from behavior.profiles.payment_instrument import PaymentInstrumentProfile


@dataclass
class BehavioralFeatureGenerator:
    """
    Generates a comprehensive behavioral feature vector for a transaction.
    This is the main entry point for generating behavioral features during scoring.
    """

    def __init__(self, feature_store=None):
        self.feature_store = feature_store

    def generate_features(
        self,
        transaction,
        customer_profile: "CustomerBehaviorProfile" = None,
        merchant_profile: "MerchantBehaviorProfile" = None,
        device_profile: "DeviceBehaviorProfile" = None,
        beneficiary_profile: "BeneficiaryBehaviorProfile" = None,
        instrument_profile: "PaymentInstrumentProfile" = None,
    ) -> Dict[str, float]:
        """
        Generate comprehensive behavioral feature vector for a transaction.

        Args:
            transaction: The transaction being scored
            customer_profile: CustomerBehaviorProfile (optional)
            merchant_profile: MerchantBehaviorProfile (optional)
            device_profile: DeviceBehaviorProfile (optional)
            beneficiary_profile: BeneficiaryBehaviorProfile (optional)
            instrument_profile: PaymentInstrumentProfile (optional)

        Returns:
            Dict of behavioral features
        """
        features = {}

        # Amount features
        features.update(self._generate_amount_features(transaction))

        # Customer behavioral features
        if customer_profile:
            features.update(
                self._generate_customer_features(transaction, customer_profile)
            )

        # Merchant features
        if merchant_profile:
            features.update(
                self._generate_merchant_features(transaction, merchant_profile)
            )

        # Device features
        if device_profile:
            features.update(self._generate_device_features(transaction, device_profile))

        # Beneficiary features
        if beneficiary_profile:
            features.update(
                self._generate_beneficiary_features(transaction, beneficiary_profile)
            )

        # Instrument features
        if instrument_profile:
            features.update(
                self._generate_instrument_features(transaction, instrument_profile)
            )

        # Cross-entity similarity features
        features.update(
            self._generate_cross_entity_features(
                transaction,
                **{
                    "customer": customer_profile,
                    "merchant": merchant_profile,
                    "device": device_profile,
                    "beneficiary": beneficiary_profile,
                    "instrument": instrument_profile,
                },
            )
        )

        return features

    def _generate_amount_features(self, transaction) -> Dict[str, float]:
        """Generate amount-based features."""
        features = {}
        amount = transaction.amount

        features["amount"] = transaction.amount
        features["amount_log"] = math.log1p(transaction.amount)

        # Amount features would be populated by customer profile
        # amount_zscore, amount_vs_mean_ratio, etc.
        return features

    def _generate_customer_features(self, transaction, profile) -> Dict[str, float]:
        """Generate customer behavioral features."""
        features = {}

        # Amount features
        if hasattr(profile, "amount_stats") and profile.amount_stats.count > 1:
            features["customer_amount_zscore"] = profile.amount_stats.get_zscore(
                transaction.amount
            )

        if hasattr(profile, "amount_ema") and profile.amount_ema:
            features["amount_vs_ema"] = transaction.amount / max(
                profile.amount_ema.get(), 1
            )

        # Velocity features
        if hasattr(profile, "velocity_windows"):
            for window_name, window in profile.velocity_windows.items():
                prefix = f"acct_v_{window_name}"
                if window.count > 0:
                    features[f"{prefix}_count"] = float(window.count)
                    features[f"{prefix}_total_amt"] = (
                        window.sum if hasattr(window, "sum") else 0.0
                    )
                    features[f"{prefix}_mean_amt"] = (
                        window.mean if window.count > 0 else 0.0
                    )

        # Device features
        if hasattr(profile, "device_velocity_windows"):
            for window_name, window in profile.device_velocity_windows.items():
                if window.count > 0:
                    prefix = f"dev_v_{window_name}"
                    features[f"{prefix}_count"] = float(window.count)

        # Device trust
        if hasattr(profile, "trusted_devices"):
            features["is_new_device"] = (
                0.0 if transaction.device_id in profile.trusted_devices else 1.0
            )

        # Merchant features
        if hasattr(profile, "merchant_frequency"):
            features["is_new_merchant"] = (
                0.0 if transaction.merchant_id in profile.merchant_frequency else 1.0
            )

        # Country
        if hasattr(profile, "country_frequency"):
            features["cross_country_flag"] = (
                0.0 if transaction.country_code in profile.country_frequency else 1.0
            )

        # Time features
        features["is_night"] = (
            1.0
            if transaction.timestamp.hour < 6 or transaction.timestamp.hour > 22
            else 0.0
        )
        features["is_weekend"] = 1.0 if transaction.timestamp.weekday() >= 5 else 0.0

        # Hour encoding
        hour = transaction.timestamp.hour
        features["hour_sin"] = math.sin(2 * math.pi * hour / 24)
        features["hour_cos"] = math.cos(2 * math.pi * hour / 24)

        # Weekday
        features["day_of_week"] = float(transaction.timestamp.weekday())
        features["is_weekend"] = 1.0 if transaction.timestamp.weekday() >= 5 else 0.0

        return features

    def _generate_merchant_features(self, transaction, profile) -> Dict[str, float]:
        """Generate merchant behavioral features."""
        features = {}

        if hasattr(profile, "risk_score"):
            features["merchant_risk_score"] = profile.get_risk_score()

        if hasattr(profile, "fraud_rate"):
            features["merchant_fraud_rate"] = profile.fraud_rate

        if hasattr(profile, "amount_stats"):
            features["merchant_avg_amount"] = profile.amount_stats.mean

        return features

    def _generate_device_features(self, transaction, profile) -> Dict[str, float]:
        features = {}

        if hasattr(profile, "get_risk_score"):
            features["device_risk_score"] = profile.get_risk_score()

        # Check for trusted_devices if it exists
        if hasattr(profile, "trusted_devices"):
            features["is_new_device"] = (
                0.0 if transaction.device_id in profile.trusted_devices else 1.0
            )

        if hasattr(profile, "device_frequency") and transaction.device_id:
            features["device_account_count"] = float(
                profile.device_frequency.get(transaction.device_id, 0)
            )

        # Also check historical_customers as alternative
        if hasattr(profile, "historical_customers"):
            features["device_historical_customers"] = float(
                len(profile.historical_customers)
            )

        return features

    def _generate_beneficiary_features(self, transaction, profile) -> Dict[str, float]:
        features = {}

        if hasattr(profile, "get_risk_score"):
            features["beneficiary_risk_score"] = profile.get_risk_score()

        features["new_sender_frequency"] = profile.new_sender_frequency

        if hasattr(profile, "velocity_windows"):
            for window_name, window in profile.velocity_windows.items():
                prefix = f"benef_v_{window_name}"
                features[f"{prefix}_count"] = float(window.count)

        return features

    def _generate_instrument_features(self, transaction, profile) -> Dict[str, float]:
        features = {}

        if hasattr(profile, "fraud_count"):
            features["instrument_fraud_count"] = profile.fraud_count

        features["is_new_instrument"] = 0.0  # Would check against trusted instruments

        return features

    def _generate_cross_entity_features(
        self,
        transaction,
        customer=None,
        merchant=None,
        device=None,
        beneficiary=None,
        instrument=None,
    ) -> Dict[str, float]:
        """Generate cross-entity similarity and relationship features."""
        features = {}

        # Customer-Merchant similarity
        if customer and merchant:
            # Would compare customer's preferred merchants with this merchant
            pass

        # Customer-Device similarity
        if customer and device:
            pass

        # Device-Customer trust
        if device and customer:
            pass

        return features


# Convenience function for easy use
def generate_behavioral_features(
    transaction,
    customer_profile=None,
    merchant_profile=None,
    device_profile=None,
    beneficiary_profile=None,
    instrument_profile=None,
) -> Dict[str, float]:
    """
    Main entry point for generating behavioral features.

    Usage:
        features = generate_behavioral_features(
            transaction=txn,
            customer_profile=customer_profile,
            merchant_profile=merchant_profile,
            device_profile=device_profile,
        )
    """
    generator = BehavioralFeatureGenerator()
    return generator.generate_features(
        transaction=transaction,
        customer_profile=customer_profile,
        merchant_profile=merchant_profile,
        device_profile=device_profile,
        beneficiary_profile=beneficiary_profile,
        instrument_profile=instrument_profile,
    )
