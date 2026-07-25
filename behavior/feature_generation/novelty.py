"""
RiskLens Behavioral Intelligence Layer
Novelty Detection Features
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Set
from datetime import datetime, timezone
from collections import defaultdict

from behavior.profiles.customer import CustomerBehaviorProfile


@dataclass
class NoveltyDetector:
    """
    Detects novel/unseen entities and patterns in transactions.
    """

    @staticmethod
    def get_new_device_flag(
        customer_profile: CustomerBehaviorProfile, device_id: Optional[str]
    ) -> float:
        """Check if device is new (0 or 1)."""
        if not device_id:
            return 1.0  # No device = unknown = suspicious
        return 0.0 if device_id in customer_profile.trusted_devices else 1.0

    @staticmethod
    def get_new_ip_flag(customer_profile, ip_hash: Optional[str]) -> float:
        if not ip_hash:
            return 1.0
        return 0.0 if ip_hash in customer_profile.trusted_ips else 1.0

    @staticmethod
    def get_new_country_flag(customer_profile, country_code: Optional[str]) -> float:
        if not country_code:
            return 1.0
        return (
            0.0
            if country_code.upper()
            in [c.upper() for c in customer_profile.country_frequency.keys()]
            else 1.0
        )

    @staticmethod
    def get_new_merchant_flag(customer_profile, merchant_id: Optional[str]) -> float:
        if not merchant_id:
            return 0.0
        return 0.0 if merchant_id in customer_profile.merchant_frequency else 1.0

    @staticmethod
    def get_new_device_flag(customer_profile, device_id: Optional[str]) -> float:
        if not device_id:
            return 1.0
        return 0.0 if device_id in customer_profile.trusted_devices else 1.0

    @staticmethod
    def get_new_beneficiary_flag(
        customer_profile, beneficiary_id: Optional[str]
    ) -> float:
        # Would need beneficiary tracking - simplified for now
        return 0.0

    @staticmethod
    def get_new_ip_flag(customer_profile, ip_hash: str) -> float:
        return 0.0 if ip_hash in customer_profile.trusted_ips else 1.0


# Convenience functions
def get_new_device_flag(customer_profile, device_id: Optional[str]) -> float:
    if not device_id:
        return 1.0
    return 0.0 if device_id in customer_profile.trusted_devices else 1.0


def get_new_merchant_flag(customer_profile, merchant_id: Optional[str]) -> float:
    if not merchant_id:
        return 0.0
    return 0.0 if merchant_id in customer_profile.merchant_frequency else 1.0


def get_new_ip_flag(customer_profile, ip_hash: str) -> float:
    return 0.0 if ip_hash in customer_profile.trusted_ips else 1.0


def get_new_country_flag(customer_profile, country_code: str) -> float:
    if not country_code:
        return 1.0
    return (
        0.0
        if country_code.upper()
        in [c.upper() for c in customer_profile.country_frequency.keys()]
        else 1.0
    )


def get_new_merchant_flag(customer_profile, merchant_id: str) -> float:
    if not merchant_id:
        return 0.0
    return 0.0 if merchant_id in customer_profile.merchant_frequency else 1.0


def get_new_device_flag(customer_profile, device_id: Optional[str]) -> float:
    if not device_id:
        return 1.0
    return 0.0 if device_id in customer_profile.trusted_devices else 1.0
