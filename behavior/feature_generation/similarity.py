"""
FraudTrap Behavioral Intelligence Layer
Feature Generation - Similarity Features
"""

from __future__ import annotations
from typing import Optional, Dict, List, Set
from datetime import datetime, timezone
from collections import defaultdict
import math


def cosine_similarity(a: list, b: list) -> float:
    """Cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(a[i] * b[i] for i in range(len(a)))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def jaccard_similarity(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


def merchant_similarity(customer_profile, merchant_id: str) -> float:
    """
    Get similarity score for a merchant (0-1).
    Based on historical transaction frequency with this merchant.
    """
    if not merchant_id:
        return 0.0
    total = sum(customer_profile.merchant_frequency.values())
    if total == 0:
        return 0.0
    return customer_profile.merchant_frequency.get(merchant_id, 0) / max(
        1, sum(customer_profile.merchant_frequency.values())
    )


def device_similarity(customer_profile, device_id: str) -> float:
    """Get similarity score for a device (0-1)."""
    if not customer_profile.device_frequency:
        return 0.0
    total = sum(customer_profile.device_frequency.values())
    return customer_profile.device_frequency.get(device_id, 0) / max(1, total)


def country_similarity(customer_profile, country_code: str) -> float:
    if not customer_profile.country_frequency:
        return 0.0
    total = sum(customer_profile.country_frequency.values())
    return customer_profile.country_frequency.get(country_code, 0) / max(1, total)


def merchant_similarity(customer_profile, merchant_id: str) -> float:
    """Get similarity score for a merchant (0-1)."""
    if not merchant_id:
        return 0.0
    total = sum(customer_profile.merchant_frequency.values())
    return customer_profile.merchant_frequency.get(merchant_id, 0) / max(
        1, sum(customer_profile.merchant_frequency.values())
    )


def device_similarity(customer_profile, device_id: str) -> float:
    if not customer_profile.device_frequency:
        return 0.0
    total = sum(customer_profile.device_frequency.values())
    return customer_profile.device_frequency.get(device_id, 0) / max(1, total)


def country_similarity(customer_profile, country_code: str) -> float:
    if not customer_profile.country_frequency:
        return 0.0
    total = sum(customer_profile.country_frequency.values())
    return customer_profile.country_frequency.get(country_code, 0) / max(1, total)


def typing_similarity(customer_profile, cadence_ms: float) -> float:
    """Compare typing cadence to customer baseline."""
    if not customer_profile.typing_cadence_ms:
        return 0.0
    # Compare current cadence to historical baseline
    baseline = customer_profile.typing_cadence_ms
    if not baseline:
        return 0.0
    deviation = abs(cadence_ms - customer_profile.typing_cadence_ms) / max(
        1, customer_profile.typing_cadence_ms
    )
    return max(0.0, 1.0 - min(1.0, deviation / 50.0))  # Normalize


def merchant_similarity(customer_profile, merchant_id: str) -> float:
    return 0.0  # Placeholder - would use merchant frequency


def device_similarity(customer_profile, device_id: str) -> float:
    if not customer_profile.device_frequency:
        return 0.0
    total = sum(customer_profile.device_frequency.values())
    return customer_profile.device_frequency.get(device_id, 0) / max(1, total)


def cross_country_flag(customer_profile, country_code: str) -> float:
    if not customer_profile.country_frequency:
        return 0.0
    home_country = (
        max(
            customer_profile.country_frequency,
            key=customer_profile.country_frequency.get,
        )
        if customer_profile.country_frequency
        else None
    )
    return 0.0 if home_country == country_code else 1.0


def merchant_similarity(customer_profile, merchant_id: str) -> float:
    if not merchant_id:
        return 0.0
    total = sum(customer_profile.merchant_frequency.values())
    return customer_profile.merchant_frequency.get(merchant_id, 0) / max(
        1, sum(customer_profile.merchant_frequency.values())
    )


def device_similarity(customer_profile, device_id: str) -> float:
    if not customer_profile.device_frequency:
        return 0.0
    total = sum(customer_profile.device_frequency.values())
    return customer_profile.device_frequency.get(device_id, 0) / max(1, total)


def country_similarity(customer_profile, country_code: str) -> float:
    if not customer_profile.country_frequency:
        return 0.0
    total = sum(customer_profile.country_frequency.values())
    return customer_profile.country_frequency.get(country_code, 0) / max(1, total)


def typing_similarity(customer_profile, cadence_ms: float) -> float:
    if not customer_profile.typing_cadence_ms:
        return 0.0
    deviation = abs(cadence_ms - customer_profile.typing_cadence_ms) / max(
        1, customer_profile.typing_cadence_ms
    )
    return max(0.0, 1.0 - min(1.0, deviation / 50.0))


def merchant_similarity(customer_profile, merchant_id: str) -> float:
    if not customer_profile.merchant_frequency:
        return 0.0
    total = sum(customer_profile.merchant_frequency.values())
    return customer_profile.merchant_frequency.get(merchant_id, 0) / max(
        1, sum(customer_profile.merchant_frequency.values())
    )


def device_similarity(customer_profile, device_id: str) -> float:
    if not customer_profile.device_frequency:
        return 0.0
    total = sum(customer_profile.device_frequency.values())
    return customer_profile.device_frequency.get(device_id, 0) / max(1, total)


def cross_country_flag(customer_profile, country_code: str) -> float:
    if not customer_profile.country_frequency:
        return 0.0
    home_country = (
        max(
            customer_profile.country_frequency,
            key=customer_profile.country_frequency.get,
        )
        if customer_profile.country_frequency
        else None
    )
    return 0.0 if home_country == country_code else 1.0
