"""
FraudTrap Behavioral Intelligence Layer
Payment Instrument Profile
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, Set, List
from collections import defaultdict


@dataclass
class PaymentInstrumentProfile:
    """
    Payment instrument (card, account, wallet) behavior profile.
    """

    instrument_id: str
    instrument_type: str  # CARD, ACCOUNT, WALLET
    tenant_id: str

    # Usage statistics
    usage_frequency: int = 0
    average_spend: float = 0.0
    total_spend: float = 0.0

    # Trusted entities
    trusted_merchants: set = field(default_factory=set)
    trusted_devices: set = field(default_factory=set)
    trusted_countries: set = field(default_factory=set)

    # Fraud history
    fraud_count: int = 0
    fraud_history: list = field(default_factory=list)

    # Metadata
    first_seen: datetime = None
    last_seen: datetime = None

    def update(self, transaction: "Transaction", is_fraud: bool = False) -> None:
        """Update profile with transaction."""
        self.total_transactions += 1
        self.total_spend += transaction.amount
        self.average_spend = self.total_spend / max(1, self.total_transactions)

        if is_fraud:
            self.fraud_count += 1
            self.fraud_history.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "amount": transaction.amount,
                    "merchant": transaction.merchant_id,
                }
            )

    def save(self, path: str) -> None:
        pass

    @classmethod
    def load(cls, path: str) -> "PaymentInstrumentProfile":
        pass
