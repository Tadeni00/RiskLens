"""
FraudTrap Behavioral Intelligence Layer
Beneficiary Behavior Profile
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Set, List
from collections import defaultdict

from behavior.utils.online_statistics import RollingWindow


@dataclass
class BeneficiaryBehaviorProfile:
    """
    Beneficiary/Recipient behavior profile.
    Tracks recipient behavior for detecting mule accounts, money laundering, etc.
    """

    beneficiary_id: str
    tenant_id: str

    # Basic info
    average_received_amount: float = 0.0
    total_received: float = 0.0
    transaction_count: int = 0

    # Sender diversity
    unique_senders: set = field(default_factory=set)
    sender_frequency: dict = field(default_factory=lambda: defaultdict(int))
    sender_diversity: float = 0.0

    # New sender frequency
    new_sender_frequency: float = 0.0
    new_sender_count: int = 0

    # Velocity
    velocity_last_hour: int = 0
    velocity_last_24h: int = 0
    velocity_last_7d: int = 0

    # Fraud tracking
    fraud_involvement: int = 0
    fraud_rate: float = 0.0
    fraud_types: list = field(default_factory=list)

    # Velocity tracking
    velocity_windows: dict = field(
        default_factory=lambda: {
            "1m": RollingWindow(max_size=100),
            "5m": RollingWindow(max_size=500),
            "1h": RollingWindow(max_size=1000),
            "24h": RollingWindow(max_size=1000),
            "7d": RollingWindow(max_size=5000),
        }
    )

    # Metadata
    first_seen: datetime = None
    last_seen: datetime = None
    last_updated: datetime = None

    def update(self, transaction: "Transaction", is_fraud: bool = False) -> None:
        """Update profile with transaction."""
        self.total_received += transaction.amount
        self.transaction_count += 1
        self.average_received_amount = self.total_received / max(
            1, self.transaction_count
        )

        # Track sender
        if transaction.account_id:
            self.unique_senders.add(transaction.account_id)
            self.sender_frequency[transaction.account_id] += 1

        # Update velocity
        # ... velocity window updates would go here

        if is_fraud:
            self.fraud_involvement += 1

        self.last_updated = datetime.now(timezone.utc)

    def get_risk_score(self) -> float:
        """Calculate risk score for this beneficiary."""
        score = 0.0

        # High velocity
        if self.velocity_last_hour > 10:
            score += 0.3

        # Low sender diversity
        if self.sender_diversity < 2:
            score += 0.2

        # Fraud history
        if self.fraud_involvement > 0:
            score += min(0.5, self.fraud_involvement * 0.1)

        return min(1.0, score)


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
