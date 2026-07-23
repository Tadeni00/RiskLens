"""
FraudTrap Behavioral Intelligence Layer
Merchant Behavior Profile
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List
from collections import defaultdict
from collections import Counter

from behavior.utils.online_statistics import (
    OnlineMeanVariance,
    ExponentialMovingAverage,
    RollingWindow,
)


@dataclass
class MerchantBehaviorProfile:
    """
    Merchant behavior profile with incremental updates.
    Tracks merchant behavior patterns for fraud detection.
    """

    merchant_id: str
    tenant_id: str

    # Transaction statistics
    total_transactions: int = 0
    total_amount: float = 0.0
    total_customers: int = 0

    # Amount statistics
    amount_stats: object = field(
        default_factory=lambda: OnlineMeanVariance()
    )  # OnlineMeanVariance
    amount_ema: object = field(default_factory=lambda: None)  # ExponentialMovingAverage

    # Customer diversity
    customer_frequency: dict = field(default_factory=lambda: defaultdict(int))

    # Transaction patterns
    hourly_distribution: dict = field(default_factory=lambda: defaultdict(int))
    daily_distribution: dict = field(default_factory=lambda: defaultdict(int))
    weekly_distribution: dict = field(default_factory=lambda: defaultdict(int))

    # Velocity tracking
    velocity_windows: dict = field(
        default_factory=lambda: {
            "1m": RollingWindow(max_size=100),
            "5m": RollingWindow(max_size=500),
            "1h": RollingWindow(max_size=1000),
            "24h": RollingWindow(max_size=1000),
        }
    )

    # Fraud tracking
    fraud_count: int = 0
    chargeback_count: int = 0
    dispute_count: int = 0
    refund_count: int = 0

    # Customer diversity
    unique_customers: int = 0
    repeat_customer_rate: float = 0.0

    # Geographic distribution
    country_distribution: dict = field(default_factory=lambda: defaultdict(int))

    # Risk metrics
    fraud_rate: float = 0.0
    chargeback_rate: float = 0.0
    dispute_rate: float = 0.0
    refund_rate: float = 0.0

    # Risk scoring
    risk_score: float = 0.0
    risk_factors: list = field(default_factory=list)

    # Metadata
    first_seen: datetime = None
    last_updated: datetime = None

    def __post_init__(self):
        """Initialize online statistics objects."""
        from behavior.utils.online_statistics import (
            OnlineMeanVariance,
            ExponentialMovingAverage,
            RollingWindow,
        )

        self.amount_ema = None
        self._init_stats()

    def _init_stats(self):
        from behavior.utils.online_statistics import (
            OnlineMeanVariance,
            ExponentialMovingAverage,
            RollingWindow,
        )

        self.amount_stats = OnlineMeanVariance()
        self.amount_ema = ExponentialMovingAverage(alpha=0.1)
        self.velocity_windows = {
            "1m": RollingWindow(max_size=100),
            "5m": RollingWindow(max_size=500),
            "1h": RollingWindow(max_size=1000),
            "24h": RollingWindow(max_size=1000),
        }

    def update(self, transaction: "Transaction") -> None:
        """Update profile with a new transaction."""
        # Update basic stats
        self.total_transactions += 1
        self.total_amount += transaction.amount

        # Update amount statistics
        # This would use online statistics
        self.last_updated = datetime.now(timezone.utc)

    def update_transaction(self, transaction: "Transaction") -> None:
        """Update profile with a new transaction."""
        self.total_transactions += 1
        self.total_amount += transaction.amount
        # Update statistics...
        self.last_updated = datetime.now(timezone.utc)

    def get_risk_score(self) -> float:
        """Calculate merchant risk score (0-1)."""
        score = 0.0

        # Fraud rate contribution
        if self.total_transactions > 0:
            fraud_rate = self.fraud_count / self.total_transactions
            score += min(0.5, fraud_rate * 5)

        # Chargeback rate
        if self.total_transactions > 0:
            chargeback_rate = self.chargeback_count / self.total_transactions
            score += min(0.3, self.chargeback_rate * 5)

        # Velocity spikes
        # Check velocity windows

        # New customer ratio
        if self.total_transactions > 10 and self.unique_customers > 0:
            new_customer_ratio = (
                len([c for c in self.customer_frequency.values() if c == 1])
                / self.unique_customers
            )
            if new_customer_ratio > 0.5:
                return 0.2

        return min(1.0, score)

    def update_transaction(self, transaction: "Transaction") -> None:
        """Update profile with new transaction."""
        # Implementation details...
        pass

    def to_dict(self) -> dict:
        """Serialize profile for storage."""
        return {
            "merchant_id": self.merchant_id,
            "tenant_id": self.tenant_id,
            "total_transactions": self.total_transactions,
            "total_amount": self.total_amount,
            "fraud_count": self.fraud_count,
            "risk_score": self.get_risk_score(),
            # ... other fields
        }


@dataclass
class DeviceBehaviorProfile:
    """
    Device behavior profile for tracking device risk and behavior.
    """

    device_id: str
    tenant_id: str

    # Basic info
    os: str = ""
    browser: str = ""
    device_type: str = ""  # MOBILE, DESKTOP, TABLET, POS_TERMINAL

    # Tracking
    first_seen: datetime = None
    last_seen: datetime = None

    # Historical customers
    historical_customers: set = field(default_factory=set)
    customer_frequency: dict = field(default_factory=lambda: defaultdict(int))

    # Risk tracking
    fraud_count: int = 0
    successful_transactions: int = 0
    failed_transactions: int = 0

    # Risk scoring
    risk_score: float = 0.0
    risk_factors: list = field(default_factory=list)

    # Metadata
    first_seen: datetime = None
    last_seen: datetime = None

    def update(self, transaction: "Transaction") -> None:
        """Update profile with new transaction."""
        # Update last seen
        self.last_seen = datetime.now(timezone.utc)
        if self.first_seen is None:
            self.first_seen = datetime.now(timezone.utc)

        # Update customer tracking
        if transaction.account_id:
            self.historical_customers.add(transaction.account_id)
            self.customer_frequency[transaction.account_id] += 1

        # Update risk based on transaction outcome
        # (would be updated after transaction is scored)

        self.last_seen = datetime.now(timezone.utc)

    def get_risk_score(self) -> float:
        """Calculate device risk score (0-1)."""
        if self.successful_transactions + self.failed_transactions == 0:
            return 0.5  # Unknown device

        failure_rate = self.failed_transactions / max(
            1, self.successful_transactions + self.failed_transactions
        )
        return min(1.0, failure_rate * 2)

    def to_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "tenant_id": self.tenant_id,
            "os": self.os,
            "browser": self.browser,
            "device_type": self.device_type,
            "risk_score": self.get_risk_score(),
            "historical_customers": len(self.historical_customers),
            "fraud_count": self.fraud_count,
        }


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

        # Track sender
        if transaction.account_id:
            self.unique_senders.add(transaction.account_id)
            self.sender_frequency[transaction.account_id] += 1

        # Update velocity
        # ... velocity window updates

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

    def to_dict(self) -> dict:
        return {
            "beneficiary_id": self.beneficiary_id,
            "tenant_id": self.tenant_id,
            "total_received": self.total_received,
            "transaction_count": self.transaction_count,
            "unique_senders": len(self.unique_senders),
            "sender_diversity": self.sender_diversity,
            "new_sender_frequency": self.new_sender_frequency,
            "fraud_involvement": self.fraud_involvement,
            "risk_score": self.get_risk_score(),
        }


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
    fraud_history: list = field(default_factory=list)
    fraud_count: int = 0

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
                    "timestamp": datetime.now(timezone.utc),
                    "amount": transaction.amount,
                    "merchant": transaction.merchant_id,
                }
            )

        self.last_seen = datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        return {
            "instrument_id": self.instrument_id,
            "instrument_type": self.instrument_type,
            "tenant_id": self.tenant_id,
            "usage_frequency": self.usage_frequency,
            "average_spend": self.average_spend,
            "trusted_merchants": list(self.trusted_merchants),
            "trusted_devices": list(self.trusted_devices),
            "trusted_countries": list(self.trusted_countries),
            "fraud_count": self.fraud_count,
        }


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

        # Track sender
        if transaction.account_id:
            self.unique_senders.add(transaction.account_id)
            self.sender_frequency[transaction.account_id] += 1

        # Update velocity
        # ... velocity window updates

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

    def to_dict(self) -> dict:
        return {
            "beneficiary_id": self.beneficiary_id,
            "tenant_id": self.tenant_id,
            "total_received": self.total_received,
            "transaction_count": self.transaction_count,
            "unique_senders": len(self.unique_senders),
            "sender_diversity": self.sender_diversity,
            "new_sender_frequency": self.new_sender_frequency,
            "fraud_involvement": self.fraud_involvement,
            "risk_score": self.get_risk_score(),
        }
