"""
FraudTrap Behavioral Intelligence Layer
Device Behavior Profile
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Set, List
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta


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
        
        failure_rate = self.failed_transactions / max(1, self.successful_transactions + self.failed_transactions)
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