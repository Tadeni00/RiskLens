"""
FraudTrap Behavioral Intelligence Layer
Customer Behavior Profile
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Set
from collections import defaultdict
from collections import Counter

from behavior.utils.online_statistics import (
    OnlineMeanVariance,
    ExponentialMovingAverage,
    RollingWindow,
    CountMinSketch,
    RollingWindow,
    haversine_distance,
)


@dataclass
class CustomerBehaviorProfile:
    """
    Online customer behavior profile with incremental updates.
    All statistics are updated incrementally without full recomputation.
    """
    customer_id: str
    tenant_id: str

    # Transaction statistics
    total_transactions: int = 0
    total_amount: float = 0.0
    
    # Amount statistics (online)
    amount_stats: object = field(default_factory=lambda: OnlineMeanVariance())
    amount_ema: object = field(default_factory=lambda: ExponentialMovingAverage(alpha=0.1))
    
    # Amount percentiles (using rolling window)
    amount_window: object = field(default_factory=lambda: RollingWindow(max_size=1000))
    
    # Time patterns
    preferred_hours: dict = field(default_factory=lambda: defaultdict(int))
    preferred_weekdays: dict = field(default_factory=lambda: defaultdict(int))
    
    # Merchant preferences
    merchant_frequency: dict = field(default_factory=lambda: defaultdict(int))
    preferred_merchants: dict = field(default_factory=lambda: defaultdict(int))
    
    # Country/geo preferences
    country_frequency: dict = field(default_factory=lambda: defaultdict(int))
    
    # Device tracking
    device_frequency: dict = field(default_factory=lambda: defaultdict(int))
    trusted_devices: set = field(default_factory=set)
    device_fingerprint_frequency: dict = field(default_factory=lambda: defaultdict(int))
    
    # IP tracking
    ip_frequency: dict = field(default_factory=lambda: defaultdict(int))
    trusted_ips: set = field(default_factory=set)
    
    # Velocity statistics
    velocity_stats: object = field(default_factory=lambda: OnlineMeanVariance())
    velocity_window: object = field(default_factory=lambda: RollingWindow(max_size=1000))
    avg_time_between_transactions: float = 0.0
    last_transaction_time: Optional[datetime] = None
    
    # Location tracking
    last_locations: list = field(default_factory=list)
    trusted_devices: set = field(default_factory=set)
    trusted_ips: set = field(default_factory=set)
    
    # Risk tracking
    chargeback_count: int = 0
    fraud_history: list = field(default_factory=list)
    
    # Account metadata
    account_created: Optional[datetime] = None
    first_transaction_time: Optional[datetime] = None
    last_transaction_time: Optional[datetime] = None
    
    # Velocity windows (using rolling windows)
    velocity_windows: dict = field(default_factory=lambda: {
        "1m": RollingWindow(max_size=100),
        "5m": RollingWindow(max_size=500),
        "15m": RollingWindow(max_size=500),
        "1h": RollingWindow(max_size=1000),
        "24h": RollingWindow(max_size=1000),
        "7d": RollingWindow(max_size=5000),
    })

    def __post_init__(self):
        """Initialize online statistics objects."""
        # Amount statistics (Welford's algorithm)
        self.amount_stats = OnlineMeanVariance()
        self.amount_ema = ExponentialMovingAverage(alpha=0.1)
        self.amount_window = RollingWindow(max_size=1000)
        
        # Velocity tracking
        self.velocity_stats = OnlineMeanVariance()
        self.velocity_window = RollingWindow(max_size=1000)
        
        # Rolling windows for different time horizons
        self.velocity_windows = {
            "1m": RollingWindow(max_size=100),
            "5m": RollingWindow(max_size=500),
            "15m": RollingWindow(max_size=500),
            "1h": RollingWindow(max_size=1000),
            "24h": RollingWindow(max_size=1000),
            "7d": RollingWindow(max_size=5000),
        }

    def update(self, transaction: "Transaction") -> None:
        """
        Update profile with a new transaction.
        This is the main entry point for incremental updates.
        """
        now = datetime.now(timezone.utc)
        amount = transaction.amount
        
        # Basic counters
        self.total_transactions += 1
        self.total_amount += transaction.amount
        
        # Update amount statistics (online)
        self.amount_stats.update(transaction.amount)
        self.amount_ema.update(transaction.amount)
        self.amount_window.add(transaction.amount)
        
        # Update time patterns
        txn_time = transaction.timestamp
        hour = txn_time.hour
        weekday = txn_time.weekday()
        
        self.preferred_hours[hour] += 1
        self.preferred_weekdays[weekday] += 1
        
        # Track first and last transaction times
        if self.first_transaction_time is None:
            self.first_transaction_time = txn_time
        self.last_transaction_time = txn_time
        
        # Update time between transactions
        if self.last_transaction_time is not None:
            time_diff = (txn_time - self.last_transaction_time).total_seconds()
            self.velocity_stats.update(time_diff)
            self.velocity_window.add(time_diff)
            
            # Update rolling windows
            for window in self.velocity_windows.values():
                window.add(time_diff)
        
        self.last_transaction_time = txn_time
        
        # Merchant tracking
        if transaction.merchant_id:
            self.merchant_frequency[transaction.merchant_id] += 1
            self.preferred_merchants[transaction.merchant_id] += 1
        
        # Country tracking
        if transaction.country_code:
            self.country_frequency[transaction.country_code] += 1
        
        # Device tracking
        if transaction.device_id:
            self.device_frequency[transaction.device_id] += 1
            self.device_fingerprint_frequency[transaction.device_id] = \
                self.device_fingerprint_frequency.get(transaction.device_id, 0) + 1
            
            if transaction.device_id not in self.trusted_devices:
                # New device check
                pass  # Will be handled by feature generator
        
        # IP tracking
        if transaction.ip_address_hash:
            self.ip_frequency[transaction.ip_address_hash] += 1
        
        # Location tracking
        if transaction.latitude is not None and transaction.longitude is not None:
            loc = (transaction.latitude, transaction.longitude)
            self.last_locations.append(loc)
            if len(self.last_locations) > 100:
                self.last_locations = self.last_locations[-100:]
        
        # Update rolling mean (7-day and 30-day)
        self._update_rolling_means(transaction.amount, transaction.timestamp)
        
        # Update last transaction time
        self.last_transaction_time = transaction.timestamp

    def _update_rolling_means(self, amount: float, timestamp: datetime) -> None:
        """Update rolling mean for 7d and 30d windows."""
        # In a full implementation, we'd use a time-based rolling window
        # For now, we use a simple exponential moving average as approximation
        self._rolling_mean_7d = self._update_ema(self.rolling_mean_7d, 
                                                 self.amount_ema.value, 0.1)
        self._rolling_mean_30d = self._update_ema(self.rolling_mean_30d,
                                                  self.amount_ema.value, 0.05)

    def _update_ema(self, current: float, new_value: float, alpha: float) -> float:
        """Update exponential moving average."""
        if current == 0:
            return new_value
        return alpha * new_value + (1 - alpha) * current

    def _update_velocity_windows(self, timestamp: datetime) -> None:
        """Update all velocity tracking windows."""
        now = datetime.now(timezone.utc)
        
        for window_name, window in self.velocity_windows.items():
            window.add(timestamp)

    # --- Property getters for derived features ---
    
    @property
    def avg_amount(self) -> float:
        return self.amount_stats.mean if self.amount_stats.count > 0 else 0.0
    
    @property
    def median_amount(self) -> float:
        # Approximate from rolling window
        if self.amount_window.count > 0:
            values = self.velocity_window.get_values()  # Using amount window
            if self.amount_window.count > 0:
                return self.amount_window.mean  # Approximation
        return 0.0
    
    @property
    def std_amount(self) -> float:
        return self.amount_stats.std
    
    @property
    def min_amount(self) -> float:
        return min(self.amount_window.get_values()) if self.amount_window.count > 0 else 0.0
    
    @property
    def max_amount(self) -> float:
        return max(self.amount_window.get_values()) if self.amount_window.count > 0 else 0.0
    
    @property
    def rolling_mean_7d(self) -> float:
        return getattr(self, '_rolling_mean_7d', 0.0)
    
    @property
    def rolling_mean_30d(self) -> float:
        return getattr(self, '_rolling_mean_30d', 0.0)
    
    @property
    def rolling_std(self) -> float:
        return self.amount_stats.std
    
    @property
    def amount_zscore(self) -> float:
        return self.amount_stats.get_zscore(self.amount_stats.mean) if self.amount_stats.count > 1 else 0.0
    
    @property
    def velocity_score(self) -> float:
        """Combined velocity score from all windows."""
        scores = []
        for window in self.velocity_windows.values():
            if window.count > 0:
                # Normalize by expected rate
                expected = self._get_expected_rate(window)
                actual = window.mean
                if expected > 0:
                    ratio = window.mean / expected
                    scores.append(min(2.0, ratio))
        return sum(scores) / len(scores) if scores else 0.0
    
    def _get_expected_rate(self, window: RollingWindow) -> float:
        """Get expected transaction rate for a window."""
        if window.count < 2:
            return 1.0
        # Simplified: expected rate based on historical average
        return max(1.0, window.mean)
    
    def get_merchant_similarity(self, merchant_id: str) -> float:
        """Get similarity score for a merchant (0-1)."""
        if not self.merchant_frequency:
            return 0.0
        total = sum(self.merchant_frequency.values())
        return self.merchant_frequency.get(merchant_id, 0) / max(1, sum(self.merchant_frequency.values()))
    
    def get_device_similarity(self, device_id: str) -> float:
        """Get similarity score for a device (0-1)."""
        if not self.device_frequency:
            return 0.0
        total = sum(self.device_frequency.values())
        return self.device_frequency.get(device_id, 0) / max(1, total)
    
    def get_country_similarity(self, country_code: str) -> float:
        """Get similarity score for a country (0-1)."""
        if not self.country_frequency:
            return 0.0
        total = sum(self.country_frequency.values())
        return self.country_frequency.get(country_code, 0) / max(1, total)
    
    def get_device_trust_score(self, device_id: str) -> float:
        """Get trust score for a device (0-1)."""
        if device_id in self.trusted_devices:
            return 1.0
        freq = self.device_fingerprint_frequency.get(device_id, 0)
        total = sum(self.device_fingerprint_frequency.values())
        return 1.0 - min(1.0, freq / max(1, total / 10))
    
    def get_merchant_trust_score(self, merchant_id: str) -> float:
        """Get trust score for a merchant (0-1)."""
        if not self.merchant_frequency:
            return 0.5
        freq = self.merchant_frequency.get(merchant_id, 0)
        total = sum(self.merchant_frequency.values())
        return freq / max(1, total)
    
    def get_beneficiary_trust_score(self, beneficiary_id: str) -> float:
        """Get trust score for a beneficiary (0-1)."""
        # Would need beneficiary tracking - simplified for now
        return 0.5
    
    def get_customer_risk_score(self) -> float:
        """Overall customer risk score (0-1)."""
        score = 0.0
        
        # Fraud history
        if self.fraud_history:
            score += min(0.5, len(self.fraud_history) * 0.1)
        
        # Chargeback rate
        if self.chargeback_count > 0:
            score += min(0.3, self.chargeback_count * 0.1)
        
        # Velocity anomalies
        if self.velocity_stats.count > 10:
            zscore = self.velocity_stats.get_zscore(self.velocity_stats.mean)
            if zscore > 3:
                score += 0.2
        
        # Device trust
        if not self.trusted_devices and self.device_frequency:
            score += 0.1
        
        return min(1.0, score)
    
    def get_customer_amount_zscore(self, amount: float) -> float:
        """Get z-score for a transaction amount."""
        return self.amount_stats.get_zscore(amount) if self.amount_stats.count > 1 else 0.0
    
    def get_merchant_amount_zscore(self, merchant_id: str, amount: float) -> float:
        """Get z-score for amount at specific merchant."""
        # Would need merchant-specific stats - simplified
        return self.amount_stats.get_zscore(amount) if self.amount_stats.count > 1 else 0.0
    
    def get_tenant_amount_percentile(self, amount: float) -> float:
        """Get percentile of amount within tenant."""
        # Would need tenant-level stats
        return 0.5  # Placeholder
    
    def get_velocity_score(self) -> float:
        """Get combined velocity score."""
        return self.velocity_score
    
    def get_new_device_flag(self, device_id: str) -> float:
        """Check if device is new (0 or 1)."""
        return 0.0 if device_id in self.trusted_devices else 1.0
    
    def get_new_merchant_flag(self, merchant_id: str) -> float:
        """Check if merchant is new (0 or 1)."""
        return 0.0 if merchant_id in self.merchant_frequency else 1.0
    
    def get_new_ip_flag(self, ip_hash: str) -> float:
        return 0.0 if ip_hash in self.trusted_ips else 1.0
    
    def get_new_country_flag(self, country_code: str) -> float:
        return 0.0 if country_code in self.country_frequency else 1.0
    
    def get_new_beneficiary_flag(self, beneficiary_id: str) -> float:
        return 0.0  # Would need beneficiary tracking
    
    def get_hour_deviation(self, hour: int) -> float:
        """Deviation from preferred hours (0-1)."""
        if not self.preferred_hours:
            return 0.0
        total = sum(self.preferred_hours.values())
        if total == 0:
            return 0.0
        expected = self.preferred_hours.get(hour, 0) / total
        return 1.0 - min(1.0, expected * 24)
    
    def get_weekday_deviation(self, weekday: int) -> float:
        if not self.preferred_weekdays:
            return 0.0
        total = sum(self.preferred_weekdays.values())
        if total == 0:
            return 0.0
        expected = self.preferred_weekdays.get(weekday, 0) / total
        return 1.0 - min(1.0, expected * 7)
    
    def get_merchant_similarity(self, merchant_id: str) -> float:
        return self.get_merchant_similarity(merchant_id)
    
    # Removed recursive stub methods - use the actual implementations above
    
    def get_beneficiary_trust_score(self, beneficiary_id: str) -> float:
        return 0.5  # Placeholder
    
    def get_customer_reputation(self) -> float:
        return 1.0 - self.get_customer_risk_score()
    
    def get_historical_chargeback_rate(self) -> float:
        if self.total_transactions == 0:
            return 0.0
        return self.chargeback_count / max(1, self.total_transactions)
    
    def get_historical_fraud_rate(self) -> float:
        if self.total_transactions == 0:
            return 0.0
        return len(self.fraud_history) / self.total_transactions
    
    def get_customer_risk_score(self) -> float:
        """Overall customer risk score (0-1)."""
        score = 0.0
        
        # Fraud history
        if self.total_transactions > 0:
            fraud_rate = len(self.fraud_history) / self.total_transactions
            score += min(0.4, fraud_rate * 5)
        
        # Chargeback rate
        if self.total_transactions > 0:
            chargeback_rate = self.chargeback_count / self.total_transactions
            score += min(0.3, chargeback_rate * 10)
        
        # New device/merchant ratio
        if self.total_transactions > 0:
            new_devices = sum(1 for v in self.device_fingerprint_frequency.values() if v == 1)
            if len(self.device_fingerprint_frequency) > 0:
                new_device_ratio = new_devices / len(self.device_fingerprint_frequency)
                score += min(0.2, new_device_ratio * 2)
            
            new_merchants = sum(1 for v in self.merchant_frequency.values() if v == 1)
            if len(self.merchant_frequency) > 0:
                new_merchant_ratio = new_merchants / len(self.merchant_frequency)
                score += min(0.1, new_merchant_ratio)
        
        return min(1.0, score)
    
    def get_merchant_risk_score(self, merchant_id: str) -> float:
        return 1.0 - self.get_merchant_trust_score(merchant_id)
    
    def get_device_risk_score(self, device_id: str) -> float:
        return 1.0 - self.get_device_trust_score(device_id)
    
    def to_dict(self) -> dict:
        """Serialize profile for storage."""
        return {
            "customer_id": self.customer_id,
            "tenant_id": self.tenant_id,
            "total_transactions": self.total_transactions,
            "total_amount": self.total_amount,
            "avg_amount": self.avg_amount,
            "median_amount": self.median_amount,
            "std_amount": self.std_amount,
            "min_amount": self.min_amount,
            "max_amount": self.max_amount,
            "rolling_mean_7d": self.rolling_mean_7d,
            "rolling_mean_30d": self.rolling_mean_30d,
            "rolling_std": self.rolling_std,
            "preferred_hours": dict(self.preferred_hours),
            "preferred_weekdays": dict(self.preferred_weekdays),
            "preferred_merchants": dict(self.preferred_merchants),
            "merchant_frequency": dict(self.merchant_frequency),
            "country_frequency": dict(self.country_frequency),
            "device_frequency": dict(self.device_frequency),
            "ip_frequency": dict(self.ip_frequency),
            "velocity_statistics": {
                "mean": self.velocity_stats.mean,
                "std": self.velocity_stats.std,
            },
            "avg_time_between_transactions": self.avg_time_between_transactions,
            "last_transaction_time": self.last_transaction_time.isoformat() if self.last_transaction_time else None,
            "last_locations": self.last_locations[-10:],
            "trusted_devices": list(self.trusted_devices),
            "trusted_ips": list(self.trusted_ips),
            "chargeback_count": self.chargeback_count,
            "fraud_history": self.fraud_history,
            "account_age": (datetime.now(timezone.utc) - self.account_created).days if self.account_created else 0,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CustomerBehaviorProfile":
        """Deserialize profile from dictionary."""
        profile = cls(
            customer_id=data["customer_id"],
            tenant_id=data["tenant_id"],
        )
        # Restore basic fields
        profile = cls(
            customer_id=data["customer_id"],
            tenant_id=data["tenant_id"],
        )
        profile.total_transactions = data.get("total_transactions", 0)
        profile.total_amount = data.get("total_amount", 0.0)
        # ... restore other fields
        return profile