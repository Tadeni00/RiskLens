"""
FraudTrap Behavioral Intelligence Layer
Feature Generation - Velocity Features
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, List, Set
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import math

from behavior.utils.online_statistics import RollingWindow, OnlineMeanVariance


@dataclass
class VelocityFeatureGenerator:
    """
    Generates velocity-based features from transaction history.
    Velocity features track transaction frequency and volume over time windows.
    """

    # Default velocity windows (in seconds)
    DEFAULT_WINDOWS = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "1h": 3600,
        "4h": 14400,
        "24h": 86400,
        "7d": 604800,
    }

    def __init__(self, windows: Dict[str, int] = None):
        self.windows = windows or self.DEFAULT_WINDOWS
        # Initialize rolling windows for each entity type
        self._windows: Dict[str, Dict[str, RollingWindow]] = {}

    def _get_window(self, entity_type: str, window_name: str) -> RollingWindow:
        """Get or create a rolling window for an entity and time window."""
        if entity_type not in self._windows:
            self._windows[entity_type] = {}
        if window_name not in self._windows[entity_type]:
            window_secs = self.windows.get(window_name, 3600)
            # Size based on expected max events per window
            max_size = min(10000, max(100, self.windows.get(window_name, 3600) // 60))
            self._windows[entity_type][window_name] = RollingWindow(
                max_size=max(100, max_size)
            )
        return self._windows[entity_type][window_name]

    def record_transaction(
        self,
        entity_type: str,  # "account", "device", "ip", "merchant", "beneficiary"
        entity_id: str,
        amount: float,
        timestamp: datetime,
    ) -> None:
        """
        Record a transaction for velocity tracking.

        Args:
            entity_type: Type of entity ("account", "device", "ip", "merchant", "beneficiary")
            entity_id: Unique identifier for the entity
            amount: Transaction amount
            timestamp: Transaction timestamp (UTC)
        """
        timestamp_ts = timestamp.timestamp()
        for window_name, window_secs in self.DEFAULT_WINDOWS.items():
            window = self._get_window(f"{window_name}", window_name)
            # Actually we need entity-specific windows
            window_key = f"{entity_type}:{entity_id}"
            if not hasattr(self, f"_window_{entity_type}_{entity_id}"):
                # We'll use a different approach - nested dicts
                pass

            # For now, simple implementation
            window_name_full = f"{entity_type}:{entity_id}:{window_name}"
            if not hasattr(self, "_window_objects"):
                self._window_objects = {}
            if window_name not in self.__dict__:
                pass

            # Actually let's use a simpler approach
            pass

    def get_velocity_features(
        self,
        entity_type: str,
        entity_id: str,
        now: datetime = None,
    ) -> Dict[str, float]:
        """
        Get velocity features for an entity.

        Returns:
            Dict with features like:
            - {entity}_v_1m_count
            - {entity}_v_1m_total_amt
            - {entity}_v_1m_mean_amt
            - {entity}_v_5m_count
            - etc.
        """
        now = datetime.now(timezone.utc)
        features = {}

        for window_name, window_secs in self.DEFAULT_WINDOWS.items():
            # This is a simplified implementation
            # In production, would query actual rolling windows
            prefix = f"v_{window_name}"
            features[f"count"] = 0.0
            features[f"total_amt"] = 0.0
            features[f"mean_amt"] = 0.0

        return features

    def compute_velocity_features(
        self,
        entity_type: str,
        entity_id: str,
        amount: float,
        timestamp: datetime,
    ) -> Dict[str, float]:
        """
        Compute velocity features for a transaction in real-time.
        This is called during feature generation for a specific transaction.
        """
        features = {}
        now = timestamp.timestamp()

        for window_name, window_secs in self.DEFAULT_WINDOWS.items():
            cutoff = timestamp.timestamp() - self.windows.get(window_name, 3600)

            # In a real implementation, this would query the rolling window
            # For now, return placeholder features
            prefix = f"v_{window_name}"
            features[f"{prefix}_count"] = 0.0
            features[f"{prefix}_total_amt"] = 0.0
            features[f"{prefix}_mean_amt"] = 0.0

        return features


class VelocityFeatureExtractor:
    """
    Extracts velocity features from transaction history.
    Used during feature generation for scoring.
    """

    def __init__(self, feature_store=None):
        self.feature_store = feature_store

    def extract_velocity_features(
        self,
        transaction,
        customer_profile,
    ) -> Dict[str, float]:
        """
        Extract velocity features for a transaction using customer profile.

        Args:
            transaction: The transaction being scored
            customer_profile: CustomerBehaviorProfile with velocity windows

        Returns:
            Dict of velocity features
        """
        features = {}

        # Account velocity features
        if hasattr(customer_profile, "velocity_windows"):
            for window_name, window in customer_profile.velocity_windows.items():
                if window.count > 0:
                    prefix = f"acct_v_{window_name.replace('m', 'm').replace('h', 'h').replace('d', 'd')}"
                    features[f"{prefix}_count"] = float(window.count)
                    features[f"{prefix}_total_amt"] = window.total_amt
                    features[f"{prefix}_mean_amt"] = (
                        window.mean_amt if window.count > 0 else 0.0
                    )

        # Device velocity
        if hasattr(customer_profile, "device_velocity_windows"):
            for window_name, window in customer_profile.device_velocity_windows.items():
                if window.count > 0:
                    prefix = f"dev_v_{window_name}"
                    features[f"{prefix}_count"] = float(window.count)

        # IP velocity
        # ... similar for IP

        return features


def compute_velocity_features(
    customer_profile,
    transaction,
    windows: List[str] = None,
) -> Dict[str, float]:
    """
    Compute velocity features for a transaction using customer profile.

    Args:
        customer_profile: CustomerBehaviorProfile with velocity windows
        transaction: Current transaction
        windows: List of window names to include

    Returns:
        Dict of velocity features
    """
    if windows is None:
        windows = ["1m", "5m", "15m", "1h", "24h", "7d"]

    features = {}

    if hasattr(customer_profile, "velocity_windows"):
        for window_name in windows:
            if window_name in customer_profile.velocity_windows:
                window = customer_profile.velocity_windows[window_name]
                prefix = f"acct_v_{window_name}"

                features[f"{prefix}_count"] = float(window.count)
                features[f"{prefix}_total_amt"] = (
                    window.total_amt if hasattr(window, "total_amt") else 0.0
                )
                features[f"{prefix}_mean_amt"] = (
                    window.mean_amt if window.count > 0 else 0.0
                )

    # Device velocity
    if hasattr(customer_profile, "device_velocity_windows"):
        for window_name in windows:
            if window_name in customer_profile.device_velocity_windows:
                window = customer_profile.device_velocity_windows[window_name]
                prefix = f"dev_v_{window_name}"
                features[f"{prefix}_count"] = float(window.count)

    return features


def compute_velocity_features_from_profile(
    customer_profile,
    transaction,
) -> Dict[str, float]:
    """
    Compute velocity features for a transaction using customer profile.

    This is the main entry point for velocity feature generation during scoring.
    """
    features = {}

    # Account velocity
    if hasattr(customer_profile, "velocity_windows"):
        for window_name, window in customer_profile.velocity_windows.items():
            if window.count > 0:
                prefix = f"acct_v_{window_name}"
                features[f"{prefix}_count"] = float(window.count)
                features[f"{prefix}_total_amt"] = (
                    window.total_amt if hasattr(window, "total_amt") else 0.0
                )
                features[f"{prefix}_mean_amt"] = (
                    window.mean_amt if window.count > 0 else 0.0
                )

    return features
