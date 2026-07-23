"""
FraudTrap — Feature Store Client (Abstraction Layer)
Provides a clean interface for feature operations, decoupled from Redis.
Supports multiple backends (Redis, In-Memory, Mock) for testing and deployment.
"""

from __future__ import annotations
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
import redis
import numpy as np

from loguru import logger


@dataclass
class VelocityFeatures:
    """Velocity features for an entity."""

    count_1m: int = 0
    count_5m: int = 0
    count_1h: int = 0
    count_24h: int = 0
    count_7d: int = 0
    total_amt_1m: float = 0.0
    total_amt_5m: float = 0.0
    total_amt_1h: float = 0.0
    total_amt_24h: float = 0.0
    total_amt_7d: float = 0.0
    mean_amt_1m: float = 0.0
    mean_amt_5m: float = 0.0
    mean_amt_1h: float = 0.0
    mean_amt_24h: float = 0.0
    mean_amt_7d: float = 0.0


@dataclass
class SeenEntities:
    """Previously seen entities for new-entity detection."""

    merchants: set[str]
    devices: set[str]
    ips: set[str]


class FeatureStoreClient(ABC):
    """
    Abstract base class for feature store operations.

    Implementations:
    - RedisFeatureStore: Production Redis backend
    - InMemoryFeatureStore: Testing/local development
    - MockFeatureStore: Graceful degradation (returns zeros)
    """

    @abstractmethod
    def get_velocity_features(
        self,
        entity_type: str,
        entity_id: str,
        tenant_id: str,
        windows_seconds: list[int] = None,
    ) -> VelocityFeatures:
        """Get velocity features for an entity."""
        pass

    @abstractmethod
    def increment_velocity(
        self,
        entity_type: str,
        entity_id: str,
        tenant_id: str,
        amount: float,
        transaction_id: str,
        timestamp: float = None,
    ) -> None:
        """Record a new transaction for velocity tracking."""
        pass

    @abstractmethod
    def get_seen_entities(
        self,
        tenant_id: str,
        entity_type: str,  # "merchant", "device", "ip"
    ) -> set[str]:
        """Get set of seen entities for new-entity detection."""
        pass

    @abstractmethod
    def add_seen_entity(
        self,
        tenant_id: str,
        entity_type: str,
        entity_id: str,
    ) -> bool:
        """Mark entity as seen. Returns True if new, False if already seen."""
        pass

    @abstractmethod
    def get_hist_stats(self, tenant_id: str, account_id: str) -> dict:
        """Get historical amount statistics (mean, std)."""
        pass

    @abstractmethod
    def update_hist_stats(self, tenant_id: str, account_id: str, amount: float) -> None:
        """Update historical statistics with new amount."""
        pass

    @abstractmethod
    def get_last_location(self, tenant_id: str, account_id: str) -> Optional[dict]:
        """Get last known location (lat, lon, timestamp)."""
        pass

    @abstractmethod
    def update_last_location(
        self,
        tenant_id: str,
        account_id: str,
        lat: float,
        lon: float,
        timestamp: float = None,
    ) -> None:
        """Update last known location."""
        pass

    @abstractmethod
    def get_home_country(self, tenant_id: str, account_id: str) -> Optional[str]:
        """Get account's home country."""
        pass

    @abstractmethod
    def set_home_country(
        self, tenant_id: str, account_id: str, country_code: str
    ) -> None:
        """Set account's home country."""
        pass

    @abstractmethod
    def get_typing_baseline(self, tenant_id: str, account_id: str) -> Optional[dict]:
        """Get typing cadence baseline (mean, std, n)."""
        pass

    @abstractmethod
    def update_typing_baseline(
        self,
        tenant_id: str,
        account_id: str,
        cadence_ms: float,
    ) -> None:
        """Update typing cadence baseline."""
        pass

    @abstractmethod
    def get_device_accounts(self, tenant_id: str, device_id: str) -> set[str]:
        """Get accounts associated with a device."""
        pass

    @abstractmethod
    def add_device_account(
        self, tenant_id: str, device_id: str, account_id: str
    ) -> None:
        """Record device-account association."""
        pass

    @abstractmethod
    def get_blocklist(self, list_name: str, tenant_id: str) -> set[str]:
        """Get blocklist entries."""
        pass

    @abstractmethod
    def add_to_blocklist(self, list_name: str, value: str, tenant_id: str) -> None:
        """Add to blocklist."""
        pass

    @abstractmethod
    def remove_from_blocklist(self, list_name: str, value: str, tenant_id: str) -> None:
        """Remove from blocklist."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if store is healthy."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
        pass


class RedisFeatureStore(FeatureStoreClient):
    """Redis-backed feature store implementation."""

    WINDOWS = {
        "1m": 60,
        "5m": 300,
        "1h": 3600,
        "24h": 86400,
        "7d": 604800,
    }

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: str = "",
        db: int = 0,
        tenant_hash_len: int = 12,
        socket_timeout: float = 0.050,
        socket_connect_timeout: float = 1.0,
    ):
        self.tenant_hash_len = tenant_hash_len
        self._client = redis.Redis(
            host=host,
            port=port,
            password=password or None,
            db=db,
            decode_responses=True,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
        )
        self._connected = False

    def _tenant_hash(self, tenant_id: str) -> str:
        import hashlib

        return hashlib.sha256(tenant_id.encode()).hexdigest()[: self.tenant_hash_len]

    def _key(self, tenant_id: str, *parts: str) -> str:
        """Build namespaced key: ft:{tenant_hash}:{parts...}"""
        th = self._tenant_hash(tenant_id)
        return f"ft:{th}:" + ":".join(parts)

    def _ensure_connected(self) -> bool:
        if not self._connected:
            try:
                self._client.ping()
                self._connected = True
            except Exception:
                return False
        return True

    def get_velocity_features(
        self,
        entity_type: str,
        entity_id: str,
        tenant_id: str,
        windows_seconds: list[int] = None,
    ) -> VelocityFeatures:
        if not self._ensure_connected():
            return VelocityFeatures()

        windows = windows_seconds or [60, 300, 3600, 86400, 604800]
        now_ts = time.time()
        features = VelocityFeatures()

        try:
            pipe = self._client.pipeline(transaction=False)

            ts_key = self._key(tenant_id, entity_type, entity_id, "txn_ts")
            amt_key = self._key(tenant_id, entity_type, entity_id, "txn_amt")

            for window_sec in windows:
                cutoff = now_ts - window_sec
                pipe.zcount(ts_key, cutoff, "+inf")
                pipe.zrangebyscore(amt_key, cutoff, "+inf")

            results = pipe.execute()

            for i, window_sec in enumerate(windows):
                count = results[i * 2] or 0
                amounts = [float(a) for a in (results[i * 2 + 1] or [])]
                total_amt = sum(amounts)
                mean_amt = total_amt / count if count > 0 else 0.0

                # Map window to attribute
                if window_sec == 60:
                    features.count_1m, features.total_amt_1m, features.mean_amt_1m = (
                        count,
                        total_amt,
                        mean_amt,
                    )
                elif window_sec == 300:
                    features.count_5m, features.total_amt_5m, features.mean_amt_5m = (
                        count,
                        total_amt,
                        mean_amt,
                    )
                elif window_sec == 3600:
                    features.count_1h, features.total_amt_1h, features.mean_amt_1h = (
                        count,
                        total_amt,
                        mean_amt,
                    )
                elif window_sec == 86400:
                    (
                        features.count_24h,
                        features.total_amt_24h,
                        features.mean_amt_24h,
                    ) = (count, total_amt, mean_amt)
                elif window_sec == 604800:
                    features.count_7d, features.total_amt_7d, features.mean_amt_7d = (
                        count,
                        total_amt,
                        mean_amt,
                    )

        except Exception as exc:
            logger.warning("Velocity feature fetch failed: {}", exc)

        return features

    def increment_velocity(
        self,
        entity_type: str,
        entity_id: str,
        tenant_id: str,
        amount: float,
        transaction_id: str,
        timestamp: float = None,
    ) -> None:
        if not self._ensure_connected():
            return

        now_ts = timestamp or time.time()

        try:
            pipe = self._client.pipeline(transaction=False)

            ts_key = self._key(tenant_id, entity_type, entity_id, "txn_ts")
            amt_key = self._key(tenant_id, entity_type, entity_id, "txn_amt")

            pipe.zadd(ts_key, {transaction_id: now_ts})
            pipe.zadd(amt_key, {transaction_id: amount})

            # Prune old entries (keep 7 days + 1 hour buffer)
            prune_cutoff = time.time() - 604800 - 3600
            pipe.zremrangebyscore(ts_key, "-inf", prune_cutoff)
            pipe.zremrangebyscore(amt_key, "-inf", prune_cutoff)

            # TTL
            pipe.expire(ts_key, 604800 + 3600)
            pipe.expire(amt_key, 604800 + 3600)

            pipe.execute()

        except Exception as exc:
            logger.warning("Velocity increment failed: {}", exc)

    def get_seen_entities(self, tenant_id: str, entity_type: str) -> set[str]:
        if not self._ensure_connected():
            return set()

        try:
            key = self._key(tenant_id, "acct", "seen", entity_type)
            return set(self._client.smembers(key))
        except Exception:
            return set()

    def add_seen_entity(self, tenant_id: str, entity_type: str, entity_id: str) -> bool:
        if not self._ensure_connected():
            return False

        try:
            key = self._key(tenant_id, "acct", "seen", entity_type)
            added = self._client.sadd(key, entity_id)
            if added:
                self._client.expire(key, 604800)
            return bool(added)
        except Exception:
            return False

    def get_hist_stats(self, tenant_id: str, account_id: str) -> dict:
        if not self._ensure_connected():
            return {}

        try:
            key = self._key(tenant_id, "acct", account_id, "hist_stats")
            data = self._client.hgetall(key)
            return {
                "mean_amt": float(data.get("mean_amt", 0)),
                "std_amt": float(data.get("std_amt", 1)),
                "count": int(data.get("count", 0)),
            }
        except Exception:
            return {}

    def update_hist_stats(self, tenant_id: str, account_id: str, amount: float) -> None:
        if not self._ensure_connected():
            return

        try:
            key = self._key(tenant_id, "acct", account_id, "hist_stats")
            stats = self._client.hgetall(key)

            count = int(stats.get("count", 0)) + 1
            old_mean = float(stats.get("mean_amt", amount))
            old_std = float(stats.get("std_amt", 1.0))

            # Welford's online algorithm
            new_mean = old_mean + (amount - old_mean) / count
            new_std = (
                np.sqrt(
                    (
                        (count - 1) * old_std**2
                        + (amount - old_mean) * (amount - new_mean)
                    )
                    / count
                )
                if count > 1
                else 1.0
            )

            pipe = self._client.pipeline(transaction=False)
            pipe.hset(
                key,
                mapping={
                    "mean_amt": new_mean,
                    "std_amt": new_std,
                    "count": count,
                },
            )
            pipe.expire(key, 2592000)  # 30 days
            pipe.execute()

        except Exception as exc:
            logger.warning("Hist stats update failed: {}", exc)

    def get_last_location(self, tenant_id: str, account_id: str) -> Optional[dict]:
        if not self._ensure_connected():
            return None

        try:
            key = self._key(tenant_id, "acct", account_id, "last_loc")
            data = self._client.hgetall(key)
            if data:
                return {
                    "lat": float(data["lat"]),
                    "lon": float(data["lon"]),
                    "ts": float(data["ts"]),
                }
            return None
        except Exception:
            return None

    def update_last_location(
        self,
        tenant_id: str,
        account_id: str,
        lat: float,
        lon: float,
        timestamp: float = None,
    ) -> None:
        if not self._ensure_connected():
            return

        try:
            key = self._key(tenant_id, "acct", account_id, "last_loc")
            self._client.hset(
                key,
                mapping={
                    "lat": lat,
                    "lon": lon,
                    "ts": timestamp or time.time(),
                },
            )
            self._client.expire(key, 604800)
        except Exception as exc:
            logger.warning("Location update failed: {}", exc)

    def get_home_country(self, tenant_id: str, account_id: str) -> Optional[str]:
        if not self._ensure_connected():
            return None

        try:
            key = self._key(tenant_id, "acct", account_id, "home_country")
            return self._client.get(key)
        except Exception:
            return None

    def set_home_country(
        self, tenant_id: str, account_id: str, country_code: str
    ) -> None:
        if not self._ensure_connected():
            return

        try:
            key = self._key(tenant_id, "acct", account_id, "home_country")
            self._client.set(key, country_code, ex=2592000)  # 30 days
        except Exception as exc:
            logger.warning("Home country set failed: {}", exc)

    def get_typing_baseline(self, tenant_id: str, account_id: str) -> Optional[dict]:
        if not self._ensure_connected():
            return None

        try:
            key = self._key(tenant_id, "acct", account_id, "typing_baseline")
            data = self._client.hgetall(key)
            if data:
                return {
                    "mean": float(data["mean"]),
                    "std": float(data["std"]),
                    "n": int(data["n"]),
                }
            return None
        except Exception:
            return None

    def update_typing_baseline(
        self,
        tenant_id: str,
        account_id: str,
        cadence_ms: float,
    ) -> None:
        if not self._ensure_connected():
            return

        try:
            key = self._key(tenant_id, "acct", account_id, "typing_baseline")
            data = self._client.hgetall(key)

            n = int(data.get("n", 0)) + 1
            old_mean = float(data.get("mean", cadence_ms))
            old_std = float(data.get("std", 50.0))

            new_mean = old_mean + (cadence_ms - old_mean) / n
            new_std = (
                np.sqrt(
                    (
                        (n - 1) * old_std**2
                        + (cadence_ms - old_mean) * (cadence_ms - new_mean)
                    )
                    / n
                )
                if n > 1
                else 50.0
            )

            pipe = self._client.pipeline(transaction=False)
            pipe.hset(
                key,
                mapping={
                    "mean": new_mean,
                    "std": new_std,
                    "n": n,
                },
            )
            pipe.expire(key, 2592000)
            pipe.execute()

        except Exception as exc:
            logger.warning("Typing baseline update failed: {}", exc)

    def get_device_accounts(self, tenant_id: str, device_id: str) -> set[str]:
        if not self._ensure_connected():
            return set()

        try:
            key = self._key(tenant_id, "dev", device_id, "accounts")
            return set(self._client.smembers(key))
        except Exception:
            return set()

    def add_device_account(
        self, tenant_id: str, device_id: str, account_id: str
    ) -> None:
        if not self._ensure_connected():
            return

        try:
            key = self._key(tenant_id, "dev", device_id, "accounts")
            self._client.sadd(key, account_id)
            self._client.expire(key, 604800)
        except Exception as exc:
            logger.warning("Device account add failed: {}", exc)

    def get_blocklist(self, list_name: str, tenant_id: str) -> set[str]:
        if not self._ensure_connected():
            return set()

        try:
            key = f"ft:blocklist:{list_name}:{tenant_id}"
            return set(self._client.smembers(key))
        except Exception:
            return set()

    def add_to_blocklist(self, list_name: str, value: str, tenant_id: str) -> None:
        if not self._ensure_connected():
            return

        try:
            key = f"ft:blocklist:{list_name}:{tenant_id}"
            self._client.sadd(key, value)
        except Exception as exc:
            logger.warning("Blocklist add failed: {}", exc)

    def remove_from_blocklist(self, list_name: str, value: str, tenant_id: str) -> None:
        if not self._ensure_connected():
            return

        try:
            key = f"ft:blocklist:{list_name}:{tenant_id}"
            self._client.srem(key, value)
        except Exception as exc:
            logger.warning("Blocklist remove failed: {}", exc)

    def health_check(self) -> bool:
        try:
            return self._client.ping()
        except Exception:
            return False

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


class InMemoryFeatureStore(FeatureStoreClient):
    """In-memory feature store for testing and local development."""

    def __init__(self):
        self._data: dict = {}

    def _make_key(self, *parts) -> str:
        return ":".join(str(p) for p in parts)

    # Implement all abstract methods with in-memory dicts
    # (Full implementation omitted for brevity - similar to Redis but using dict)

    def get_velocity_features(
        self,
        entity_type: str,
        entity_id: str,
        tenant_id: str,
        windows_seconds: list[int] = None,
    ) -> VelocityFeatures:
        return VelocityFeatures()

    def increment_velocity(
        self,
        entity_type: str,
        entity_id: str,
        tenant_id: str,
        amount: float,
        transaction_id: str,
        timestamp: float = None,
    ) -> None:
        pass

    def get_seen_entities(self, tenant_id: str, entity_type: str) -> set[str]:
        return set()

    def add_seen_entity(self, tenant_id: str, entity_type: str, entity_id: str) -> bool:
        return True

    def get_hist_stats(self, tenant_id: str, account_id: str) -> dict:
        return {}

    def update_hist_stats(self, tenant_id: str, account_id: str, amount: float) -> None:
        pass

    def get_last_location(self, tenant_id: str, account_id: str) -> Optional[dict]:
        return None

    def update_last_location(
        self,
        tenant_id: str,
        account_id: str,
        lat: float,
        lon: float,
        timestamp: float = None,
    ) -> None:
        pass

    def get_home_country(self, tenant_id: str, account_id: str) -> Optional[str]:
        return None

    def set_home_country(
        self, tenant_id: str, account_id: str, country_code: str
    ) -> None:
        pass

    def get_typing_baseline(self, tenant_id: str, account_id: str) -> Optional[dict]:
        return None

    def update_typing_baseline(
        self, tenant_id: str, account_id: str, cadence_ms: float
    ) -> None:
        pass

    def get_device_accounts(self, tenant_id: str, device_id: str) -> set[str]:
        return set()

    def add_device_account(
        self, tenant_id: str, device_id: str, account_id: str
    ) -> None:
        pass

    def get_blocklist(self, list_name: str, tenant_id: str) -> set[str]:
        return set()

    def add_to_blocklist(self, list_name: str, value: str, tenant_id: str) -> None:
        pass

    def remove_from_blocklist(self, list_name: str, value: str, tenant_id: str) -> None:
        pass

    def health_check(self) -> bool:
        return True

    def close(self) -> None:
        pass


class MockFeatureStore(FeatureStoreClient):
    """Mock feature store for graceful degradation - returns zeros/empty."""

    def get_velocity_features(
        self,
        entity_type: str,
        entity_id: str,
        tenant_id: str,
        windows_seconds: list[int] = None,
    ) -> VelocityFeatures:
        return VelocityFeatures()

    def increment_velocity(
        self,
        entity_type: str,
        entity_id: str,
        tenant_id: str,
        amount: float,
        transaction_id: str,
        timestamp: float = None,
    ) -> None:
        pass

    def get_seen_entities(self, tenant_id: str, entity_type: str) -> set[str]:
        return set()

    def add_seen_entity(self, tenant_id: str, entity_type: str, entity_id: str) -> bool:
        return False

    def get_hist_stats(self, tenant_id: str, account_id: str) -> dict:
        return {}

    def update_hist_stats(self, tenant_id: str, account_id: str, amount: float) -> None:
        pass

    def get_last_location(self, tenant_id: str, account_id: str) -> Optional[dict]:
        return None

    def update_last_location(
        self,
        tenant_id: str,
        account_id: str,
        lat: float,
        lon: float,
        timestamp: float = None,
    ) -> None:
        pass

    def get_home_country(self, tenant_id: str, account_id: str) -> Optional[str]:
        return None

    def set_home_country(
        self, tenant_id: str, account_id: str, country_code: str
    ) -> None:
        pass

    def get_typing_baseline(self, tenant_id: str, account_id: str) -> Optional[dict]:
        return None

    def update_typing_baseline(
        self, tenant_id: str, account_id: str, cadence_ms: float
    ) -> None:
        pass

    def get_device_accounts(self, tenant_id: str, device_id: str) -> set[str]:
        return set()

    def add_device_account(
        self, tenant_id: str, device_id: str, account_id: str
    ) -> None:
        pass

    def get_blocklist(self, list_name: str, tenant_id: str) -> set[str]:
        return set()

    def add_to_blocklist(self, list_name: str, value: str, tenant_id: str) -> None:
        pass

    def remove_from_blocklist(self, list_name: str, value: str, tenant_id: str) -> None:
        pass

    def health_check(self) -> bool:
        return False

    def close(self) -> None:
        pass


def get_feature_store() -> FeatureStoreClient:
    """Factory function to get appropriate feature store based on environment."""
    from config.settings import get_settings

    settings = get_settings()

    # Check if Redis is configured and reachable
    try:
        import redis

        r = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password or None,
            db=settings.redis_db,
            socket_timeout=1.0,
            socket_connect_timeout=1.0,
        )
        r.ping()
        return RedisFeatureStore(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password or "",
            db=settings.redis_db,
        )
    except Exception as exc:
        logger.warning("Redis unavailable, using MockFeatureStore: {}", exc)
        return MockFeatureStore()
