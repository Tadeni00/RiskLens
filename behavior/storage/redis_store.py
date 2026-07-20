"""
FraudTrap Behavioral Intelligence Layer
Feature Store - Redis and In-Memory implementations
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any
from abc import ABC, abstractmethod
from abc import abstractmethod
from collections import defaultdict
import json

from behavior.profiles.customer import CustomerBehaviorProfile
from behavior.profiles.merchant import MerchantBehaviorProfile
from behavior.profiles.device import DeviceBehaviorProfile
from behavior.profiles.beneficiary import BeneficiaryBehaviorProfile
from behavior.profiles.payment_instrument import PaymentInstrumentProfile
from behavior.profiles.device import DeviceBehaviorProfile
from behavior.profiles.beneficiary import BeneficiaryBehaviorProfile
from behavior.profiles.payment_instrument import PaymentInstrumentProfile


class FeatureStoreClient(ABC):
    """Abstract base class for feature store implementations."""
    
    @abstractmethod
    def get_customer_profile(self, tenant_id: str, customer_id: str) -> Optional[object]:
        pass
    
    @abstractmethod
    def set_customer_profile(self, profile) -> None:
        pass
    
    @abstractmethod
    def get_merchant_profile(self, tenant_id: str, merchant_id: str):
        pass
    
    @abstractmethod
    def set_merchant_profile(self, profile) -> None:
        pass
    
    @abstractmethod
    def get_device_profile(self, tenant_id: str, device_id: str):
        pass
    
    @abstractmethod
    def set_device_profile(self, profile) -> None:
        pass
    
    @abstractmethod
    def get_beneficiary_profile(self, tenant_id: str, beneficiary_id: str):
        pass
    
    @abstractmethod
    def set_beneficiary_profile(self, profile) -> None:
        pass
    
    @abstractmethod
    def get_payment_instrument_profile(self, tenant_id: str, instrument_id: str):
        pass
    
    @abstractmethod
    def set_payment_instrument_profile(self, profile) -> None:
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        pass
    
    @abstractmethod
    def close(self) -> None:
        pass


@dataclass
class RedisFeatureStore:
    """
    Redis-backed feature store for production use.
    Uses Redis for hot storage with automatic TTL management.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: str = None,
        db: int = 0,
        tenant_hash_len: int = 12,
        default_ttl: int = 86400 * 30,  # 30 days
    ):
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self.tenant_hash_len = tenant_hash_len
        self.default_ttl = default_ttl
        self._client = None
        self._connected = False
    
    def _ensure_connected(self) -> bool:
        if self._connected and self._client:
            try:
                self._client.ping()
                return True
            except Exception:
                self._connected = False
        
        try:
            import redis
            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                password=self.password or None,
                db=self.db,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                max_connections=50,
            )
            self._client.ping()
            self._connected = True
            return True
        except Exception as e:
            print(f"Redis connection failed: {e}")
            self._connected = False
            return False
    
    def _key(self, tenant_id: str, entity_type: str, entity_id: str) -> str:
        """Generate tenant-isolated Redis key."""
        import hashlib
        tenant_hash = hashlib.sha256(tenant_id.encode()).hexdigest()[:12]
        return f"bt:{tenant_hash}:{entity_type}:{entity_id}"
    
    def _ensure_connected(self) -> bool:
        if not self._connected:
            return self._ensure_connected()
        return True
    
    # Customer Profile Methods
    def get_customer_profile(self, tenant_id: str, customer_id: str):
        if not self._ensure_connected():
            return None
        
        try:
            key = self._key(tenant_id, "customer", customer_id)
            data = self._client.get(key)
            if data:
                import json
                from behavior.profiles.customer import CustomerBehaviorProfile
                return CustomerBehaviorProfile.from_json(data)
        except Exception as e:
            print(f"Redis get_customer_profile error: {e}")
        return None
    
    def set_customer_profile(self, profile) -> bool:
        try:
            key = self._key(profile.tenant_id, "customer", profile.customer_id)
            import json
            self._client.setex(
                self._key(profile.tenant_id, "customer", profile.customer_id),
                86400 * 30,  # 30 days TTL
                profile.to_json()
            )
            return True
        except Exception as e:
            print(f"Redis set_customer_profile error: {e}")
            return False
    
    def get_merchant_profile(self, tenant_id: str, merchant_id: str):
        try:
            key = self._key(tenant_id, "merchant", merchant_id)
            data = self._client.get(key)
            if data:
                from behavior.profiles.merchant import MerchantBehaviorProfile
                return MerchantBehaviorProfile.from_json(data)
        except Exception:
            pass
        return None
    
    def set_merchant_profile(self, profile) -> bool:
        try:
            key = self._key(profile.tenant_id, "merchant", profile.merchant_id)
            self._client.setex(key, 86400 * 30, profile.to_json())
            return True
        except Exception as e:
            print(f"Redis set_merchant_profile error: {e}")
            return False
    
    def get_device_profile(self, tenant_id: str, device_id: str):
        try:
            key = self._key(tenant_id, "device", device_id)
            data = self._client.get(key)
            if data:
                from behavior.profiles.device import DeviceBehaviorProfile
                return DeviceBehaviorProfile.from_json(data)
        except Exception:
            pass
        return None
    
    def set_device_profile(self, profile) -> bool:
        try:
            key = self._key(profile.tenant_id, "device", profile.device_id)
            self._client.setex(key, 86400 * 30, profile.to_json())
            return True
        except Exception as e:
            print(f"Redis set_device_profile error: {e}")
            return False
    
    def get_beneficiary_profile(self, tenant_id: str, beneficiary_id: str):
        try:
            key = self._key(tenant_id, "beneficiary", beneficiary_id)
            data = self._client.get(key)
            if data:
                from behavior.profiles.beneficiary import BeneficiaryBehaviorProfile
                return BeneficiaryBehaviorProfile.from_json(data)
        except Exception:
            pass
        return None
    
    def set_beneficiary_profile(self, profile) -> bool:
        try:
            key = self._key(profile.tenant_id, "beneficiary", profile.beneficiary_id)
            self._client.setex(key, 86400 * 30, profile.to_json())
            return True
        except Exception as e:
            print(f"Redis set_beneficiary_profile error: {e}")
            return False
    
    def get_payment_instrument_profile(self, tenant_id: str, instrument_id: str):
        try:
            key = self._key(tenant_id, "instrument", instrument_id)
            data = self._client.get(key)
            if data:
                from behavior.profiles.payment_instrument import PaymentInstrumentProfile
                return PaymentInstrumentProfile.from_json(data)
        except Exception:
            pass
        return None
    
    def set_payment_instrument_profile(self, profile) -> bool:
        try:
            key = self._key(profile.tenant_id, "instrument", profile.instrument_id)
            self._client.setex(key, 86400 * 30, profile.to_json())
            return True
        except Exception as e:
            print(f"Redis set_payment_instrument_profile error: {e}")
            return False
    
    def health_check(self) -> bool:
        try:
            return self._client.ping()
        except Exception:
            return False
    
    def close(self):
        try:
            self._client.close()
        except Exception:
            pass


class InMemoryFeatureStore:
    """In-memory feature store for testing/local development without Redis."""
    
    def __init__(self):
        self._customer_profiles = {}
        self._merchant_profiles = {}
        self._device_profiles = {}
        self._beneficiary_profiles = {}
        self._instrument_profiles = {}
    
    def _make_key(self, *parts) -> str:
        return ":".join(str(p) for p in parts)
    
    # Customer
    def get_customer_profile(self, tenant_id: str, customer_id: str):
        key = self._make_key(tenant_id, "customer", customer_id)
        return self._customer_profiles.get(key)
    
    def set_customer_profile(self, profile):
        key = self._make_key(profile.tenant_id, "customer", profile.customer_id)
        self._customer_profiles[key] = profile
    
    def get_merchant_profile(self, tenant_id: str, merchant_id: str):
        key = self._make_key(tenant_id, "merchant", merchant_id)
        return self._merchant_profiles.get(key)
    
    def set_merchant_profile(self, profile):
        key = self._make_key(profile.tenant_id, "merchant", profile.merchant_id)
        self._merchant_profiles[key] = profile
    
    def get_device_profile(self, tenant_id: str, device_id: str):
        key = self._make_key(tenant_id, "device", device_id)
        return self._device_profiles.get(key)
    
    def set_device_profile(self, profile):
        key = self._make_key(profile.tenant_id, "device", profile.device_id)
        self._device_profiles[key] = profile
    
    def get_beneficiary_profile(self, tenant_id: str, beneficiary_id: str):
        key = self._make_key(tenant_id, "beneficiary", beneficiary_id)
        return self._beneficiary_profiles.get(key)
    
    def set_beneficiary_profile(self, profile):
        key = self._make_key(profile.tenant_id, "beneficiary", profile.beneficiary_id)
        self._beneficiary_profiles[key] = profile
    
    def get_payment_instrument_profile(self, tenant_id: str, instrument_id: str):
        key = self._make_key(tenant_id, "instrument", instrument_id)
        return self._instrument_profiles.get(key)
    
    def set_payment_instrument_profile(self, profile):
        key = self._make_key(profile.tenant_id, "instrument", profile.instrument_id)
        self._instrument_profiles[key] = profile
    
    def health_check(self) -> bool:
        return True
    
    def close(self):
        pass


class MockFeatureStore:
    """Mock feature store for testing/graceful degradation."""
    
    def get_customer_profile(self, tenant_id: str, customer_id: str):
        return None
    
    def set_customer_profile(self, profile) -> None:
        pass
    
    def get_merchant_profile(self, tenant_id: str, merchant_id: str):
        return None
    
    def set_merchant_profile(self, profile) -> None:
        pass
    
    def get_device_profile(self, tenant_id: str, device_id: str):
        return None
    
    def set_device_profile(self, profile) -> None:
        pass
    
    def get_beneficiary_profile(self, tenant_id: str, beneficiary_id: str):
        return None
    
    def set_beneficiary_profile(self, profile) -> None:
        pass
    
    def get_payment_instrument_profile(self, tenant_id: str, instrument_id: str):
        return None
    
    def set_payment_instrument_profile(self, profile) -> None:
        pass
    
    def health_check(self) -> bool:
        return True
    
    def close(self):
        pass


def get_feature_store() -> "FeatureStoreClient":
    """Factory function to get appropriate feature store instance."""
    import os
    
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            store = RedisFeatureStore.from_url(redis_url)
            if store.health_check():
                return store
        except Exception as e:
            print(f"Redis unavailable, falling back to in-memory: {e}")
    
    # Fallback to in-memory
    return InMemoryFeatureStore()


# For backward compatibility
FeatureStoreClient = None  # Will be set to appropriate class