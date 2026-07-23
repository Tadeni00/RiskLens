"""
FraudTrap — Explanation Cache
Caches explanations and SHAP values to reduce latency.
Supports TTL-based expiration and tenant isolation.
"""
from __future__ import annotations
import time
import hashlib
import threading
from typing import Optional, Dict, Any
from collections import OrderedDict
from loguru import logger

from models.explainability.types import FullExplanation, SHAPExplanation


class ExplanationCache:
    """
    In-memory LRU cache with TTL for explanation results.
    
    Tenant-isolated: all cache keys are prefixed with tenant hash.
    Thread-safe for concurrent scoring requests.
    
    Features:
    - LRU eviction when max_size is reached
    - TTL-based expiration
    - Hit/miss rate tracking
    - Tenant-scoped keys
    """
    
    def __init__(
        self,
        max_size: int = 10_000,
        ttl_seconds: int = 1800,  # 30 minutes default
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.RLock()
        
        # Metrics
        self._hits = 0
        self._misses = 0
    
    def _make_key(self, tenant_id: str, transaction_id: str) -> str:
        """Generate a tenant-scoped cache key."""
        tenant_hash = hashlib.sha256(tenant_id.encode()).hexdigest()[:12]
        return f"{tenant_hash}:{transaction_id}"
    
    def get(self, tenant_id: str, transaction_id: str) -> Optional[FullExplanation]:
        """Retrieve a cached explanation if available and not expired."""
        key = self._make_key(tenant_id, transaction_id)
        
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry, timestamp = self._cache[key]
            
            # Check TTL
            if time.monotonic() - timestamp > self.ttl_seconds:
                del self._cache[key]
                self._misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return entry
    
    def put(self, tenant_id: str, transaction_id: str, explanation: FullExplanation) -> None:
        """Store an explanation in the cache."""
        key = self._make_key(tenant_id, transaction_id)
        
        with self._lock:
            # Remove if exists (to update position)
            if key in self._cache:
                del self._cache[key]
            
            # Evict LRU if at capacity
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            
            self._cache[key] = (explanation, time.monotonic())
    
    def get_shap_cache(self, tenant_id: str) -> Dict[str, SHAPExplanation]:
        """
        Get all cached SHAP explanations for a tenant.
        Used for aggregation and drift monitoring.
        """
        # In production, this would be a separate cache layer
        # For now, return empty dict
        return {}
    
    def invalidate(self, tenant_id: str) -> int:
        """Invalidate all cache entries for a tenant. Returns count removed."""
        tenant_hash = hashlib.sha256(tenant_id.encode()).hexdigest()[:12]
        prefix = f"{tenant_hash}:"
        
        with self._lock:
            keys_to_remove = [k for k in self._cache if k.startswith(prefix)]
            for key in keys_to_remove:
                del self._cache[key]
        
        return len(keys_to_remove)
    
    def clear(self) -> None:
        """Clear the entire cache."""
        with self._lock:
            self._cache.clear()
    
    @property
    def size(self) -> int:
        return len(self._cache)
    
    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    @property
    def metrics(self) -> Dict[str, Any]:
        return {
            "size": self.size,
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
        }
    
    def reset_metrics(self) -> None:
        self._hits = 0
        self._misses = 0


class SHAPCache:
    """
    Cache for SHAP explainer objects to avoid re-initialization.
    One explainer per tenant (keyed by model version hash).
    """
    
    def __init__(self, max_explainers: int = 50):
        self.max_explainers = max_explainers
        self._cache: Dict[str, Any] = {}
        self._access_order: list[str] = []
        self._lock = threading.RLock()
    
    def _make_key(self, tenant_id: str, model_version: str) -> str:
        return f"{tenant_id}:{model_version}"
    
    def get(self, tenant_id: str, model_version: str):
        """Get a cached SHAP explainer."""
        key = self._make_key(tenant_id, model_version)
        
        with self._lock:
            if key in self._cache:
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
                return self._cache[key]
            return None
    
    def put(self, tenant_id: str, model_version: str, explainer) -> None:
        """Cache a SHAP explainer."""
        key = self._make_key(tenant_id, model_version)
        
        with self._lock:
            if key in self._cache:
                return  # Don't overwrite
            
            # Evict LRU if at capacity
            while len(self._cache) >= self.max_explainers and self._access_order:
                oldest = self._access_order.pop(0)
                del self._cache[oldest]
            
            self._cache[key] = explainer
            self._access_order.append(key)
    
    def invalidate(self, tenant_id: str) -> int:
        """Remove all explainers for a tenant."""
        prefix = f"{tenant_id}:"
        with self._lock:
            keys = [k for k in self._cache if k.startswith(prefix)]
            for k in keys:
                del self._cache[k]
                if k in self._access_order:
                    self._access_order.remove(k)
        return len(keys)
