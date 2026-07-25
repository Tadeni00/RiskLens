"""
RiskLens Behavioral Intelligence Layer
Online Statistics Utilities
"""

from __future__ import annotations
import math
import random
import threading
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class OnlineMeanVariance:
    """
    Welford's online algorithm for mean and variance.
    Numerically stable single-pass algorithm.
    """

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0  # Sum of squared differences from the mean

    def update(self, value: float) -> None:
        """Update with a new value."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    def merge(self, other: "OnlineMeanVariance") -> None:
        """Merge another OnlineMeanVariance into this one (parallel algorithm)."""
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self.mean = other.mean
            self.m2 = other.m2
            return

        delta = other.mean - self.mean
        total_count = self.count + other.count
        self.mean = (self.count * self.mean + other.count * other.mean) / total_count
        self.m2 = (
            self.m2 + other.m2 + delta * delta * self.count * other.count / total_count
        )
        self.count = total_count
        self.m2 = (
            self.m2 + other.m2 + delta * delta * self.count * other.count / total_count
        )

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        return math.sqrt(max(0.0, self.variance))

    def get_zscore(self, value: float) -> float:
        """Get z-score for a value."""
        if self.count < 2:
            return 0.0
        std = self.std
        if std == 0:
            return 0.0
        return (value - self.mean) / self.std


@dataclass
class ExponentialMovingAverage:
    """
    Exponential Moving Average with configurable alpha.
    """

    alpha: float
    value: Optional[float] = None

    def update(self, value: float) -> float:
        if self.value is None:
            self.value = value
        else:
            self.value = self.alpha * value + (1 - self.alpha) * self.value
        return self.value

    def get(self) -> Optional[float]:
        return self.value


@dataclass
class RollingWindow:
    """
    Fixed-size rolling window with efficient statistics.
    """

    max_size: int
    values: list = field(default_factory=list)
    _sum: float = 0.0
    _sum_sq: float = 0.0

    def add(self, value: float) -> None:
        self.values.append(value)
        self._sum += value
        self._sum_sq += value * value
        if len(self.values) > self.max_size:
            removed = self.values.pop(0)
            self._sum -= removed
            self._sum_sq -= removed * removed

    def clear(self) -> None:
        self.values.clear()
        self._sum = 0.0
        self._sum_sq = 0.0

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def sum(self) -> float:
        return self._sum

    @property
    def mean(self) -> float:
        if not self.values:
            return 0.0
        return self._sum / len(self.values)

    @property
    def variance(self) -> float:
        n = len(self.values)
        if len(self.values) < 2:
            return 0.0
        mean = self.mean
        return (self._sum_sq / len(self.values) - mean * mean) * n / (n - 1)

    @property
    def mean(self) -> float:
        if not self.values:
            return 0.0
        return self._sum / len(self.values)

    @property
    def std(self) -> float:
        return math.sqrt(max(0.0, self.variance))

    @property
    def variance(self) -> float:
        n = len(self.values)
        if n < 2:
            return 0.0
        mean = self.mean
        return max(0.0, (self._sum_sq / len(self.values) - mean * mean) * n / (n - 1))

    def get_zscore(self, value: float) -> float:
        std = self.std
        if std == 0:
            return 0.0
        return (value - self.mean) / self.std


@dataclass
class CountMinSketch:
    """
    Count-Min Sketch for frequency estimation with bounded memory.
    Useful for high-cardinality frequency estimation.
    """

    width: int
    depth: int
    _table: list = field(default_factory=list)
    _hash_seeds: list = field(default_factory=list)

    def __post_init__(self):
        self._table = [[0] * self.width for _ in range(self.depth)]
        self._hash_seeds = [random.randint(1, 2**31 - 1) for _ in range(self.depth)]

    def _hash(self, item: str, seed: int) -> int:
        h = hash((item, seed))
        return abs(h) % self.width

    def add(self, item: str, count: int = 1) -> None:
        for i in range(self.depth):
            idx = self._hash(item, self._hash_seeds[i])
            self._table[i][idx] += count

    def estimate(self, item: str) -> int:
        return min(
            self._table[i][self._hash(item, self._hash_seeds[i])]
            for i in range(self.depth)
        )


class CircularBuffer:
    """
    Fixed-size circular buffer for time-series data.
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.data = [0.0] * max_size
        self.timestamps = [0.0] * max_size
        self.size = 0
        self.head = 0

    def append(self, value: float, timestamp: float) -> None:
        self.data[self.head] = value
        self.timestamps[self.head] = timestamp
        self.head = (self.head + 1) % self.max_size
        if self.size < self.max_size:
            self.size += 1

    def get_values(self) -> list:
        if self.size < self.max_size:
            return self.data[: self.size]
        return self.data[self.head :] + self.data[: self.head]

    def get_timestamps(self) -> list:
        if self.size < self.max_size:
            return self.timestamps[: self.size]
        return self.timestamps[self.head :] + self.timestamps[: self.head]


def percentile(values: list, p: float) -> float:
    """Calculate percentile from sorted values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(values) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


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


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in kilometers."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))
