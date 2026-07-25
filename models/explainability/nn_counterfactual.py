"""
RiskLens — Nearest Neighbor Counterfactual Engine
Retrieves the nearest legitimate transaction as a production counterfactual.
Uses FAISS for approximate nearest neighbor search with tenant isolation.
"""

from __future__ import annotations
import time
import pickle
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from loguru import logger

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not installed — using brute-force fallback")

from models.explainability.types import (
    CounterfactualChange,
    CounterfactualExplanation,
    NearestNeighbor,
)


class WeightedDistanceMetric:
    """
    Configurable weighted distance for feature comparison.
    Behavioral features get higher weights than static metadata.
    """

    def __init__(self, feature_weights: Optional[Dict[str, float]] = None):
        self.feature_weights = feature_weights or {}

    def compute(self, a: np.ndarray, b: np.ndarray, feature_names: List[str]) -> float:
        """Weighted Euclidean distance between two feature vectors."""
        weights = np.array(
            [self.feature_weights.get(fn, 1.0) for fn in feature_names],
            dtype=np.float32,
        )
        diff = a - b
        return float(np.sqrt(np.sum(weights * diff**2)))

    def set_weight(self, feature: str, weight: float) -> None:
        self.feature_weights[feature] = weight

    def get_weights(self) -> Dict[str, float]:
        return dict(self.feature_weights)


class NearestNeighborIndex:
    """
    Tenant-isolated approximate nearest neighbor index.

    Supports:
    - FAISS (preferred) or brute-force fallback
    - Incremental updates
    - Background rebuilding
    - Weighted distance metrics
    """

    def __init__(
        self,
        tenant_id: str,
        n_features: int,
        feature_names: List[str],
        distance_metric: Optional[WeightedDistanceMetric] = None,
        use_faiss: bool = True,
    ):
        self.tenant_id = tenant_id
        self.n_features = n_features
        self.feature_names = feature_names
        self.distance_metric = distance_metric or WeightedDistanceMetric()
        self.use_faiss = use_faiss and FAISS_AVAILABLE

        self._index = None
        self._data: List[np.ndarray] = []
        self._transaction_ids: List[str] = []
        self._labels: List[int] = []  # 0=legit, 1=fraud
        self._lock = threading.RLock()

        if self.use_faiss:
            self._init_faiss()

    def _init_faiss(self) -> None:
        """Initialize FAISS index."""
        try:
            # Use L2 (Euclidean) index — we apply weights in pre-processing
            self._index = faiss.IndexFlatL2(self.n_features)
            logger.info(
                "FAISS index initialized for tenant={} (dim={})",
                self.tenant_id,
                self.n_features,
            )
        except Exception as exc:
            logger.warning("FAISS init failed for tenant={}: {}", self.tenant_id, exc)
            self.use_faiss = False

    def fit(
        self, X: np.ndarray, transaction_ids: List[str], labels: np.ndarray
    ) -> None:
        """
        Build the index from historical legitimate transactions.

        Args:
            X: Feature matrix (n_samples, n_features)
            transaction_ids: Transaction IDs for each sample
            labels: Labels (0=legit, 1=fraud) — only legit are indexed
        """
        with self._lock:
            # Filter to legitimate transactions only
            legit_mask = labels == 0
            X_legit = X[legit_mask]
            ids_legit = [
                tid for tid, is_legit in zip(transaction_ids, legit_mask) if is_legit
            ]

            if len(X_legit) == 0:
                logger.warning(
                    "No legitimate transactions for tenant={}", self.tenant_id
                )
                return

            # Apply feature weights for weighted distance
            X_weighted = self._apply_weights(X_legit)

            self._data = list(X_weighted)
            self._transaction_ids = list(ids_legit)
            self._labels = [0] * len(ids_legit)

            if self.use_faiss:
                self._rebuild_faiss()

            logger.info(
                "NearestNeighborIndex built for tenant={}: {} legitimate transactions",
                self.tenant_id,
                len(ids_legit),
            )

    def add(
        self, X: np.ndarray, transaction_ids: List[str], labels: np.ndarray
    ) -> None:
        """Incrementally add transactions to the index."""
        with self._lock:
            for i in range(len(X)):
                tid = transaction_ids[i]
                label = int(labels[i])

                X_weighted = self._apply_weights(X[i : i + 1])
                self._data.append(X_weighted[0])
                self._transaction_ids.append(tid)
                self._labels.append(label)

                if self.use_faiss and label == 0:
                    self._index.add(X_weighted)

    def remove(self, transaction_ids: List[str]) -> int:
        """Remove transactions by ID. Returns count of removed items."""
        with self._lock:
            remove_set = set(transaction_ids)
            new_data = []
            new_ids = []
            new_labels = []

            removed = 0
            for i, tid in enumerate(self._transaction_ids):
                if tid in remove_set:
                    removed += 1
                else:
                    new_data.append(self._data[i])
                    new_ids.append(tid)
                    new_labels.append(self._labels[i])

            self._data = new_data
            self._transaction_ids = new_ids
            self._labels = new_labels

            if self.use_faiss and removed > 0:
                self._rebuild_faiss()

            return removed

    def query(self, X: np.ndarray, k: int = 1) -> List[List[NearestNeighbor]]:
        """
        Find k nearest neighbors for each query point.

        Args:
            X: Query points (n_queries, n_features)
            k: Number of neighbors to retrieve

        Returns:
            List of lists of NearestNeighbor results
        """
        with self._lock:
            if len(self._data) == 0:
                return [[] for _ in range(len(X))]

            X_weighted = self._apply_weights(X)
            results = []

            if self.use_faiss and self._index is not None:
                k = min(k, self._index.ntotal)
                if k == 0:
                    results = [[] for _ in range(len(X))]
                    return results

                distances, indices = self._index.search(X_weighted, k)

                for i in range(len(X)):
                    neighbors = []
                    for j in range(k):
                        idx = int(indices[i, j])
                        if idx < 0 or idx >= len(self._transaction_ids):
                            continue
                        # Skip fraud transactions
                        if self._labels[idx] == 1:
                            continue
                        neighbors.append(
                            NearestNeighbor(
                                transaction_id=self._transaction_ids[idx],
                                distance=float(distances[i, j]),
                                features=self._get_raw_features(idx),
                            )
                        )
                    results.append(neighbors[:k])
            else:
                # Brute-force fallback
                for i in range(len(X)):
                    distances = [
                        (
                            j,
                            self.distance_metric.compute(
                                X_weighted[i], self._data[j], self.feature_names
                            ),
                        )
                        for j in range(len(self._data))
                        if self._labels[j] == 0  # Only legitimate
                    ]
                    distances.sort(key=lambda x: x[1])

                    neighbors = [
                        NearestNeighbor(
                            transaction_id=self._transaction_ids[idx],
                            distance=dist,
                            features=self._get_raw_features(idx),
                        )
                        for idx, dist in distances[:k]
                    ]
                    results.append(neighbors)

            return results

    def _apply_weights(self, X: np.ndarray) -> np.ndarray:
        """Apply feature weights for distance computation."""
        weights = np.array(
            [
                self.distance_metric.feature_weights.get(fn, 1.0)
                for fn in self.feature_names
            ],
            dtype=np.float32,
        )
        return X * np.sqrt(weights)

    def _get_raw_features(self, idx: int) -> Dict[str, float]:
        """Get raw (unweighted) features for a stored transaction."""
        return {
            self.feature_names[i]: float(
                self._data[idx][i]
                / np.sqrt(
                    self.distance_metric.feature_weights.get(self.feature_names[i], 1.0)
                )
            )
            for i in range(min(len(self.feature_names), len(self._data[idx])))
        }

    def _rebuild_faiss(self) -> None:
        """Rebuild the FAISS index from scratch."""
        if not self.use_faiss:
            return

        self._index = faiss.IndexFlatL2(self.n_features)
        legit_data = np.array(
            [self._data[i] for i in range(len(self._data)) if self._labels[i] == 0],
            dtype=np.float32,
        )

        if len(legit_data) > 0:
            self._index.add(legit_data)

    def save(self, path: Path) -> None:
        """Persist the index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "nn_index.pkl", "wb") as f:
            pickle.dump(
                {
                    "tenant_id": self.tenant_id,
                    "n_features": self.n_features,
                    "feature_names": self.feature_names,
                    "distance_metric": self.distance_metric,
                    "data": self._data,
                    "transaction_ids": self._transaction_ids,
                    "labels": self._labels,
                    "use_faiss": self.use_faiss,
                },
                f,
            )

        # Save FAISS index separately if available
        if self.use_faiss and self._index is not None:
            faiss.write_index(self._index, str(path / "faiss.index"))

        logger.info("NearestNeighborIndex saved for tenant={}", self.tenant_id)

    @classmethod
    def load(cls, path: Path) -> "NearestNeighborIndex":
        """Load a persisted index."""
        path = Path(path)

        with open(path / "nn_index.pkl", "rb") as f:
            payload = pickle.load(f)

        obj = cls(
            tenant_id=payload["tenant_id"],
            n_features=payload["n_features"],
            feature_names=payload["feature_names"],
            distance_metric=payload["distance_metric"],
            use_faiss=payload["use_faiss"],
        )

        obj._data = payload["data"]
        obj._transaction_ids = payload["transaction_ids"]
        obj._labels = payload["labels"]

        # Load FAISS index if available
        faiss_path = path / "faiss.index"
        if obj.use_faiss and faiss_path.exists() and FAISS_AVAILABLE:
            try:
                obj._index = faiss.read_index(str(faiss_path))
            except Exception as exc:
                logger.warning("FAISS index load failed, rebuilding: {}", exc)
                obj._rebuild_faiss()
        else:
            obj._rebuild_faiss()

        logger.info("NearestNeighborIndex loaded for tenant={}", obj.tenant_id)
        return obj

    @property
    def size(self) -> int:
        return len(self._data)

    @property
    def legitimate_count(self) -> int:
        return sum(1 for l in self._labels if l == 0)


class NearestNeighborCounterfactual:
    """
    Production counterfactual engine using nearest legitimate neighbor.

    For each flagged transaction:
    1. Find the nearest legitimate transaction in the tenant's history
    2. Compute feature differences as counterfactual changes
    3. Return realistic, grounded-in-data explanations
    """

    def __init__(self, max_neighbors: int = 10):
        self.max_neighbors = max_neighbors
        self._indexes: Dict[str, NearestNeighborIndex] = {}

    def register_index(self, tenant_id: str, index: NearestNeighborIndex) -> None:
        """Register a pre-built ANN index for a tenant."""
        self._indexes[tenant_id] = index

    def get_index(self, tenant_id: str) -> Optional[NearestNeighborIndex]:
        return self._indexes.get(tenant_id)

    def explain(
        self,
        tenant_id: str,
        X: np.ndarray,
        transaction_id: str,
        fraud_probability: float,
        feature_names: List[str],
    ) -> Optional[CounterfactualExplanation]:
        """
        Generate counterfactual explanation using nearest legitimate neighbor.

        Args:
            tenant_id: Tenant ID for index lookup
            X: Feature vector (1, n_features) or (n_features,)
            transaction_id: Current transaction ID
            fraud_probability: Model's fraud probability
            feature_names: Ordered feature names

        Returns:
            CounterfactualExplanation or None if no index available
        """
        t_start = time.perf_counter()

        index = self._indexes.get(tenant_id)
        if index is None or index.legitimate_count == 0:
            return None

        X_query = X.reshape(1, -1) if X.ndim == 1 else X

        neighbors = index.query(
            X_query, k=min(self.max_neighbors, index.legitimate_count)
        )
        if not neighbors or not neighbors[0]:
            return None

        nearest = neighbors[0][0]
        x_current = X_query[0]

        changes = []
        for i, fname in enumerate(feature_names):
            current_val = float(x_current[i])
            nearest_val = nearest.features.get(fname, current_val)

            if abs(current_val - nearest_val) > 1e-6:
                changes.append(
                    CounterfactualChange(
                        feature=fname,
                        current_value=current_val,
                        counterfactual_value=nearest_val,
                        realistic=True,
                    )
                )

        # Sort by absolute difference (most impactful changes first)
        changes.sort(
            key=lambda c: abs(c.current_value - c.counterfactual_value), reverse=True
        )

        latency_ms = (time.perf_counter() - t_start) * 1000

        return CounterfactualExplanation(
            prediction_delta=round(fraud_probability - 0.1, 4),  # Approximate
            changes=tuple(changes[: self.max_neighbors]),
            source="nearest_neighbor",
            nearest_neighbor=nearest,
            latency_ms=round(latency_ms, 2),
        )

    def save(self, base_path: Path) -> None:
        """Save all tenant indexes."""
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)

        for tenant_id, index in self._indexes.items():
            tenant_path = base_path / tenant_id
            index.save(tenant_path)

        logger.info("All NearestNeighborIndex saved to {}", base_path)

    def load(self, base_path: Path) -> None:
        """Load all tenant indexes from disk."""
        base_path = Path(base_path)
        if not base_path.exists():
            return

        for tenant_dir in base_path.iterdir():
            if tenant_dir.is_dir():
                index_path = tenant_dir / "nn_index.pkl"
                if index_path.exists():
                    try:
                        index = NearestNeighborIndex.load(tenant_dir)
                        self._indexes[index.tenant_id] = index
                    except Exception as exc:
                        logger.warning(
                            "Failed to load NN index for {}: {}", tenant_dir.name, exc
                        )

        logger.info("Loaded {} tenant NN indexes", len(self._indexes))
