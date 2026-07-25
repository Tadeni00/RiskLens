"""
RiskLens — Explainability Framework Tests
Unit, integration, and performance tests for the explainability layer.
"""

from __future__ import annotations
import time
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.explainability.types import (
    FeatureAttribution,
    SHAPExplanation,
    CounterfactualChange,
    NearestNeighbor,
    CounterfactualExplanation,
    ConfidenceInfo,
    FormattedReport,
    FullExplanation,
)
from models.explainability.shap_explainer import SHAPExplainer
from models.explainability.nn_counterfactual import (
    NearestNeighborCounterfactual,
    NearestNeighborIndex,
    WeightedDistanceMetric,
)
from models.explainability.formatter import ExplanationFormatter
from models.explainability.cache import ExplanationCache, SHAPCache
from models.explainability.monitoring import ExplainabilityMonitor
from models.explainability.engine import ExplainabilityEngine, ExplainabilityConfig

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_data():
    """Generate synthetic data for testing."""
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    feature_names = [f"feature_{i}" for i in range(n_features)]

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = (np.mean(X[:, :3], axis=1) > 0).astype(int)

    return X, y, feature_names


@pytest.fixture
def sample_attribution():
    return FeatureAttribution(
        feature="transaction_velocity",
        value=8.0,
        impact=0.29,
        direction="increase",
        method="shap",
    )


@pytest.fixture
def sample_shap_explanation(sample_attribution):
    return SHAPExplanation(
        fraud_probability=0.94,
        base_value=0.1,
        top_features=(sample_attribution,),
        latency_ms=5.0,
    )


@pytest.fixture
def sample_counterfactual():
    return CounterfactualExplanation(
        prediction_delta=0.78,
        changes=(
            CounterfactualChange(
                feature="amount",
                current_value=850000,
                counterfactual_value=180000,
                realistic=True,
            ),
            CounterfactualChange(
                feature="device_age",
                current_value=0,
                counterfactual_value=30,
                realistic=True,
            ),
        ),
        source="nearest_neighbor",
        nearest_neighbor=NearestNeighbor(
            transaction_id="legit_txn_001",
            distance=2.5,
            features={"amount": 180000, "device_age": 30},
        ),
        latency_ms=3.0,
    )


@pytest.fixture
def sample_confidence():
    return ConfidenceInfo(
        expert_used="CatBoost",
        confidence=0.92,
        ft_invoked=False,
    )


# ── Type tests ────────────────────────────────────────────────────────────────


class TestTypes:
    def test_feature_attribution(self, sample_attribution):
        assert sample_attribution.feature == "transaction_velocity"
        assert sample_attribution.direction == "increase"

    def test_shap_explanation(self, sample_shap_explanation):
        assert sample_shap_explanation.fraud_probability == 0.94
        d = sample_shap_explanation.to_dict()
        assert d["fraud_probability"] == 0.94
        assert len(d["top_features"]) == 1

    def test_counterfactual_explanation(self, sample_counterfactual):
        assert len(sample_counterfactual.changes) == 2
        assert sample_counterfactual.source == "nearest_neighbor"
        d = sample_counterfactual.to_dict()
        assert len(d["changes"]) == 2

    def test_confidence_info(self, sample_confidence):
        assert sample_confidence.expert_used == "CatBoost"
        assert sample_confidence.ft_invoked is False

    def test_full_explanation(
        self, sample_shap_explanation, sample_counterfactual, sample_confidence
    ):
        full = FullExplanation(
            transaction_id="txn_001",
            tenant_id="tenant_001",
            fraud_probability=0.94,
            shap=sample_shap_explanation,
            counterfactual=sample_counterfactual,
            confidence=sample_confidence,
            total_latency_ms=15.0,
        )
        d = full.to_dict()
        assert d["transaction_id"] == "txn_001"
        assert d["shap"]["fraud_probability"] == 0.94
        assert d["counterfactual"]["source"] == "nearest_neighbor"


# ── SHAP Explainer tests ─────────────────────────────────────────────────────


class TestSHAPExplainer:
    def test_init(self):
        explainer = SHAPExplainer(top_features=5)
        assert explainer.top_features == 5
        assert not explainer._is_fitted

    def test_fallback_when_not_fitted(self, synthetic_data):
        X, _, feature_names = synthetic_data
        explainer = SHAPExplainer(top_features=3)
        result = explainer.explain(X[:1])
        assert result.fraud_probability == 0.5
        assert result.latency_ms >= 0


# ── Nearest Neighbor Counterfactual tests ─────────────────────────────────────


class TestNearestNeighborCounterfactual:
    def test_weighted_distance(self):
        metric = WeightedDistanceMetric({"f1": 2.0, "f2": 1.0})
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        dist = metric.compute(a, b, ["f1", "f2"])
        # weights * diff^2: 2.0 * 1 + 1.0 * 1 = 3.0, sqrt(3.0) ≈ 1.732
        assert dist == pytest.approx(np.sqrt(3.0), rel=1e-3)

    def test_nn_index_fit_and_query(self, synthetic_data):
        X, y, feature_names = synthetic_data
        index = NearestNeighborIndex(
            tenant_id="test",
            n_features=len(feature_names),
            feature_names=feature_names,
        )
        tids = [f"txn_{i}" for i in range(len(X))]
        index.fit(X, tids, y)

        assert index.size > 0
        assert index.legitimate_count > 0

        neighbors = index.query(X[:1], k=3)
        assert len(neighbors) == 1
        assert len(neighbors[0]) <= 3

    def test_nn_counterfactual_explain(self, synthetic_data):
        X, y, feature_names = synthetic_data
        index = NearestNeighborIndex(
            tenant_id="test",
            n_features=len(feature_names),
            feature_names=feature_names,
        )
        tids = [f"txn_{i}" for i in range(len(X))]
        index.fit(X, tids, y)

        engine = NearestNeighborCounterfactual(max_neighbors=5)
        engine.register_index("test", index)

        cf = engine.explain(
            tenant_id="test",
            X=X[:1],
            transaction_id="test_txn",
            fraud_probability=0.9,
            feature_names=feature_names,
        )

        assert cf is not None
        assert cf.source == "nearest_neighbor"
        assert cf.nearest_neighbor is not None
        assert cf.latency_ms >= 0

    def test_nn_index_save_load(self, synthetic_data, tmp_path):
        X, y, feature_names = synthetic_data
        index = NearestNeighborIndex(
            tenant_id="test",
            n_features=len(feature_names),
            feature_names=feature_names,
        )
        tids = [f"txn_{i}" for i in range(len(X))]
        index.fit(X, tids, y)
        index.save(tmp_path / "test_index")

        loaded = NearestNeighborIndex.load(tmp_path / "test_index")
        assert loaded.tenant_id == "test"
        assert loaded.size == index.size


# ── Formatter tests ───────────────────────────────────────────────────────────


class TestFormatter:
    def test_format_with_all(
        self, sample_shap_explanation, sample_counterfactual, sample_confidence
    ):
        formatter = ExplanationFormatter(max_drivers=5)
        report = formatter.format(
            fraud_probability=0.94,
            shap=sample_shap_explanation,
            counterfactual=sample_counterfactual,
            confidence=sample_confidence,
        )
        assert report.fraud_probability == 0.94
        assert len(report.risk_drivers) > 0
        assert report.counterfactual_summary is not None
        assert len(report.minimal_changes) > 0

    def test_format_without_counterfactual(
        self, sample_shap_explanation, sample_confidence
    ):
        formatter = ExplanationFormatter()
        report = formatter.format(
            fraud_probability=0.94,
            shap=sample_shap_explanation,
            confidence=sample_confidence,
        )
        assert report.counterfactual_summary is None
        assert len(report.minimal_changes) == 0

    def test_format_value(self):
        assert ExplanationFormatter._format_value(1_500_000) == "1.5M"
        assert ExplanationFormatter._format_value(45_000) == "45.0K"
        assert ExplanationFormatter._format_value(0.123) == "0.12"


# ── Cache tests ───────────────────────────────────────────────────────────────


class TestCache:
    def test_put_and_get(self):
        cache = ExplanationCache(max_size=100, ttl_seconds=60)
        exp = FullExplanation(
            transaction_id="txn_001",
            tenant_id="tenant_001",
            fraud_probability=0.9,
        )
        cache.put("tenant_001", "txn_001", exp)
        result = cache.get("tenant_001", "txn_001")
        assert result is not None
        assert result.transaction_id == "txn_001"

    def test_cache_miss(self):
        cache = ExplanationCache()
        result = cache.get("tenant_001", "nonexistent")
        assert result is None

    def test_cache_invalidation(self):
        cache = ExplanationCache()
        exp = FullExplanation(
            transaction_id="txn_001",
            tenant_id="tenant_001",
            fraud_probability=0.9,
        )
        cache.put("tenant_001", "txn_001", exp)
        removed = cache.invalidate("tenant_001")
        assert removed == 1
        assert cache.get("tenant_001", "txn_001") is None

    def test_cache_lru_eviction(self):
        cache = ExplanationCache(max_size=2, ttl_seconds=60)
        for i in range(4):
            exp = FullExplanation(
                transaction_id=f"txn_{i}",
                tenant_id="t",
                fraud_probability=0.5,
            )
            cache.put("t", f"txn_{i}", exp)
        assert cache.size == 2

    def test_cache_hit_rate(self):
        cache = ExplanationCache()
        exp = FullExplanation(
            transaction_id="txn_001", tenant_id="t", fraud_probability=0.5
        )
        cache.put("t", "txn_001", exp)
        cache.get("t", "txn_001")  # hit
        cache.get("t", "nonexistent")  # miss
        assert cache.hit_rate == pytest.approx(0.5)

    def test_shap_cache(self):
        cache = SHAPCache(max_explainers=2)
        cache.put("tenant_1", "v1", "explainer_1")
        cache.put("tenant_2", "v1", "explainer_2")
        assert cache.get("tenant_1", "v1") == "explainer_1"


# ── Monitor tests ─────────────────────────────────────────────────────────────


class TestMonitor:
    def test_record_explanation(self):
        monitor = ExplainabilityMonitor()
        monitor.record_explanation(
            shap_latency_ms=10.0,
            total_latency_ms=25.0,
            counterfactual_latency_ms=8.0,
            cache_hit=False,
            counterfactual_success=True,
            fraud_drivers=["amount is 850K"],
        )
        metrics = monitor.get_metrics()
        assert metrics.total_explanations == 1

    def test_detect_issues(self):
        monitor = ExplainabilityMonitor()
        for _ in range(200):
            monitor.record_explanation(
                shap_latency_ms=30.0,  # Too high
                total_latency_ms=50.0,  # Too high
            )
        issues = monitor.detect_issues(max_total_latency_ms=40.0)
        assert issues["has_issues"]


# ── ExplainabilityEngine tests ────────────────────────────────────────────────


class TestExplainabilityEngine:
    def test_init(self):
        config = ExplainabilityConfig(enabled=True)
        engine = ExplainabilityEngine(config=config)
        assert not engine._is_fitted

    def test_explain_without_fit(self, synthetic_data):
        X, _, feature_names = synthetic_data
        engine = ExplainabilityEngine()
        result = engine.explain(
            tenant_id="test",
            transaction_id="txn_001",
            X=X[:1],
            fraud_probability=0.9,
            feature_names=feature_names,
        )
        # Should return something even without fitting
        assert result is not None
        assert result.fraud_probability == 0.9

    def test_cache_integration(self, synthetic_data):
        X, _, feature_names = synthetic_data
        engine = ExplainabilityEngine()

        result1 = engine.explain(
            tenant_id="test",
            transaction_id="txn_001",
            X=X[:1],
            fraud_probability=0.9,
            feature_names=feature_names,
        )
        result2 = engine.explain(
            tenant_id="test",
            transaction_id="txn_001",
            X=X[:1],
            fraud_probability=0.9,
            feature_names=feature_names,
        )
        # Second call should be cached
        assert result1.transaction_id == result2.transaction_id


# ── Performance tests ─────────────────────────────────────────────────────────


class TestPerformance:
    def test_cache_get_latency(self):
        cache = ExplanationCache(max_size=1000)
        exp = FullExplanation(
            transaction_id="txn_001",
            tenant_id="t",
            fraud_probability=0.5,
        )
        cache.put("t", "txn_001", exp)

        times = []
        for _ in range(1000):
            t0 = time.perf_counter()
            cache.get("t", "txn_001")
            times.append((time.perf_counter() - t0) * 1000)

        p99 = np.percentile(times, 99)
        assert p99 < 1.0, f"Cache get P99={p99:.3f}ms exceeds 1ms"

    def test_formatter_latency(self, sample_shap_explanation, sample_confidence):
        formatter = ExplanationFormatter()
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            formatter.format(
                fraud_probability=0.94,
                shap=sample_shap_explanation,
                confidence=sample_confidence,
            )
            times.append((time.perf_counter() - t0) * 1000)

        p99 = np.percentile(times, 99)
        assert p99 < 5.0, f"Formatter P99={p99:.3f}ms exceeds 5ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
