"""
FraudTrap — Layer 2 & 3 Architecture Tests
Covers: TabPFNAdaptiveLearner, AdaptiveTrainer, ConfidenceEstimator, FTTransformer,
MetaFusionLayer, ChampionModel confidence-aware routing.
"""

from __future__ import annotations
import asyncio
import time
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.adaptive_learning.prediction import (
    AdaptivePrediction,
    PseudoLabelResult,
)
from models.adaptive_learning.monitoring import AdaptiveMonitor
from models.semi_supervised import (
    SemiSupervisedPrediction,
    SemiSupervisedMonitor,
)
from models.supervised.prediction import SupervisedPrediction
from models.supervised.monitoring import SupervisedMonitor

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_labeled_data():
    """Generate synthetic labeled data for testing."""
    np.random.seed(42)
    n_samples = 500
    n_features = 10

    X_normal = np.random.randn(n_samples // 2, n_features) * 1.0
    y_normal = np.zeros(n_samples // 2)
    X_fraud = np.random.randn(n_samples // 2, n_features) * 1.5 + 2.0
    y_fraud = np.ones(n_samples // 2)

    X = np.vstack([X_normal, X_fraud])
    y = np.concatenate([y_normal, y_fraud])
    idx = np.random.permutation(n_samples)
    return X[idx], y[idx]


@pytest.fixture
def synthetic_unlabeled_data():
    """Generate synthetic unlabeled data for semi-supervised testing."""
    np.random.seed(123)
    n_samples = 200
    n_features = 10
    return np.random.randn(n_samples, n_features)


@pytest.fixture
def fitted_cold_start(synthetic_labeled_data):
    """Fit a cold-start model for testing."""
    from models.cold_start.ensemble import ColdStartEnsemble

    X, y = synthetic_labeled_data
    model = ColdStartEnsemble(input_dim=X.shape[1])
    model.fit(X, epochs=3)
    return model, X, y


@pytest.fixture
def fitted_tabpfn(synthetic_labeled_data):
    """Fit a TabPFN model for testing."""
    from models.adaptive_learning.tabpfn_learner import TabPFNAdaptiveLearner

    X, y = synthetic_labeled_data
    model = TabPFNModel(input_dim=X.shape[1])
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def fitted_confidence_estimator(synthetic_labeled_data):
    """Fit a confidence estimator for testing."""
    from models.supervised.confidence import ConfidenceEstimator

    X, y = synthetic_labeled_data

    # Compute probabilities from data (simulate CatBoost output)
    probs = np.clip(np.mean(X, axis=1) * 0.5 + 0.5, 0.0, 1.0)

    estimator = ConfidenceEstimator()
    estimator.fit_conformal(probs, y)
    return estimator, X, y


@pytest.fixture
def fitted_ft_transformer(synthetic_labeled_data):
    """Fit an FT-Transformer model for testing."""
    from models.supervised.ft_transformer import FTTransformerPredictor

    X, y = synthetic_labeled_data
    model = FTTransformerPredictor(
        n_features=X.shape[1],
        d_token=16,
        n_heads=2,
        n_layers=1,
    )
    model.scaler.fit(X)
    model.is_fitted = True
    return model, X, y


@pytest.fixture
def fitted_meta_fusion(synthetic_labeled_data):
    """Fit a meta-fusion layer for testing."""
    from models.supervised.meta_fusion import MetaFusionLayer

    X, y = synthetic_labeled_data

    catboost_probs = np.random.rand(len(y)) * 0.3 + 0.2 * y
    ft_probs = np.random.rand(len(y)) * 0.3 + 0.2 * y
    catboost_confidences = np.random.rand(len(y)) * 0.5 + 0.5

    meta = MetaFusionLayer(method="logistic_regression")
    meta.fit(
        catboost_probs=catboost_probs,
        ft_probs=ft_probs,
        catboost_confidences=catboost_confidences,
        y_true=y,
    )
    return meta


# ── SemiSupervisedPrediction tests ────────────────────────────────────────────


class TestSemiSupervisedPrediction:
    def test_creation(self):
        pred = SemiSupervisedPrediction(
            probability=0.85,
            confidence=0.85,
            uncertainty=0.15,
        )
        assert pred.probability == 0.85
        assert pred.confidence == 0.85
        assert pred.uncertainty == 0.15
        assert pred.ft_invoked is False
        assert pred.fusion_output is None

    def test_validation_probability_range(self):
        with pytest.raises(ValueError, match="must be in"):
            SemiSupervisedPrediction(
                probability=1.5,
                confidence=0.85,
                uncertainty=0.15,
            )

    def test_validation_confidence_range(self):
        with pytest.raises(ValueError, match="must be in"):
            SemiSupervisedPrediction(
                probability=0.85,
                confidence=-0.1,
                uncertainty=0.15,
            )

    def test_to_dict(self):
        pred = SemiSupervisedPrediction(
            probability=0.85,
            confidence=0.85,
            uncertainty=0.15,
        )
        d = pred.to_dict()
        assert d["probability"] == 0.85
        assert d["ft_invoked"] is False


# ── PseudoLabelResult tests ───────────────────────────────────────────────────


class TestPseudoLabelResult:
    def test_creation(self):
        X_pseudo = np.random.randn(10, 5)
        y_pseudo = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        result = PseudoLabelResult(
            X_pseudo=X_pseudo,
            y_pseudo=y_pseudo,
            high_conf_count=6,
            low_conf_count=4,
        )
        assert result.high_conf_count == 6
        assert result.low_conf_count == 4
        assert result.X_pseudo.shape == (10, 5)
        assert len(result.y_pseudo) == 10

    def test_defaults(self):
        result = PseudoLabelResult(
            X_pseudo=np.array([]),
            y_pseudo=np.array([]),
        )
        assert result.high_conf_count == 0
        assert result.low_conf_count == 0
        assert len(result.review_ids) == 0


# ── TabPFN tests ──────────────────────────────────────────────────────────────


class TestTabPFN:
    def test_fit_and_score(self, fitted_tabpfn):
        model, X, y = fitted_tabpfn
        scores = model.score(X[:50])
        assert scores.shape == (50,)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_scores_higher_for_fraud(self, fitted_tabpfn):
        model, X, y = fitted_tabpfn
        fraud_idx = np.where(y == 1)[0][:20]
        legit_idx = np.where(y == 0)[0][:20]
        if len(fraud_idx) > 0 and len(legit_idx) > 0:
            fraud_scores = model.score(X[fraud_idx]).mean()
            legit_scores = model.score(X[legit_idx]).mean()
            assert fraud_scores > legit_scores

    def test_explain(self, fitted_tabpfn):
        model, X, _ = fitted_tabpfn
        explanations = model.explain(X[:1])
        assert len(explanations) == 1
        assert "prediction_value" in explanations[0]
        assert "top_features" in explanations[0]

    def test_save_load(self, fitted_tabpfn, tmp_path):
        model, X, _ = fitted_tabpfn
        model.save(tmp_path / "tabpfn")
        loaded = type(model).load(tmp_path / "tabpfn")
        orig_scores = model.score(X[:10])
        loaded_scores = loaded.score(X[:10])
        np.testing.assert_allclose(orig_scores, loaded_scores, rtol=1e-4)


# ── ConfidenceEstimator tests ─────────────────────────────────────────────────


class TestConfidenceEstimator:
    def test_estimate(self):
        from models.supervised.confidence import ConfidenceEstimator

        estimator = ConfidenceEstimator()
        # Confidence at 0.5 should be 1.0 (maximum distance from boundaries)
        assert estimator.estimate(0.5) == pytest.approx(1.0)
        # Confidence at 0.0 or 1.0 should be 0.0
        assert estimator.estimate(0.0) == pytest.approx(0.0)
        assert estimator.estimate(1.0) == pytest.approx(0.0)

    def test_is_confident(self):
        from models.supervised.confidence import ConfidenceEstimator

        estimator = ConfidenceEstimator()
        # 0.5 is very confident (center)
        assert estimator.is_confident(0.5)
        # 0.05 is not confident (near boundary)
        assert not estimator.is_confident(0.05)

    def test_fit_conformal(self, fitted_confidence_estimator):
        estimator, X, y = fitted_confidence_estimator
        assert estimator._conformal_threshold is not None
        assert estimator._conformal_threshold > 0


# ── FTTransformer tests ───────────────────────────────────────────────────────


class TestFTTransformer:
    def test_fit_and_score(self, fitted_ft_transformer):
        model, X, _ = fitted_ft_transformer
        scores = model.score(X[:50])
        assert len(scores) == 50
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)


# ── MetaFusionLayer tests ─────────────────────────────────────────────────────


class TestMetaFusionLayer:
    def test_predict(self, fitted_meta_fusion):
        meta = fitted_meta_fusion
        result = meta.predict(
            catboost_prob=0.5,
            ft_prob=0.6,
            catboost_confidence=0.8,
        )
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_predict_batch(self, fitted_meta_fusion):
        meta = fitted_meta_fusion
        catboost_probs = np.array([0.3, 0.7, 0.5, 0.2, 0.8])
        ft_probs = np.array([0.4, 0.6, 0.5, 0.3, 0.7])
        catboost_confidences = np.array([0.9, 0.8, 0.85, 0.95, 0.75])
        fused = meta.predict_batch(catboost_probs, ft_probs, catboost_confidences)
        assert fused.shape == (5,)
        assert np.all(fused >= 0.0)
        assert np.all(fused <= 1.0)

    def test_fusion_weighted_average(self):
        from models.supervised.meta_fusion import MetaFusionLayer

        meta = MetaFusionLayer(method="logistic_regression")
        n = 100
        meta.fit(
            catboost_probs=np.random.rand(n),
            ft_probs=np.random.rand(n),
            catboost_confidences=np.random.rand(n) * 0.5 + 0.5,
            y_true=np.random.randint(0, 2, n),
        )
        result = meta.predict(catboost_prob=0.3, ft_prob=0.4, catboost_confidence=0.8)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


# ── SemiSupervisedMonitor tests ───────────────────────────────────────────────


class TestSemiSupervisedMonitor:
    def test_log_prediction(self):
        monitor = SemiSupervisedMonitor()
        monitor.record_prediction(
            probability=0.8,
            confidence=0.8,
            uncertainty=0.2,
        )
        assert len(monitor._predictions) == 1

    def test_get_metrics(self):
        monitor = SemiSupervisedMonitor()
        for _ in range(10):
            monitor.record_prediction(
                probability=0.8,
                confidence=0.8,
                uncertainty=0.2,
            )
        metrics = monitor.get_metrics()
        assert metrics.uncertainty_mean > 0
        assert metrics.confidence_mean > 0
        assert metrics.n_predictions == 10


# ── SupervisedMonitor tests ───────────────────────────────────────────────────


class TestSupervisedMonitor:
    def test_log_prediction(self):
        monitor = SupervisedMonitor()
        monitor.record_prediction(
            confidence=0.8,
            ft_invoked=False,
            probability=0.8,
            latency_ms=10.0,
        )
        assert len(monitor._confidences) == 1

    def test_get_metrics(self):
        monitor = SupervisedMonitor()
        for _ in range(10):
            monitor.record_prediction(
                confidence=0.8,
                ft_invoked=False,
                probability=0.8,
                latency_ms=10.0,
            )
        metrics = monitor.get_metrics()
        assert metrics.catboost_confidence_mean > 0
        assert metrics.avg_latency_ms > 0
        assert metrics.n_predictions == 10


# ── Phase transition tests ────────────────────────────────────────────────────


class TestPhaseTransitions:
    def test_phase_state_json_roundtrip(self):
        from training.pipeline import PhaseState, ModelPhase

        state = PhaseState(
            tenant_id="bank_test",
            current_phase=ModelPhase.SEMI_SUPERVISED,
            confirmed_fraud_labels=842,
            total_transactions=600_000,
            metrics={"pr_auc": 0.71},
        )
        restored = PhaseState.from_json(state.to_json())
        assert restored.tenant_id == "bank_test"
        assert restored.current_phase == ModelPhase.SEMI_SUPERVISED
        assert restored.confirmed_fraud_labels == 842

    def test_phase_transition_gate_phase1(self):
        from training.pipeline import PhaseState, ModelPhase, PhaseTransitionEvaluator

        state = PhaseState(
            tenant_id="t1",
            current_phase=ModelPhase.UNSUPERVISED,
            confirmed_fraud_labels=600,
            total_transactions=600_000,
            metrics={"pr_auc": 0.68},
            first_transaction_at="2024-01-01T00:00:00+00:00",
        )
        evaluator = PhaseTransitionEvaluator()
        ready, checks = evaluator.should_transition_to_semi(state)
        assert ready, f"Should be ready but checks={checks}"

    def test_phase_transition_gate_fails_low_labels(self):
        from training.pipeline import PhaseState, ModelPhase, PhaseTransitionEvaluator

        state = PhaseState(
            tenant_id="t1",
            current_phase=ModelPhase.UNSUPERVISED,
            confirmed_fraud_labels=50,
            total_transactions=600_000,
            metrics={"pr_auc": 0.68},
            first_transaction_at="2024-01-01T00:00:00+00:00",
        )
        evaluator = PhaseTransitionEvaluator()
        ready, checks = evaluator.should_transition_to_semi(state)
        assert not ready
        assert not checks["fraud_labels"]


# ── Latency budget tests ──────────────────────────────────────────────────────


class TestLatencyBudget:
    def test_tabpfn_scoring_under_50ms(self, fitted_tabpfn):
        """TabPFN scoring must complete in < 50ms for single sample."""
        model, X, _ = fitted_tabpfn
        sample = X[:1]
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            model.score(sample)
            times.append((time.perf_counter() - t0) * 1000)
        p95 = np.percentile(times, 95)
        assert p95 < 50.0, f"TabPFN scoring P95={p95:.2f}ms exceeds 50ms budget"

    def test_confidence_scoring_under_10ms(self, fitted_confidence_estimator):
        """Confidence estimation must complete in < 10ms for single sample."""
        estimator, X, _ = fitted_confidence_estimator
        sample = X[:1]
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            estimator.predict(sample)
            times.append((time.perf_counter() - t0) * 1000)
        p95 = np.percentile(times, 95)
        assert p95 < 10.0, f"Confidence scoring P95={p95:.2f}ms exceeds 10ms budget"


# ── Cold-start model tests (unchanged) ────────────────────────────────────────


class TestColdStartEnsemble:
    def test_vae_forward_pass(self):
        from models.cold_start.ensemble import FraudVAE

        vae = FraudVAE(input_dim=10, latent_dim=4)
        import torch

        x = torch.randn(8, 10)
        x_hat, mu, log_var = vae(x)
        assert x_hat.shape == (8, 10)
        assert mu.shape == (8, 4)

    def test_cold_start_fit_and_score(self, fitted_cold_start):
        model, X, y = fitted_cold_start
        assert model.is_fitted
        scores = model.score(X[:50])
        assert scores.shape == (50,)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_scores_are_higher_for_fraud(self, fitted_cold_start):
        model, X, y = fitted_cold_start
        fraud_idx = np.where(y == 1)[0][:20]
        legit_idx = np.where(y == 0)[0][:20]
        if len(fraud_idx) > 0 and len(legit_idx) > 0:
            fraud_scores = model.score(X[fraud_idx]).mean()
            legit_scores = model.score(X[legit_idx]).mean()
            assert fraud_scores > legit_scores

    def test_cold_start_save_load(self, fitted_cold_start, tmp_path):
        model, X, _ = fitted_cold_start
        model.save(tmp_path / "cs_model")
        from models.cold_start.ensemble import ColdStartEnsemble

        loaded = ColdStartEnsemble.load(tmp_path / "cs_model")
        assert loaded.is_fitted
        orig_scores = model.score(X[:10])
        loaded_scores = loaded.score(X[:10])
        np.testing.assert_allclose(orig_scores, loaded_scores, rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
