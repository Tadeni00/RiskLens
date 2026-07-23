"""
FraudTrap — Test Suite
Covers: feature engineering, model scoring, API contract, rules engine,
phase transitions, and latency budget validation.
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

from ingestion.schema import TransactionRequest, LabelPayload
from features.engineering import (
    compute_transaction_features,
    compute_device_geo_features,
    assemble_feature_vector,
    _haversine,
)
from models.cold_start.ensemble import ColdStartEnsemble, FraudVAE
from training.pipeline import _generate_synthetic_data

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_txn() -> TransactionRequest:
    return TransactionRequest(
        transaction_id="test_txn_001",
        tenant_id="bank_test",
        account_id="tok_acct_abc",
        amount=45_000.0,
        currency="NGN",
        timestamp=datetime(2025, 5, 15, 14, 23, 11, tzinfo=timezone.utc),
        transaction_type="PAYMENT",
        channel="MOBILE",
        merchant_id="tok_merch_xyz",
        merchant_category_code="5411",
        device_id="tok_dev_001",
        country_code="NG",
    )


@pytest.fixture
def fraud_txn() -> TransactionRequest:
    return TransactionRequest(
        transaction_id="test_fraud_001",
        tenant_id="bank_test",
        account_id="tok_acct_sus",
        amount=999_999.0,
        currency="NGN",
        timestamp=datetime(2025, 5, 15, 3, 0, 0, tzinfo=timezone.utc),  # 3am
        transaction_type="TRANSFER",
        channel="API",
        device_id="tok_dev_new",
        country_code="US",
        latitude=40.7128,
        longitude=-74.0060,
        is_very_round_amount=True,
    )


@pytest.fixture(scope="module")
def synthetic_dataset():
    return _generate_synthetic_data(n=2_000)


@pytest.fixture(scope="module")
def fitted_cold_start(synthetic_dataset):
    df = synthetic_dataset
    feature_cols = [
        c
        for c in df.columns
        if c not in ("label", "transaction_id", "tenant_id", "transaction_timestamp")
    ]
    X = df[feature_cols].fillna(0.0).values
    model = ColdStartEnsemble(input_dim=X.shape[1])
    model.fit(X, epochs=5)  # Fast for testing
    return model, X, df["label"].values


# ── Schema tests ──────────────────────────────────────────────────────────────


class TestSchema:
    def test_transaction_request_valid(self, sample_txn):
        assert sample_txn.amount == 45_000.0
        assert sample_txn.currency == "NGN"
        assert sample_txn.channel == "MOBILE"

    def test_currency_uppercased(self):
        txn = TransactionRequest(
            tenant_id="t1",
            account_id="a1",
            amount=100.0,
            currency="usd",
            timestamp=datetime.now(timezone.utc),
            transaction_type="PAYMENT",
            channel="WEB",
        )
        assert txn.currency == "USD"

    def test_amount_must_be_positive(self):
        with pytest.raises(Exception):
            TransactionRequest(
                tenant_id="t1",
                account_id="a1",
                amount=-100.0,
                currency="NGN",
                timestamp=datetime.now(timezone.utc),
                transaction_type="PAYMENT",
                channel="WEB",
            )

    def test_label_payload_fraud(self):
        label = LabelPayload(
            transaction_id="txn_001",
            tenant_id="bank_test",
            label=1,
            label_source="CHARGEBACK",
            labelled_at=datetime.now(timezone.utc),
        )
        assert label.label == 1

    def test_label_out_of_range(self):
        with pytest.raises(Exception):
            LabelPayload(
                transaction_id="txn_001",
                tenant_id="bank_test",
                label=2,  # must be 0 or 1
                label_source="MANUAL_REVIEW",
                labelled_at=datetime.now(timezone.utc),
            )


# ── Feature engineering tests ─────────────────────────────────────────────────


class TestFeatureEngineering:
    def test_haversine_same_point(self):
        assert _haversine(0, 0, 0, 0) == pytest.approx(0.0)

    def test_haversine_london_lagos(self):
        # London (51.5, -0.13) → Lagos (6.46, 3.39)
        dist = _haversine(51.5, -0.13, 6.46, 3.39)
        assert 5_000 < dist < 6_000, f"Expected ~5500km, got {dist:.0f}km"

    def test_transaction_features_no_redis(self, sample_txn):
        features = assemble_feature_vector(sample_txn, r=None)
        assert "amount" in features
        assert "amount_log" in features
        assert "hour_sin" in features
        assert features["amount"] == pytest.approx(45_000.0)

    def test_log_amount(self, sample_txn):
        import math

        features = assemble_feature_vector(sample_txn, r=None)
        assert features["amount_log"] == pytest.approx(math.log1p(45_000.0))

    def test_night_transaction(self):
        txn = TransactionRequest(
            tenant_id="t1",
            account_id="a1",
            amount=100.0,
            currency="NGN",
            timestamp=datetime(2025, 1, 1, 3, 0, tzinfo=timezone.utc),  # 3am
            transaction_type="PAYMENT",
            channel="WEB",
        )
        features = assemble_feature_vector(txn, r=None)
        assert features["is_night"] == 1.0

    def test_weekend_flag(self):
        # 2025-01-04 is a Saturday
        txn = TransactionRequest(
            tenant_id="t1",
            account_id="a1",
            amount=100.0,
            currency="NGN",
            timestamp=datetime(2025, 1, 4, 10, 0, tzinfo=timezone.utc),
            transaction_type="PAYMENT",
            channel="WEB",
        )
        features = assemble_feature_vector(txn, r=None)
        assert features["is_weekend"] == 1.0

    def test_features_are_finite(self, sample_txn):
        features = assemble_feature_vector(sample_txn, r=None)
        for k, v in features.items():
            assert np.isfinite(v), f"Feature {k} is not finite: {v}"


# ── Cold-start model tests ────────────────────────────────────────────────────


class TestColdStartEnsemble:
    def test_vae_forward_pass(self):
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
            # Fraud scores should be higher on average
            assert (
                fraud_scores > legit_scores
            ), f"Expected fraud_mean ({fraud_scores:.3f}) > legit_mean ({legit_scores:.3f})"

    def test_cold_start_save_load(self, fitted_cold_start, tmp_path):
        model, X, _ = fitted_cold_start
        model.save(tmp_path / "cs_model")
        loaded = ColdStartEnsemble.load(tmp_path / "cs_model")
        assert loaded.is_fitted
        orig_scores = model.score(X[:10])
        loaded_scores = loaded.score(X[:10])
        np.testing.assert_allclose(orig_scores, loaded_scores, rtol=1e-4)


# ── Rules engine tests ────────────────────────────────────────────────────────


class TestRulesEngine:
    def test_impossible_travel_rule(self, sample_txn):
        from scoring.rules_engine import RulesEngine

        engine = RulesEngine(r=None)
        features = {"impossible_travel": 1.0}
        result = engine.evaluate(sample_txn, features)
        assert result.triggered
        assert "IMPOSSIBLE_TRAVEL" in result.rule_ids
        assert result.risk_boost > 0

    def test_no_rule_fires_clean_txn(self, sample_txn):
        from scoring.rules_engine import RulesEngine

        engine = RulesEngine(r=None)
        features = {"impossible_travel": 0.0, "is_very_round_amount": 0.0}
        result = engine.evaluate(sample_txn, features)
        # Clean transaction with small amount should not trigger baseline rules
        sample_txn_copy = sample_txn.model_copy(update={"amount": 100.0})
        result2 = engine.evaluate(sample_txn_copy, features)
        assert not result2.hard_block

    def test_high_round_amount_rule(self, sample_txn):
        from scoring.rules_engine import RulesEngine

        engine = RulesEngine(r=None)
        big_txn = sample_txn.model_copy(update={"amount": 1_500_000.0})
        features = {
            "is_round_amount": 1.0,
            "acct_v_5m_count": 5.0,
            "impossible_travel": 0.0,
        }
        result = engine.evaluate(big_txn, features)
        assert result.triggered


# ── Latency budget test ───────────────────────────────────────────────────────


class TestLatency:
    def test_feature_assembly_under_10ms(self, sample_txn):
        """Feature assembly (no Redis) must complete in < 10ms."""
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            assemble_feature_vector(sample_txn, r=None)
            times.append((time.perf_counter() - t0) * 1000)
        p95 = np.percentile(times, 95)
        assert p95 < 10.0, f"Feature assembly P95={p95:.2f}ms exceeds 10ms budget"

    def test_cold_start_scoring_under_50ms(self, fitted_cold_start):
        """Cold-start ensemble scoring must complete in < 50ms for single sample."""
        model, X, _ = fitted_cold_start
        sample = X[:1]
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            model.score(sample)
            times.append((time.perf_counter() - t0) * 1000)
        p95 = np.percentile(times, 95)
        assert p95 < 50.0, f"ColdStart scoring P95={p95:.2f}ms exceeds 50ms budget"


# ── Phase state tests ─────────────────────────────────────────────────────────


class TestPhaseState:
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
            confirmed_fraud_labels=50,  # below threshold
            total_transactions=600_000,
            metrics={"pr_auc": 0.68},
            first_transaction_at="2024-01-01T00:00:00+00:00",
        )
        evaluator = PhaseTransitionEvaluator()
        ready, checks = evaluator.should_transition_to_semi(state)
        assert not ready
        assert not checks["fraud_labels"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
