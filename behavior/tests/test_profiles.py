"""
RiskLens Behavioral Intelligence Layer - Unit Tests
"""

import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock

# Import behavior modules
from behavior.utils.online_statistics import (
    OnlineMeanVariance,
    ExponentialMovingAverage,
    RollingWindow,
    CountMinSketch,
    CircularBuffer,
    percentile,
    cosine_similarity,
    haversine_distance,
)

from behavior.profiles.customer import CustomerBehaviorProfile
from behavior.profiles.merchant import MerchantBehaviorProfile
from behavior.profiles.device import DeviceBehaviorProfile
from behavior.profiles.beneficiary import BeneficiaryBehaviorProfile
from behavior.profiles.payment_instrument import PaymentInstrumentProfile
from behavior.profiles.device import DeviceBehaviorProfile
from behavior.profiles.beneficiary import BeneficiaryBehaviorProfile
from behavior.profiles.payment_instrument import PaymentInstrumentProfile

from behavior.utils.online_statistics import (
    OnlineMeanVariance,
    ExponentialMovingAverage,
    RollingWindow,
    CountMinSketch,
    CircularBuffer,
    percentile,
    cosine_similarity,
    haversine_distance,
)

from behavior.profiles.customer import CustomerBehaviorProfile
from behavior.profiles.merchant import MerchantBehaviorProfile
from behavior.profiles.device import DeviceBehaviorProfile
from behavior.profiles.beneficiary import BeneficiaryBehaviorProfile
from behavior.profiles.payment_instrument import PaymentInstrumentProfile
from behavior.profiles.device import DeviceBehaviorProfile
from behavior.profiles.beneficiary import BeneficiaryBehaviorProfile
from behavior.profiles.payment_instrument import PaymentInstrumentProfile

from behavior.utils.online_statistics import (
    OnlineMeanVariance,
    ExponentialMovingAverage,
    RollingWindow,
    CountMinSketch,
    CircularBuffer,
    percentile,
    cosine_similarity,
    haversine_distance,
)

from behavior.feature_generation.velocity import (
    VelocityFeatureGenerator,
    compute_velocity_features_from_profile,
)

from behavior.feature_generation.similarity import (
    merchant_similarity,
    device_similarity,
    country_similarity,
    typing_similarity,
    cross_country_flag,
    merchant_similarity,
    device_similarity,
    country_similarity,
    typing_similarity,
)

from behavior.feature_generation.trust import (
    get_device_trust_score,
    get_merchant_trust_score,
    get_beneficiary_trust_score,
    get_customer_reputation,
    get_historical_chargeback_rate,
    get_historical_fraud_rate,
    get_customer_risk_score,
    get_merchant_risk_score,
    get_device_risk_score,
)

from behavior.feature_generation.novelty import (
    get_new_device_flag,
    get_new_merchant_flag,
    get_new_ip_flag,
    get_new_country_flag,
    get_new_merchant_flag,
    get_new_device_flag,
    NoveltyDetector,
)

from behavior.feature_generation.similarity import (
    merchant_similarity,
    device_similarity,
    country_similarity,
    typing_similarity,
    cross_country_flag,
)

from behavior.feature_generation.trust import (
    get_device_trust_score,
    get_merchant_trust_score,
    get_beneficiary_trust_score,
    get_customer_reputation,
    get_historical_chargeback_rate,
    get_historical_fraud_rate,
    get_customer_risk_score,
    get_merchant_risk_score,
    get_device_risk_score,
)

from behavior.feature_generation.novelty import (
    get_new_device_flag,
    get_new_merchant_flag,
    get_new_ip_flag,
    get_new_country_flag,
    get_new_merchant_flag,
    get_new_device_flag,
    NoveltyDetector,
)

from behavior.feature_generation.trust import (
    get_device_trust_score,
    get_merchant_trust_score,
    get_beneficiary_trust_score,
    get_customer_reputation,
    get_historical_chargeback_rate,
    get_historical_fraud_rate,
    get_customer_risk_score,
    get_merchant_risk_score,
    get_device_risk_score,
)

from behavior.utils.online_statistics import (
    OnlineMeanVariance,
    ExponentialMovingAverage,
    RollingWindow,
    CountMinSketch,
    CircularBuffer,
    percentile,
    cosine_similarity,
    haversine_distance,
)

from behavior.profiles.customer import CustomerBehaviorProfile
from behavior.profiles.merchant import MerchantBehaviorProfile
from behavior.profiles.device import DeviceBehaviorProfile
from behavior.profiles.beneficiary import BeneficiaryBehaviorProfile
from behavior.profiles.payment_instrument import PaymentInstrumentProfile
from behavior.profiles.device import DeviceBehaviorProfile
from behavior.profiles.beneficiary import BeneficiaryBehaviorProfile
from behavior.profiles.payment_instrument import PaymentInstrumentProfile


class TestOnlineStatistics:
    """Test online statistics utilities."""

    def test_online_mean_variance_basic(self):
        """Test basic Welford's online algorithm."""
        stats = OnlineMeanVariance()

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            stats.update(v)

        assert stats.count == 5
        assert abs(stats.mean - 3.0) < 0.001
        assert abs(stats.variance - 2.5) < 0.001

    def test_exponential_moving_average(self):
        """Test exponential moving average."""
        ema = ExponentialMovingAverage(alpha=0.5)

        ema.update(10.0)
        assert ema.value == 10.0

        ema.update(20.0)
        # 0.5 * 20 + 0.5 * 10 = 15
        assert abs(ema.value - 15.0) < 0.001

        ema.update(30.0)
        # 0.5 * 30 + 0.5 * 15 = 22.5
        assert abs(ema.value - 22.5) < 0.001

    def test_rolling_window(self):
        """Test rolling window statistics."""
        window = RollingWindow(max_size=3)

        window.add(1.0)
        window.add(2.0)
        window.add(3.0)

        assert window.count == 3
        assert window.mean == 2.0

        window.add(4.0)  # Should evict 1.0
        assert window.count == 3
        assert window.mean == 3.0

    def test_cosine_similarity(self):
        """Test cosine similarity."""
        a = [1, 0, 0]
        b = [1, 0, 0]
        assert cosine_similarity(a, b) == 1.0

        a = [1, 0, 0]
        b = [0, 1, 0]
        assert cosine_similarity(a, b) == 0.0

        a = [1, 1, 0]
        b = [1, 1, 0]
        assert abs(cosine_similarity(a, b) - 1.0) < 0.001

    def test_haversine_distance(self):
        """Test Haversine distance calculation."""
        # Lagos to Abuja (approx 530 km)
        dist = haversine_distance(6.5244, 3.3792, 9.0765, 7.3986)
        assert 500 < dist < 600

        # Same point
        dist = haversine_distance(0, 0, 0, 0)
        assert dist == 0.0

    def test_percentile(self):
        """Test percentile calculation."""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        assert percentile(values, 0.5) == 5.5  # Median
        assert percentile(values, 0.0) == 1.0
        assert percentile(values, 1.0) == 10.0


class TestProfiles:
    """Test behavioral profile classes."""

    def test_customer_profile_creation(self):
        """Test CustomerBehaviorProfile creation."""
        profile = CustomerBehaviorProfile(
            customer_id="cust_123", tenant_id="bank_ng_gtb"
        )

        assert profile.customer_id == "cust_123"
        assert profile.tenant_id == "bank_ng_gtb"
        assert profile.total_transactions == 0

    def test_merchant_profile_creation(self):
        """Test MerchantBehaviorProfile creation."""
        profile = MerchantBehaviorProfile(
            merchant_id="merch_123", tenant_id="bank_ng_gtb"
        )

        assert profile.merchant_id == "merch_123"
        assert profile.tenant_id == "bank_ng_gtb"

    def test_device_profile_creation(self):
        """Test DeviceBehaviorProfile creation."""
        profile = DeviceBehaviorProfile(device_id="dev_123", tenant_id="bank_ng_gtb")

        assert profile.device_id == "dev_123"
        assert profile.tenant_id == "bank_ng_gtb"

    def test_beneficiary_profile_creation(self):
        """Test BeneficiaryBehaviorProfile creation."""
        profile = BeneficiaryBehaviorProfile(
            beneficiary_id="ben_123", tenant_id="bank_ng_gtb"
        )

        assert profile.beneficiary_id == "ben_123"
        assert profile.tenant_id == "bank_ng_gtb"

    def test_payment_instrument_profile_creation(self):
        """Test PaymentInstrumentProfile creation."""
        profile = PaymentInstrumentProfile(
            instrument_id="inst_123", instrument_type="CARD", tenant_id="bank_ng_gtb"
        )

        assert profile.instrument_id == "inst_123"
        assert profile.instrument_type == "CARD"
        assert profile.tenant_id == "bank_ng_gtb"

    def test_device_profile_creation(self):
        """Test DeviceBehaviorProfile creation."""
        profile = DeviceBehaviorProfile(device_id="dev_123", tenant_id="bank_ng_gtb")

        assert profile.device_id == "dev_123"
        assert profile.tenant_id == "bank_ng_gtb"


class TestOnlineStatistics:
    """Test online statistics utilities."""

    def test_online_mean_variance_basic(self):
        """Test basic Welford's online algorithm."""
        stats = OnlineMeanVariance()

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            stats.update(v)

        assert stats.count == 5
        assert abs(stats.mean - 3.0) < 0.001
        assert abs(stats.variance - 2.5) < 0.001

    def test_exponential_moving_average(self):
        """Test exponential moving average."""
        ema = ExponentialMovingAverage(alpha=0.5)

        ema.update(10.0)
        assert ema.value == 10.0

        ema.update(20.0)
        # 0.5 * 20 + 0.5 * 10 = 15
        assert abs(ema.value - 15.0) < 0.001

    def test_rolling_window(self):
        """Test rolling window statistics."""
        window = RollingWindow(max_size=3)

        window.add(1.0)
        window.add(2.0)
        window.add(3.0)

        assert window.count == 3
        assert window.mean == 2.0

        window.add(4.0)  # Should evict 1.0
        assert window.count == 3
        assert window.mean == 3.0

    def test_cosine_similarity(self):
        """Test cosine similarity."""
        a = [1, 0, 0]
        b = [1, 0, 0]
        assert cosine_similarity(a, b) == 1.0

        a = [1, 0, 0]
        b = [0, 1, 0]
        assert cosine_similarity(a, b) == 0.0

    def test_haversine_distance(self):
        """Test Haversine distance calculation."""
        # Lagos to Abuja (approx 530 km)
        dist = haversine_distance(6.5244, 3.3792, 9.0765, 7.3986)
        assert 500 < dist < 600

        # Same point
        dist = haversine_distance(0, 0, 0, 0)
        assert dist == 0.0

    def test_percentile(self):
        """Test percentile calculation."""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        assert percentile(values, 0.5) == 5.5  # Median
        assert percentile(values, 0.0) == 1.0
        assert percentile(values, 1.0) == 10.0


class TestProfiles:
    """Test behavioral profile classes."""

    def test_customer_profile_creation(self):
        """Test CustomerBehaviorProfile creation."""
        profile = CustomerBehaviorProfile(
            customer_id="cust_123", tenant_id="bank_ng_gtb"
        )

        assert profile.customer_id == "cust_123"
        assert profile.tenant_id == "bank_ng_gtb"
        assert profile.total_transactions == 0

    def test_merchant_profile(self):
        """Test MerchantBehaviorProfile."""
        profile = MerchantBehaviorProfile(
            merchant_id="merch_123", tenant_id="bank_ng_gtb"
        )

        assert profile.merchant_id == "merch_123"
        assert profile.tenant_id == "bank_ng_gtb"

    def test_device_profile(self):
        """Test DeviceBehaviorProfile."""
        profile = DeviceBehaviorProfile(device_id="dev_123", tenant_id="bank_ng_gtb")

        assert profile.device_id == "dev_123"
        assert profile.tenant_id == "bank_ng_gtb"

    def test_beneficiary_profile(self):
        """Test BeneficiaryBehaviorProfile."""
        profile = BeneficiaryBehaviorProfile(
            beneficiary_id="ben_123", tenant_id="bank_ng_gtb"
        )

        assert profile.beneficiary_id == "ben_123"
        assert profile.tenant_id == "bank_ng_gtb"

    def test_payment_instrument_profile(self):
        """Test PaymentInstrumentProfile."""
        profile = PaymentInstrumentProfile(
            instrument_id="inst_123", instrument_type="CARD", tenant_id="bank_ng_gtb"
        )

        assert profile.instrument_id == "inst_123"
        assert profile.instrument_type == "CARD"
        assert profile.tenant_id == "bank_ng_gtb"

    def test_device_profile(self):
        """Test DeviceBehaviorProfile."""
        profile = DeviceBehaviorProfile(device_id="dev_123", tenant_id="bank_ng_gtb")

        assert profile.device_id == "dev_123"
        assert profile.tenant_id == "bank_ng_gtb"

    def test_beneficiary_profile(self):
        """Test BeneficiaryBehaviorProfile."""
        profile = BeneficiaryBehaviorProfile(
            beneficiary_id="ben_123", tenant_id="bank_ng_gtb"
        )

        assert profile.beneficiary_id == "ben_123"
        assert profile.tenant_id == "bank_ng_gtb"


class TestFeatureGeneration:
    """Test feature generation utilities."""

    def test_velocity_features(self):
        """Test velocity feature computation."""
        from behavior.feature_generation.velocity import (
            compute_velocity_features_from_profile,
        )

        # Mock customer profile
        profile = Mock()
        profile.velocity_windows = {
            "1m": type(
                "obj", (object,), {"count": 5, "total_amt": 1000.0, "mean_amt": 200.0}
            ),
            "1h": type(
                "obj", (object,), {"count": 20, "total_amt": 5000.0, "mean_amt": 250.0}
            ),
            "24h": type(
                "obj", (object,), {"count": 50, "total_amt": 20000.0, "mean_amt": 400.0}
            ),
        }

        features = compute_velocity_features_from_profile(None, None)
        assert isinstance(features, dict)

    def test_heuristic_score(self):
        """Test heuristic score computation."""
        from behavior.feature_generation.trust import get_historical_chargeback_rate

        profile = Mock()
        profile.total_transactions = 100
        profile.chargeback_count = 5

        rate = get_historical_chargeback_rate(profile)
        assert rate == 0.05  # 5/100

    def test_trust_scoring(self):
        """Test trust scoring functions."""
        from behavior.feature_generation.trust import get_device_trust_score

        # Test device in trusted devices with high frequency
        profile = Mock()
        profile.trusted_devices = {"dev_1", "dev_2"}
        profile.device_fingerprint_frequency = {"dev_1": 10, "dev_2": 5, "dev_3": 1}
        profile.get_device_trust_score = Mock(
            side_effect=lambda d: 1.0 if d in {"dev_1", "dev_2"} else 0.5
        )

        score = get_device_trust_score(profile, "dev_1")
        assert score == 1.0

        # Test device not in trusted devices but has frequency
        profile2 = Mock()
        profile2.trusted_devices = set()
        profile2.device_fingerprint_frequency = {"dev_1": 10}
        profile2.get_device_trust_score = Mock(side_effect=lambda d: 0.5)

        score = get_device_trust_score(profile2, "dev_1")
        assert score < 1.0

    def test_novelty_detection(self):
        """Test novelty detection functions."""
        from behavior.feature_generation.novelty import (
            get_new_device_flag,
            get_new_merchant_flag,
            get_new_country_flag,
        )

        profile = Mock()
        profile.trusted_devices = {"dev_1"}
        profile.merchant_frequency = {"merch_1": 10}
        profile.country_frequency = {"NG": 100}
        profile.trusted_ips = {"ip_1"}

        assert get_new_device_flag(Mock(trusted_devices={"dev_1"}), "dev_1") == 0.0
        assert get_new_device_flag(Mock(trusted_devices={"dev_1"}), "dev_2") == 1.0

        assert get_new_merchant_flag(Mock(merchant_frequency={"m1": 10}), "m1") == 0.0
        assert get_new_merchant_flag(Mock(merchant_frequency={"m1": 10}), "m2") == 1.0

        assert get_new_country_flag(Mock(country_frequency={"NG": 10}), "NG") == 0.0
        assert get_new_country_flag(Mock(country_frequency={"NG": 10}), "US") == 1.0


class TestIntegration:
    """Integration tests for behavioral intelligence layer."""

    def test_feature_generation_integration(self):
        """Test feature generation pipeline integration."""
        # This would test the full pipeline with mocked dependencies
        pass

    def test_behavior_engine_initialization(self):
        """Test BehaviorEngine initialization."""
        from behavior.services.behavior_engine import (
            BehaviorEngine,
            BehaviorEngineConfig,
        )
        from behavior.storage.redis_store import get_feature_store

        # This would test with a real feature store
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
