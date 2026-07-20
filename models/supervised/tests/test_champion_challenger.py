"""
FraudTrap — Champion-Challenger Architecture Tests
Unit and integration tests for the Champion-Challenger supervised learning architecture.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import tempfile
import shutil

from scoring.calibration import ProbabilityCalibrator, evaluate_calibration
from models.supervised.registry import (
    ModelRegistry, ModelMetadata, ModelStatus, PromotionStatus
)
from models.supervised.challengers import (
    XGBoostChallenger, LightGBMChallenger, create_challenger, get_available_algorithms
)
from models.supervised.evaluator import ModelEvaluator, EvaluationMetrics
from models.supervised.promotion import ChampionPromoter, PromotionCriteria


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_data():
    """Generate sample fraud detection data."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # Binary labels with ~10% fraud rate
    y = (np.random.rand(n_samples) < 0.1).astype(int)
    
    return X, y


@pytest.fixture
def sample_probabilities():
    """Generate sample probabilities."""
    np.random.seed(42)
    return np.random.rand(1000).astype(np.float64)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# Probability Calibration Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestProbabilityCalibrator:
    """Tests for ProbabilityCalibrator."""
    
    def test_init_isotonic(self):
        """Test isotonic calibrator initialization."""
        calibrator = ProbabilityCalibrator(method="isotonic")
        assert calibrator.method == "isotonic"
        assert not calibrator._fitted
    
    def test_init_platt(self):
        """Test Platt calibrator initialization."""
        calibrator = ProbabilityCalibrator(method="platt")
        assert calibrator.method == "platt"
        assert not calibrator._fitted
    
    def test_invalid_method(self):
        """Test invalid calibration method."""
        with pytest.raises(ValueError, match="Unknown calibration method"):
            ProbabilityCalibrator(method="invalid")
    
    def test_fit_isotonic(self, sample_data):
        """Test fitting isotonic calibrator."""
        X, y = sample_data
        probs = np.random.rand(len(y))
        
        calibrator = ProbabilityCalibrator(method="isotonic")
        calibrator.fit(probs, y)
        
        assert calibrator._fitted
        assert calibrator._isotonic is not None
    
    def test_fit_platt(self, sample_data):
        """Test fitting Platt calibrator."""
        X, y = sample_data
        probs = np.random.rand(len(y))
        
        calibrator = ProbabilityCalibrator(method="platt")
        calibrator.fit(probs, y)
        
        assert calibrator._fitted
        assert calibrator._platt_a is not None
        assert calibrator._platt_b is not None
    
    def test_transform_isotonic(self, sample_data):
        """Test isotonic calibration transformation."""
        X, y = sample_data
        probs = np.random.rand(len(y))
        
        calibrator = ProbabilityCalibrator(method="isotonic")
        calibrator.fit(probs, y)
        
        calibrated = calibrator.transform(probs)
        
        assert len(calibrated) == len(probs)
        assert np.all(calibrated >= 0)
        assert np.all(calibrated <= 1)
    
    def test_transform_platt(self, sample_data):
        """Test Platt calibration transformation."""
        X, y = sample_data
        probs = np.random.rand(len(y))
        
        calibrator = ProbabilityCalibrator(method="platt")
        calibrator.fit(probs, y)
        
        calibrated = calibrator.transform(probs)
        
        assert len(calibrated) == len(probs)
        assert np.all(calibrated >= 0)
        assert np.all(calibrated <= 1)
    
    def test_fit_transform(self, sample_data):
        """Test fit_transform method."""
        X, y = sample_data
        probs = np.random.rand(len(y))
        
        calibrator = ProbabilityCalibrator(method="isotonic")
        calibrated = calibrator.fit_transform(probs, y)
        
        assert calibrator._fitted
        assert len(calibrated) == len(probs)
    
    def test_save_load(self, sample_data, temp_dir):
        """Test save and load methods."""
        X, y = sample_data
        probs = np.random.rand(len(y))
        
        calibrator = ProbabilityCalibrator(method="isotonic")
        calibrator.fit(probs, y)
        
        # Save
        save_dir = temp_dir / "calibrator"
        calibrator.save(save_dir)
        
        # Load
        loaded = ProbabilityCalibrator.load(save_dir)
        
        assert loaded.method == calibrator.method
        assert loaded._fitted == calibrator._fitted
    
    def test_get_params(self, sample_data):
        """Test get_params method."""
        calibrator = ProbabilityCalibrator(method="isotonic")
        params = calibrator.get_params()
        
        assert params["method"] == "isotonic"
        assert params["fitted"] == False
        assert params["platt_a"] is None
        assert params["platt_b"] is None
    
    def test_repr(self):
        """Test string representation."""
        calibrator = ProbabilityCalibrator(method="isotonic")
        assert "isotonic" in repr(calibrator)
        assert "unfitted" in repr(calibrator)
    
    def test_evaluate_calibration(self, sample_data):
        """Test calibration evaluation."""
        X, y = sample_data
        raw_probs = np.random.rand(len(y))
        
        calibrator = ProbabilityCalibrator(method="isotonic")
        calibrated_probs = calibrator.fit_transform(raw_probs, y)
        
        metrics = evaluate_calibration(raw_probs, calibrated_probs, y)
        
        assert "raw_ece" in metrics
        assert "calibrated_ece" in metrics
        assert "raw_brier" in metrics
        assert "calibrated_brier" in metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Model Registry Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelRegistry:
    """Tests for ModelRegistry."""
    
    def test_init(self, temp_dir):
        """Test registry initialization."""
        registry = ModelRegistry(registry_dir=temp_dir / "registry")
        assert registry.registry_dir.exists()
        assert registry._champion_id is None
    
    def test_register_model(self, temp_dir):
        """Test model registration."""
        registry = ModelRegistry(registry_dir=temp_dir / "registry")
        
        metadata = registry.register(
            model_id="test_model_1",
            version="1.0.0",
            algorithm="catboost",
            training_date="2024-01-01",
            dataset_version="1.0.0",
            feature_version="1.0.0",
            metrics={"pr_auc": 0.85, "roc_auc": 0.90, "f2_score": 0.80, "fpr": 0.01},
        )
        
        assert metadata.model_id == "test_model_1"
        assert metadata.pr_auc == 0.85
        assert metadata.status == ModelStatus.CHALLENGER
    
    def test_get_model(self, temp_dir):
        """Test model retrieval."""
        registry = ModelRegistry(registry_dir=temp_dir / "registry")
        
        registry.register(
            model_id="test_model_1",
            version="1.0.0",
            algorithm="catboost",
            training_date="2024-01-01",
            dataset_version="1.0.0",
            feature_version="1.0.0",
            metrics={"pr_auc": 0.85},
        )
        
        model = registry.get_model("test_model_1")
        assert model is not None
        assert model.model_id == "test_model_1"
    
    def test_list_models(self, temp_dir):
        """Test model listing."""
        registry = ModelRegistry(registry_dir=temp_dir / "registry")
        
        # Register multiple models
        for i in range(3):
            registry.register(
                model_id=f"model_{i}",
                version=f"1.0.{i}",
                algorithm="catboost" if i % 2 == 0 else "xgboost",
                training_date="2024-01-01",
                dataset_version="1.0.0",
                feature_version="1.0.0",
                metrics={"pr_auc": 0.8 + i * 0.05},
            )
        
        # List all
        all_models = registry.list_models()
        assert len(all_models) == 3
        
        # List by algorithm
        catboost_models = registry.list_models(algorithm="catboost")
        assert len(catboost_models) == 2
    
    def test_promote_model(self, temp_dir):
        """Test model promotion."""
        registry = ModelRegistry(registry_dir=temp_dir / "registry")
        
        # Register champion
        registry.register(
            model_id="champion_1",
            version="1.0.0",
            algorithm="catboost",
            training_date="2024-01-01",
            dataset_version="1.0.0",
            feature_version="1.0.0",
            metrics={"pr_auc": 0.85, "fpr": 0.01, "calibration_error": 0.03},
        )
        
        # Register challenger
        registry.register(
            model_id="challenger_1",
            version="1.0.0",
            algorithm="xgboost",
            training_date="2024-01-01",
            dataset_version="1.0.0",
            feature_version="1.0.0",
            metrics={"pr_auc": 0.90, "fpr": 0.005, "calibration_error": 0.02},
        )
        
        # Promote challenger
        request = registry.promote("challenger_1", reviewer="test")
        
        assert request is not None
        assert request.status == PromotionStatus.COMPLETED
        
        # Check champion changed
        champion = registry.get_champion()
        assert champion.model_id == "challenger_1"
    
    def test_rollback(self, temp_dir):
        """Test champion rollback."""
        registry = ModelRegistry(registry_dir=temp_dir / "registry")
        
        # Register old champion
        registry.register(
            model_id="old_champion",
            version="1.0.0",
            algorithm="catboost",
            training_date="2024-01-01",
            dataset_version="1.0.0",
            feature_version="1.0.0",
            metrics={"pr_auc": 0.85, "fpr": 0.01, "calibration_error": 0.03},
        )
        registry.promote("old_champion")
        
        # Register and promote new champion
        registry.register(
            model_id="new_champion",
            version="1.0.0",
            algorithm="xgboost",
            training_date="2024-01-02",
            dataset_version="1.0.0",
            feature_version="1.0.0",
            metrics={"pr_auc": 0.90, "fpr": 0.005, "calibration_error": 0.02},
        )
        registry.promote("new_champion")
        
        # Rollback
        result = registry.rollback()
        assert result is True
        
        # Check old champion restored
        champion = registry.get_champion()
        assert champion.model_id == "old_champion"
    
    def test_archive_model(self, temp_dir):
        """Test model archival."""
        registry = ModelRegistry(registry_dir=temp_dir / "registry")
        
        registry.register(
            model_id="model_to_archive",
            version="1.0.0",
            algorithm="catboost",
            training_date="2024-01-01",
            dataset_version="1.0.0",
            feature_version="1.0.0",
            metrics={"pr_auc": 0.85},
        )
        
        result = registry.archive("model_to_archive", notes="Testing archival")
        assert result is True
        
        model = registry.get_model("model_to_archive")
        assert model.status == ModelStatus.ARCHIVED
    
    def test_compare_models(self, temp_dir):
        """Test model comparison."""
        registry = ModelRegistry(registry_dir=temp_dir / "registry")
        
        registry.register(
            model_id="model_a",
            version="1.0.0",
            algorithm="catboost",
            training_date="2024-01-01",
            dataset_version="1.0.0",
            feature_version="1.0.0",
            metrics={"pr_auc": 0.85, "roc_auc": 0.90},
        )
        
        registry.register(
            model_id="model_b",
            version="1.0.0",
            algorithm="xgboost",
            training_date="2024-01-01",
            dataset_version="1.0.0",
            feature_version="1.0.0",
            metrics={"pr_auc": 0.88, "roc_auc": 0.92},
        )
        
        comparison = registry.compare_models("model_a", "model_b")
        
        assert "model_1" in comparison
        assert "model_2" in comparison
        assert "differences" in comparison
        assert comparison["differences"]["pr_auc_diff"] == pytest.approx(-0.03, abs=0.01)
    
    def test_get_stats(self, temp_dir):
        """Test registry statistics."""
        registry = ModelRegistry(registry_dir=temp_dir / "registry")
        
        # Register models
        for i in range(5):
            registry.register(
                model_id=f"model_{i}",
                version=f"1.0.{i}",
                algorithm="catboost" if i % 2 == 0 else "xgboost",
                training_date="2024-01-01",
                dataset_version="1.0.0",
                feature_version="1.0.0",
                metrics={"pr_auc": 0.8 + i * 0.02},
            )
        
        stats = registry.get_stats()
        
        assert stats["total_models"] == 5
        assert "catboost" in stats["algorithms"]
        assert "xgboost" in stats["algorithms"]
    
    def test_save_load_registry(self, temp_dir):
        """Test registry persistence."""
        # Create and populate registry
        registry1 = ModelRegistry(registry_dir=temp_dir / "registry")
        registry1.register(
            model_id="persistent_model",
            version="1.0.0",
            algorithm="catboost",
            training_date="2024-01-01",
            dataset_version="1.0.0",
            feature_version="1.0.0",
            metrics={"pr_auc": 0.85},
        )
        
        # Create new registry (should load from disk)
        registry2 = ModelRegistry(registry_dir=temp_dir / "registry")
        
        model = registry2.get_model("persistent_model")
        assert model is not None
        assert model.pr_auc == 0.85


# ═══════════════════════════════════════════════════════════════════════════════
# Challenger Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestChallengers:
    """Tests for challenger models."""
    
    def test_get_available_algorithms(self):
        """Test available algorithms list."""
        algorithms = get_available_algorithms()
        assert "xgboost" in algorithms
        assert "lightgbm" in algorithms
        assert "ft_transformer" in algorithms
        assert "tabnet" in algorithms
    
    def test_create_xgboost_challenger(self):
        """Test XGBoost challenger creation."""
        challenger = create_challenger("xgboost")
        assert isinstance(challenger, XGBoostChallenger)
        assert challenger.algorithm == "xgboost"
        assert not challenger.is_fitted
    
    def test_create_lightgbm_challenger(self):
        """Test LightGBM challenger creation."""
        challenger = create_challenger("lightgbm")
        assert isinstance(challenger, LightGBMChallenger)
        assert challenger.algorithm == "lightgbm"
    
    def test_invalid_algorithm(self):
        """Test invalid algorithm creation."""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            create_challenger("invalid_algorithm")
    
    def test_train_xgboost_challenger(self, sample_data):
        """Test XGBoost challenger training."""
        X, y = sample_data
        
        challenger = create_challenger("xgboost")
        challenger.fit(X, y, n_estimators=10, max_depth=3)
        
        assert challenger.is_fitted
        assert challenger.model is not None
        
        metrics = challenger.compute_metrics(X, y)
        assert metrics["pr_auc"] > 0
    
    def test_train_lightgbm_challenger(self, sample_data):
        """Test LightGBM challenger training."""
        X, y = sample_data
        
        challenger = create_challenger("lightgbm")
        challenger.fit(X, y, n_estimators=10, max_depth=3)
        
        assert challenger.is_fitted
        assert challenger.model is not None
    
    def test_challenger_predict_proba(self, sample_data):
        """Test challenger probability prediction."""
        X, y = sample_data
        
        challenger = create_challenger("xgboost")
        challenger.fit(X, y, n_estimators=10, max_depth=3)
        
        probs = challenger.predict_proba(X[:10])
        
        assert len(probs) == 10
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
    
    def test_challenger_score(self, sample_data):
        """Test challenger score method."""
        X, y = sample_data
        
        challenger = create_challenger("xgboost")
        challenger.fit(X, y, n_estimators=10, max_depth=3)
        
        scores = challenger.score(X[:10])
        
        assert len(scores) == 10
    
    def test_challenger_compute_metrics(self, sample_data):
        """Test challenger metrics computation."""
        X, y = sample_data
        
        challenger = create_challenger("xgboost")
        challenger.fit(X, y, n_estimators=10, max_depth=3)
        
        metrics = challenger.compute_metrics(X, y)
        
        assert "pr_auc" in metrics
        assert "roc_auc" in metrics
        assert "f2_score" in metrics
        assert "fpr" in metrics
        assert metrics["pr_auc"] > 0
    
    def test_challenger_get_feature_importance(self, sample_data):
        """Test challenger feature importance."""
        X, y = sample_data
        
        challenger = create_challenger("xgboost")
        challenger.fit(X, y, n_estimators=10, max_depth=3)
        
        importance = challenger.get_feature_importance(top_n=5)
        
        assert len(importance) <= 5
        assert all("feature" in imp for imp in importance)
        assert all("importance" in imp for imp in importance)
    
    def test_challenger_save_load(self, sample_data, temp_dir):
        """Test challenger save and load."""
        X, y = sample_data
        
        challenger = create_challenger("xgboost")
        challenger.fit(X, y, n_estimators=10, max_depth=3)
        
        # Save
        save_dir = temp_dir / "challenger"
        challenger.save(save_dir)
        
        # Load
        loaded = XGBoostChallenger.load(save_dir)
        
        assert loaded.is_fitted
        assert loaded.algorithm == "xgboost"
        assert loaded.pr_auc_ == challenger.pr_auc_


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluator Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelEvaluator:
    """Tests for ModelEvaluator."""
    
    def test_init(self, temp_dir):
        """Test evaluator initialization."""
        evaluator = ModelEvaluator(output_dir=temp_dir / "evaluations")
        assert evaluator.output_dir.exists()
    
    def test_evaluate_model(self, sample_data):
        """Test model evaluation."""
        X, y = sample_data
        
        # Create and train a simple model
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model(
            model, X, y,
            model_id="test_model",
            algorithm="logistic_regression",
            version="1.0.0",
            include_latency=False,
        )
        
        assert isinstance(metrics, EvaluationMetrics)
        assert metrics.model_id == "test_model"
        assert metrics.pr_auc > 0
        assert metrics.roc_auc > 0
    
    def test_compare_models(self, sample_data):
        """Test model comparison."""
        X, y = sample_data
        
        # Create two models
        from sklearn.linear_model import LogisticRegression
        
        model1 = LogisticRegression(random_state=42)
        model1.fit(X, y)
        
        model2 = LogisticRegression(random_state=43)
        model2.fit(X, y)
        
        evaluator = ModelEvaluator()
        
        metrics1 = evaluator.evaluate_model(
            model1, X, y,
            model_id="model_1",
            algorithm="logistic_regression",
            version="1.0.0",
            include_latency=False,
        )
        
        metrics2 = evaluator.evaluate_model(
            model2, X, y,
            model_id="model_2",
            algorithm="logistic_regression",
            version="1.0.0",
            include_latency=False,
        )
        
        report = evaluator.compare_models(
            champion_metrics=metrics1,
            challenger_metrics=[metrics2],
        )
        
        assert report.champion_id == "model_1"
        assert "model_2" in report.challenger_ids
    
    def test_generate_leaderboard(self, sample_data):
        """Test leaderboard generation."""
        X, y = sample_data
        
        from sklearn.linear_model import LogisticRegression
        
        evaluator = ModelEvaluator()
        metrics_list = []
        
        for i in range(3):
            model = LogisticRegression(random_state=42 + i)
            model.fit(X, y)
            
            metrics = evaluator.evaluate_model(
                model, X, y,
                model_id=f"model_{i}",
                algorithm="logistic_regression",
                version=f"1.0.{i}",
                include_latency=False,
            )
            metrics_list.append(metrics)
        
        leaderboard = evaluator.generate_leaderboard(metrics_list, sort_by="pr_auc")
        
        assert len(leaderboard) == 3
        assert leaderboard[0]["rank"] == 1
    
    def test_save_load_report(self, sample_data, temp_dir):
        """Test report save and load."""
        X, y = sample_data
        
        from sklearn.linear_model import LogisticRegression
        
        evaluator = ModelEvaluator(output_dir=temp_dir / "evaluations")
        
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        
        metrics = evaluator.evaluate_model(
            model, X, y,
            model_id="test_model",
            algorithm="logistic_regression",
            version="1.0.0",
            include_latency=False,
        )
        
        report = evaluator.compare_models(
            champion_metrics=metrics,
            challenger_metrics=[metrics],
        )
        
        # Save
        filepath = evaluator.save_report(report)
        assert filepath.exists()
        
        # Load
        loaded_report = evaluator.load_report(filepath)
        assert loaded_report.report_id == report.report_id


# ═══════════════════════════════════════════════════════════════════════════════
# Promotion Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestChampionPromoter:
    """Tests for ChampionPromoter."""
    
    def test_init(self, temp_dir):
        """Test promoter initialization."""
        registry = ModelRegistry(registry_dir=temp_dir / "registry")
        promoter = ChampionPromoter(registry=registry)
        
        assert promoter.registry == registry
        assert promoter.criteria is not None
    
    def test_promotion_criteria(self):
        """Test promotion criteria."""
        criteria = PromotionCriteria(
            min_pr_auc_improvement=0.02,
            max_fpr=0.005,
            max_calibration_error=0.03,
        )
        
        assert criteria.min_pr_auc_improvement == 0.02
        assert criteria.max_fpr == 0.005
        assert criteria.max_calibration_error == 0.03
    
    def test_should_promote(self, temp_dir):
        """Test promotion decision."""
        registry = ModelRegistry(registry_dir=temp_dir / "registry")
        promoter = ChampionPromoter(registry=registry)
        
        champion_metrics = {"pr_auc": 0.85, "fpr": 0.01, "calibration_error": 0.03}
        challenger_metrics = {"pr_auc": 0.90, "fpr": 0.005, "calibration_error": 0.02}
        
        should_promote, criteria_met, criteria_failed = promoter.should_promote(
            "challenger_1",
            champion_metrics,
            challenger_metrics,
        )
        
        assert should_promote is True
        assert "pr_auc_improvement" in criteria_met
        assert "fpr_within_limit" in criteria_met
    
    def test_should_not_promote(self, temp_dir):
        """Test when promotion should not happen."""
        registry = ModelRegistry(registry_dir=temp_dir / "registry")
        promoter = ChampionPromoter(registry=registry)
        
        champion_metrics = {"pr_auc": 0.85, "fpr": 0.01, "calibration_error": 0.03}
        challenger_metrics = {"pr_auc": 0.84, "fpr": 0.02, "calibration_error": 0.05}
        
        should_promote, criteria_met, criteria_failed = promoter.should_promote(
            "challenger_1",
            champion_metrics,
            challenger_metrics,
        )
        
        assert should_promote is False
        assert len(criteria_failed) > 0
    
    def test_get_champion_status(self, temp_dir):
        """Test champion status."""
        registry = ModelRegistry(registry_dir=temp_dir / "registry")
        promoter = ChampionPromoter(registry=registry)
        
        status = promoter.get_champion_status()
        
        assert status["has_champion"] is False
        assert status["champion_id"] is None


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """Integration tests for the full Champion-Challenger workflow."""
    
    def test_full_workflow(self, sample_data, temp_dir):
        """Test complete workflow from training to promotion."""
        X, y = sample_data
        
        # 1. Create and train champion
        from models.supervised.champion import ChampionModel
        
        champion = ChampionModel(
            iterations=10,
            depth=3,
            learning_rate=0.1,
        )
        champion.fit(X, y, feature_names=[f"f_{i}" for i in range(X.shape[1])])
        
        # 2. Train challenger
        challenger = create_challenger("xgboost")
        challenger.fit(X, y, n_estimators=10, max_depth=3)
        
        # 3. Evaluate both
        evaluator = ModelEvaluator()
        
        champion_metrics = evaluator.evaluate_model(
            champion, X, y,
            model_id="champion_1",
            algorithm="catboost",
            version="1.0.0",
            include_latency=False,
        )
        
        challenger_metrics = evaluator.evaluate_model(
            challenger, X, y,
            model_id="challenger_1",
            algorithm="xgboost",
            version="1.0.0",
            include_latency=False,
        )
        
        # 4. Compare models
        report = evaluator.compare_models(
            champion_metrics=champion_metrics,
            challenger_metrics=[challenger_metrics],
        )
        
        assert report.champion_id == "champion_1"
        assert "challenger_1" in report.challenger_ids
        
        # 5. Check promotion recommendation
        if report.promotion_recommended:
            assert report.recommended_champion == "challenger_1"
        
        # 6. Save models
        champion_dir = temp_dir / "champion"
        champion.save(champion_dir)
        
        challenger_dir = temp_dir / "challenger"
        challenger.save(challenger_dir)
        
        # 7. Load and verify
        loaded_champion = ChampionModel.load(champion_dir)
        loaded_challenger = XGBoostChallenger.load(challenger_dir)
        
        assert loaded_champion.is_fitted
        assert loaded_challenger.is_fitted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
