"""
FraudTrap — Explainability Engine
Orchestrates SHAP attributions, counterfactual explanations, and analyst-friendly formatting.
Integrates with the scoring pipeline to provide real-time, regulator-friendly explanations.
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
from loguru import logger

from models.explainability.types import (
    FullExplanation,
    SHAPExplanation,
    CounterfactualExplanation,
    ConfidenceInfo,
    FeatureAttribution,
)
from models.explainability.shap_explainer import SHAPExplainer
from models.explainability.nn_counterfactual import (
    NearestNeighborCounterfactual,
    NearestNeighborIndex,
    WeightedDistanceMetric,
)
from models.explainability.dice_counterfactual import DiCECounterfactual
from models.explainability.formatter import ExplanationFormatter
from models.explainability.cache import ExplanationCache, SHAPCache
from models.explainability.monitoring import ExplainabilityMonitor


class ExplainabilityConfig:
    """Configuration for the explainability engine."""
    
    def __init__(
        self,
        enabled: bool = True,
        shap_top_features: int = 5,
        cache_ttl_seconds: int = 1800,
        cache_max_size: int = 10_000,
        counterfactual_enabled: bool = True,
        ann_engine: str = "faiss",
        max_neighbors: int = 10,
        weighted_distance: bool = True,
        analyst_dice: bool = True,
        feature_weights: Optional[Dict[str, float]] = None,
    ):
        self.enabled = enabled
        self.shap_top_features = shap_top_features
        self.cache_ttl_seconds = cache_ttl_seconds
        self.cache_max_size = cache_max_size
        self.counterfactual_enabled = counterfactual_enabled
        self.ann_engine = ann_engine
        self.max_neighbors = max_neighbors
        self.weighted_distance = weighted_distance
        self.analyst_dice = analyst_dice
        self.feature_weights = feature_weights or {}


class ExplainabilityEngine:
    """
    Production explainability engine for FraudTrap.
    
    Orchestrates:
    - SHAP feature attribution (primary)
    - Nearest-neighbor counterfactuals (production)
    - DiCE counterfactuals (analyst investigations only)
    - Explanation formatting (analyst-friendly)
    - Caching and monitoring
    
    Performance targets:
    - SHAP: <20ms
    - Nearest Neighbor: <10ms
    - Formatting: <5ms
    - Total overhead: <40ms
    """
    
    def __init__(self, config: Optional[ExplainabilityConfig] = None):
        self.config = config or ExplainabilityConfig()
        
        self._shap = SHAPExplainer(top_features=self.config.shap_top_features)
        self._nn_counterfactual = NearestNeighborCounterfactual(
            max_neighbors=self.config.max_neighbors,
        )
        self._dice_counterfactual = DiCECounterfactual() if self.config.analyst_dice else None
        self._formatter = ExplanationFormatter(max_drivers=self.config.shap_top_features)
        self._cache = ExplanationCache(
            max_size=self.config.cache_max_size,
            ttl_seconds=self.config.cache_ttl_seconds,
        )
        self._shap_cache = SHAPCache()
        self._monitor = ExplainabilityMonitor()
        
        self._is_fitted = False
    
    def fit(
        self,
        model,
        X_background: np.ndarray,
        feature_names: List[str],
        X_legitimate: Optional[np.ndarray] = None,
        legitimate_ids: Optional[List[str]] = None,
        tenant_id: str = "shared",
    ) -> None:
        """
        Initialize all explainability components.
        
        Args:
            model: Fitted CatBoost (or compatible) model
            X_background: Background dataset for SHAP
            feature_names: Feature names in order
            X_legitimate: Legitimate transactions for counterfactual index
            legitimate_ids: Transaction IDs for legitimate transactions
            tenant_id: Tenant ID for index scoping
        """
        t_start = time.perf_counter()
        
        # Fit SHAP explainer
        self._shap.fit(model, X_background, feature_names)
        
        # Build nearest-neighbor index if legitimate data available
        if X_legitimate is not None and legitimate_ids is not None and len(X_legitimate) > 0:
            metric = WeightedDistanceMetric(self.config.feature_weights) if self.config.weighted_distance else WeightedDistanceMetric()
            
            index = NearestNeighborIndex(
                tenant_id=tenant_id,
                n_features=len(feature_names),
                feature_names=feature_names,
                distance_metric=metric,
                use_faiss=(self.config.ann_engine == "faiss"),
            )
            
            labels = np.zeros(len(X_legitimate), dtype=int)  # All legitimate
            index.fit(X_legitimate, legitimate_ids, labels)
            self._nn_counterfactual.register_index(tenant_id, index)
        
        # Fit DiCE if analyst mode enabled
        if self._dice_counterfactual is not None:
            self._dice_counterfactual.fit(model, X_background, feature_names)
        
        self._is_fitted = True
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.info("ExplainabilityEngine fitted in {:.1f}ms", elapsed_ms)
    
    def explain(
        self,
        tenant_id: str,
        transaction_id: str,
        X: np.ndarray,
        fraud_probability: float,
        feature_names: List[str],
        confidence: Optional[ConfidenceInfo] = None,
    ) -> FullExplanation:
        """
        Generate a complete explanation for a single transaction.
        
        This is the primary production entry point.
        Total latency budget: <40ms.
        
        Args:
            tenant_id: Tenant ID for isolation
            transaction_id: Transaction ID
            X: Feature vector (1, n_features) or (n_features,)
            fraud_probability: Model's fraud probability
            feature_names: Feature names
            confidence: Model confidence info
        
        Returns:
            FullExplanation with all components
        """
        t_start = time.perf_counter()
        
        # Check cache first
        cached = self._cache.get(tenant_id, transaction_id)
        if cached is not None:
            self._monitor.record_explanation(
                shap_latency_ms=0.0,
                total_latency_ms=(time.perf_counter() - t_start) * 1000,
                cache_hit=True,
            )
            return cached
        
        X_input = X.reshape(1, -1) if X.ndim == 1 else X
        
        # SHAP explanation
        shap_result = self._explain_shap(X_input, feature_names)
        
        # Counterfactual explanation
        cf_result = None
        if self.config.counterfactual_enabled:
            cf_result = self._explain_counterfactual(
                tenant_id, X_input, transaction_id, fraud_probability, feature_names
            )
        
        # Format report
        formatted = self._formatter.format(
            fraud_probability=fraud_probability,
            shap=shap_result,
            counterfactual=cf_result,
            confidence=confidence,
        )
        
        total_latency_ms = (time.perf_counter() - t_start) * 1000
        
        explanation = FullExplanation(
            transaction_id=transaction_id,
            tenant_id=tenant_id,
            fraud_probability=fraud_probability,
            shap=shap_result,
            counterfactual=cf_result,
            formatted=formatted,
            confidence=confidence,
            total_latency_ms=round(total_latency_ms, 2),
        )
        
        # Cache the result
        self._cache.put(tenant_id, transaction_id, explanation)
        
        # Record monitoring
        self._monitor.record_explanation(
            shap_latency_ms=shap_result.latency_ms if shap_result else 0.0,
            total_latency_ms=total_latency_ms,
            counterfactual_latency_ms=cf_result.latency_ms if cf_result else None,
            cache_hit=False,
            counterfactual_success=cf_result is not None,
            fraud_drivers=[f"{a.feature} is {a.value}" for a in (shap_result.top_features if shap_result else [])],
            counterfactual_features=[c.feature for c in (cf_result.changes if cf_result else [])],
        )
        
        return explanation
    
    def explain_batch(
        self,
        tenant_id: str,
        transaction_ids: List[str],
        X: np.ndarray,
        fraud_probabilities: np.ndarray,
        feature_names: List[str],
        confidence: Optional[ConfidenceInfo] = None,
    ) -> List[FullExplanation]:
        """Generate explanations for a batch of transactions."""
        return [
            self.explain(
                tenant_id=tenant_id,
                transaction_id=tids,
                X=X[i:i+1],
                fraud_probability=float(fraud_probabilities[i]),
                feature_names=feature_names,
                confidence=confidence,
            )
            for i, tids in enumerate(transaction_ids)
        ]
    
    def generate_counterfactual(
        self,
        tenant_id: str,
        transaction_id: str,
        X: np.ndarray,
        fraud_probability: float,
        model=None,
        features_to_vary: Optional[List[str]] = None,
    ) -> Optional[CounterfactualExplanation]:
        """
        Generate a DiCE counterfactual for analyst investigation.
        
        This is NEVER called during production inference.
        It is exposed via the analyst API for deep investigations.
        """
        if self._dice_counterfactual is None:
            return None
        
        X_input = X.reshape(1, -1) if X.ndim == 1 else X
        
        cf = self._dice_counterfactual.generate(
            X_input,
            features_to_vary=features_to_vary,
        )
        
        if cf is None and model is not None:
            cf = self._dice_counterfactual.generate_with_model(
                model, X_input, fraud_probability
            )
        
        return cf
    
    def nearest_legitimate(
        self,
        tenant_id: str,
        X: np.ndarray,
        k: int = 1,
    ) -> Optional[List[Dict[str, Any]]]:
        """Find k nearest legitimate transactions for a given point."""
        index = self._nn_counterfactual.get_index(tenant_id)
        if index is None:
            return None
        
        X_input = X.reshape(1, -1) if X.ndim == 1 else X
        neighbors = index.query(X_input, k=k)
        
        if not neighbors or not neighbors[0]:
            return None
        
        return [
            {"transaction_id": n.transaction_id, "distance": n.distance, "features": n.features}
            for n in neighbors[0]
        ]
    
    def explain_features(
        self,
        tenant_id: str,
        X: np.ndarray,
        feature_names: List[str],
    ) -> List[Dict[str, Any]]:
        """Get SHAP values for all features (for detailed analysis)."""
        if not self._shap._is_fitted:
            return []
        
        X_input = X.reshape(1, -1) if X.ndim == 1 else X
        shap_result = self._shap.explain(X_input, top_n=len(feature_names))
        
        return [
            {"feature": attr.feature, "value": attr.value, "impact": attr.impact, "direction": attr.direction}
            for attr in shap_result.top_features
        ]
    
    def format_report(
        self,
        fraud_probability: float,
        shap: Optional[SHAPExplanation] = None,
        counterfactual: Optional[CounterfactualExplanation] = None,
        confidence: Optional[ConfidenceInfo] = None,
    ) -> Dict[str, Any]:
        """Format a report from raw explanation components."""
        report = self._formatter.format(fraud_probability, shap, counterfactual, confidence)
        return report.to_dict()
    
    def _explain_shap(self, X: np.ndarray, feature_names: List[str]) -> SHAPExplanation:
        """SHAP explanation with error handling."""
        try:
            return self._shap.explain(X)
        except Exception as exc:
            logger.warning("SHAP explanation failed: {}", exc)
            self._monitor.record_error()
            return SHAPExplanation(
                fraud_probability=0.5,
                base_value=0.0,
                top_features=(),
            )
    
    def _explain_counterfactual(
        self,
        tenant_id: str,
        X: np.ndarray,
        transaction_id: str,
        fraud_probability: float,
        feature_names: List[str],
    ) -> Optional[CounterfactualExplanation]:
        """Counterfactual explanation with error handling."""
        try:
            return self._nn_counterfactual.explain(
                tenant_id=tenant_id,
                X=X,
                transaction_id=transaction_id,
                fraud_probability=fraud_probability,
                feature_names=feature_names,
            )
        except Exception as exc:
            logger.warning("Counterfactual explanation failed: {}", exc)
            self._monitor.record_error()
            return None
    
    @property
    def monitor(self) -> ExplainabilityMonitor:
        return self._monitor
    
    @property
    def cache(self) -> ExplanationCache:
        return self._cache
    
    def save(self, path: Path) -> None:
        """Persist all explainability artifacts."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save SHAP explainer
        self._shap.save(path / "shap")
        
        # Save NN index
        self._nn_counterfactual.save(path / "nn_index")
        
        logger.info("ExplainabilityEngine saved to {}", path)
    
    @classmethod
    def load(cls, path: Path, config: Optional[ExplainabilityConfig] = None) -> "ExplainabilityEngine":
        """Load a persisted explainability engine."""
        path = Path(path)
        
        engine = cls(config=config)
        
        # Load SHAP
        shap_path = path / "shap"
        if shap_path.exists():
            engine._shap = SHAPExplainer.load(shap_path)
        
        # Load NN index
        nn_path = path / "nn_index"
        if nn_path.exists():
            engine._nn_counterfactual.load(nn_path)
        
        engine._is_fitted = True
        logger.info("ExplainabilityEngine loaded from {}", path)
        return engine
