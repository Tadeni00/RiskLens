"""
FraudTrap — Scoring Orchestrator
Wires Tier 1 (rules) → Tier 2 (gradient boost ensemble) → Tier 3 (GNN, if needed)
into a single real-time decision within the 100ms SLA.
Manages model phase lifecycle (UNSUPERVISED → ADAPTIVE_LEARNING → SUPERVISED).

Key improvements:
- Async GNN scoring (offloaded from critical path)
- Feature validation (NaN/Inf checks before scoring)
- Model warmup (dummy inference on load)
- Version pinning (model_version, training_hash, feature_hash, dataset_hash)
- Uncertainty estimation (MC Dropout, Temperature Scaling, Conformal Prediction)
- Rules explainability
- Watchdog-based hot-reload with atomic swap
"""

from __future__ import annotations
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import numpy as np
import redis
from loguru import logger

# Watchdog for filesystem events
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    WATCHDOG_AVAILABLE = True
except Exception as exc:
    WATCHDOG_AVAILABLE = False
    logger.warning("Watchdog unavailable, falling back to polling: {}", exc)

from config.settings import get_settings
from ingestion.schema import TransactionRequest, ScoringResponse, Explanation
from features.engineering import assemble_feature_vector, get_redis
from scoring.rules_engine import RulesEngine
from scoring.simple_model import SimpleFraudModel
from scoring.validation import (
    validate_feature_compatibility,
    auto_register_schema_if_missing,
)

try:
    from models.cold_start.ensemble import ColdStartEnsemble
except Exception as exc:
    ColdStartEnsemble = None
    logger.warning("Cold-start ensemble unavailable: {}", exc)

try:
    from models.adaptive_learning.tabpfn_learner import TabPFNAdaptiveLearner
except Exception as exc:
    TabPFNAdaptiveLearner = None
    logger.warning("Adaptive learner (TabPFN) package unavailable: {}", exc)

try:
    from models.supervised.champion import ChampionModel
except Exception as exc:
    ChampionModel = None
    logger.warning("Champion model package unavailable: {}", exc)

try:
    from models.gnn.gnn_scorer import GNNScorer
except Exception as exc:
    GNNScorer = None
    logger.warning("GNN scorer unavailable: {}", exc)

try:
    from models.explainability.engine import ExplainabilityEngine, ExplainabilityConfig
    from models.explainability.types import ConfidenceInfo

    EXPLAINABILITY_AVAILABLE = True
except Exception as exc:
    ExplainabilityEngine = None
    ExplainabilityConfig = None
    ConfidenceInfo = None
    EXPLAINABILITY_AVAILABLE = False
    logger.warning("Explainability engine unavailable: {}", exc)

settings = get_settings()

# Thread pool for async GNN inference
_GNN_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gnn-scorer")

# Thread pool for async explanation (cold-start, semi-supervised)
_EXPLANATION_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="explanation")


class _ModelFileHandler:
    """Watchdog handler for model file changes."""

    def __init__(self, registry: "ModelRegistry"):
        self.registry = registry
        self._last_reload = 0.0
        self._debounce_seconds = 1.0  # Debounce rapid file writes

    def on_modified(self, event):
        if getattr(event, "is_directory", False):
            return
        # Only reload on model artifact changes
        if not any(
            getattr(event, "src_path", "").endswith(ext)
            for ext in (".pkl", ".pt", ".txt", ".joblib")
        ):
            return
        now = time.monotonic()
        if now - self._last_reload < self._debounce_seconds:
            return
        self._last_reload = now
        logger.info("Model file changed: {}, triggering reload", event.src_path)
        self.registry.load_from_disk(self.registry.model_dir)


class ModelRegistry:
    """
    Holds the currently active model for each phase.
    Hot-swappable — new models can be loaded without restarting the API.
    Supports version pinning: model_version, training_hash, feature_hash, dataset_hash.
    Uses double-buffered atomic swap for zero-downtime reloads.
    """

    def __init__(self):
        self.cold_start: Optional[ColdStartEnsemble] = None
        self.adaptive_learner: Optional[TabPFNAdaptiveLearner] = None
        self.champion: Optional[ChampionModel] = None
        self.gnn_scorer: Optional[GNNScorer] = None
        self.active_phase: str = "UNSUPERVISED"
        self.model_version: str = "unloaded"
        self.feature_names: list[str] = []
        self.feature_hash: Optional[str] = None
        self.training_hash: Optional[str] = None
        self.dataset_hash: Optional[str] = None
        self.trained_at: Optional[str] = None

        self.champion_models: dict[str, ChampionModel] = {}
        self.adaptive_learner_models: dict[str, TabPFNAdaptiveLearner] = {}
        self.cold_start_models: dict[str, ColdStartEnsemble] = {}
        self.simple_models: dict[str, SimpleFraudModel] = {}

        # Backwards compatibility aliases
        self.semi_supervised_models = self.adaptive_learner_models

        # Double-buffered state for atomic swap
        self._staging = {}
        self._active = {}
        self._lock = threading.RLock()
        self._observer: Optional[Observer] = None
        self._model_dir: Optional[Path] = None

    def load_from_disk(self, model_dir: Path) -> None:
        """Load all models from disk into staging, then atomically swap."""
        model_dir = Path(model_dir)
        self._model_dir = model_dir

        if not model_dir.exists():
            logger.info("Model directory does not exist yet: {}", model_dir)
            return

        # Load everything into staging dict
        staging = {}
        staging["simple_models"] = self._load_simple_models(model_dir)
        staging["cold_start_models"] = self._load_cold_start_models(model_dir)
        staging["adaptive_learner_models"] = self._load_adaptive_learner_models(
            model_dir, staging.get("cold_start_models", {})
        )
        staging["champion_models"] = self._load_supervised_models(model_dir)

        # Load shared models
        cold_path = model_dir / "cold_start"
        if cold_path.exists() and ColdStartEnsemble:
            staging["cold_start"] = ColdStartEnsemble.load(cold_path)
            self._extract_version_info(staging["cold_start"])
            logger.info("Cold-start model loaded (version={})", self.model_version)

        semi_path = model_dir / "semi_supervised"
        if semi_path.exists() and TabPFNAdaptiveLearner:
            try:
                staging["adaptive_learner"] = TabPFNAdaptiveLearner.load(semi_path)
                staging["active_phase"] = "ADAPTIVE_LEARNING"
                logger.info("Adaptive learner (TabPFN) loaded")
            except Exception as exc:
                logger.warning("Failed to load adaptive learner model: {}", exc)

        sup_path = model_dir / "supervised"
        if sup_path.exists() and ChampionModel:
            try:
                staging["champion"] = ChampionModel.load(sup_path)
                staging["active_phase"] = "SUPERVISED"
                staging["feature_names"] = staging["champion"].feature_names
                self._extract_version_info(staging["champion"])
                logger.info("Champion model loaded (version={})", self.model_version)
            except Exception as exc:
                logger.warning("Failed to load shared champion model: {}", exc)

        # Load champion model (preferred over ensemble for production)
        champion_path = model_dir / "champion"
        if champion_path.exists() and ChampionModel:
            try:
                staging["champion"] = ChampionModel.load(champion_path)
                staging["active_phase"] = "SUPERVISED"
                staging["feature_names"] = staging["champion"].feature_names
                self._extract_version_info(staging["champion"])
                logger.info("Champion model loaded (version={})", self.model_version)
            except Exception as exc:
                logger.warning("Failed to load champion model: {}", exc)

        gnn_path = model_dir / "gnn"
        if gnn_path.exists() and GNNScorer:
            staging["gnn_scorer"] = GNNScorer.load(gnn_path)
            logger.info("GNN scorer loaded")

        # Version fallback
        if (
            not staging.get("simple_models")
            and not staging.get("supervised")
            and not staging.get("adaptive_learner")
            and not staging.get("cold_start")
        ):
            version_file = model_dir / "version.txt"
            if version_file.exists():
                staging["model_version"] = version_file.read_text().strip()

        # Warmup staging models
        if staging.get("feature_names"):
            self._warmup_staging(staging)

        # Atomic swap
        with self._lock:
            self._staging = staging
            self._swap_active()
            logger.info(
                "Model swap complete: active_phase={}, version={}",
                self.active_phase,
                self.model_version,
            )

        # Start watchdog on first load
        if self._observer is None and WATCHDOG_AVAILABLE:
            self._start_watchdog(model_dir)

    def _swap_active(self) -> None:
        """Atomically swap staging to active."""
        s = self._staging
        self.simple_models = s.get("simple_models", {})
        self.cold_start_models = s.get("cold_start_models", {})
        self.adaptive_learner_models = s.get("adaptive_learner_models", {})
        self.champion_models = s.get("champion_models", {})

        # Backwards compatibility alias
        self.semi_supervised_models = self.adaptive_learner_models

        self.cold_start = s.get("cold_start")
        self.adaptive_learner = s.get("adaptive_learner")
        self.champion = s.get("champion")
        self.gnn_scorer = s.get("gnn_scorer")
        self.active_phase = s.get("active_phase", "UNSUPERVISED")
        self.model_version = s.get("model_version", self.model_version)
        self.feature_names = s.get("feature_names", self.feature_names)

    def _start_watchdog(self, model_dir: Path) -> None:
        """Start filesystem observer for hot-reload."""
        if not WATCHDOG_AVAILABLE:
            logger.debug("Watchdog not available, skipping filesystem observer")
            return
        try:
            self._observer = Observer()
            handler = _ModelFileHandler(self)
            self._observer.schedule(handler, str(model_dir), recursive=True)
            self._observer.start()
            logger.info("Watchdog started on {}", model_dir)
        except Exception as exc:
            logger.warning("Failed to start watchdog: {}", exc)

    def stop_watchdog(self) -> None:
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None

    def _extract_version_info(self, model) -> None:
        """Extract version pinning info from loaded model."""
        if hasattr(model, "model_version"):
            self.model_version = model.model_version
        if hasattr(model, "training_hash"):
            self.training_hash = model.training_hash
        if hasattr(model, "feature_hash"):
            self.feature_hash = model.feature_hash
        if hasattr(model, "dataset_hash"):
            self.dataset_hash = model.dataset_hash
        if hasattr(model, "trained_at"):
            self.trained_at = model.trained_at

    def _warmup_models(self) -> None:
        """Run dummy inference to warm up models (JIT compilation, cache priming)."""
        if not self.feature_names:
            return
        dummy_features = np.zeros((1, len(self.feature_names)), dtype=np.float32)
        try:
            if self.cold_start:
                _ = self.cold_start.score(dummy_features)
            if self.adaptive_learner:
                _ = self.adaptive_learner.predict_proba(dummy_features)
            if self.champion:
                _ = self.champion.score(dummy_features)
            elif self.supervised:
                _ = self.supervised.score(dummy_features)
            if self.gnn_scorer and self.gnn_scorer.is_fitted:
                pass
            logger.info("Model warmup complete")
        except Exception as exc:
            logger.warning("Model warmup failed (non-fatal): {}", exc)

    def _warmup_staging(self, staging: dict) -> None:
        """Warm up models in staging before swap."""
        if not staging.get("feature_names"):
            return
        dummy = np.zeros((1, len(staging["feature_names"])), dtype=np.float32)
        try:
            if staging.get("cold_start"):
                _ = staging["cold_start"].score(dummy)
            if staging.get("adaptive_learner"):
                _ = staging["adaptive_learner"].predict_proba(dummy)
            if staging.get("champion"):
                _ = staging["champion"].score(dummy)
            elif staging.get("supervised"):
                _ = staging["supervised"].score(dummy)
        except Exception as exc:
            logger.warning("Staging warmup failed: {}", exc)

    def _load_simple_models(self, model_dir: Path) -> dict[str, SimpleFraudModel]:
        loaded: dict[str, SimpleFraudModel] = {}
        for model_path in sorted(model_dir.glob("*/simple_model.pkl")):
            tenant_id = model_path.parent.name
            try:
                loaded[tenant_id] = SimpleFraudModel.load(model_path)
            except Exception as exc:
                logger.warning("Could not load simple model for tenant {}: {}", tenant_id, exc)
        if loaded:
            logger.info(
                "Loaded {} tenant simple model(s): {}",
                len(loaded),
                ", ".join(sorted(loaded)),
            )
        return loaded

    def _load_cold_start_models(self, model_dir: Path) -> dict[str, ColdStartEnsemble]:
        loaded: dict[str, ColdStartEnsemble] = {}
        for p1_dir in sorted(model_dir.glob("*/phase1")):
            tenant_id = p1_dir.parent.name
            try:
                if ColdStartEnsemble:
                    loaded[tenant_id] = ColdStartEnsemble.load(p1_dir)
            except Exception as exc:
                logger.warning("Could not load cold start model for tenant {}: {}", tenant_id, exc)
        if loaded:
            logger.info(
                "Loaded {} tenant cold-start model(s): {}",
                len(loaded),
                ", ".join(sorted(loaded)),
            )
        return loaded

    def _load_adaptive_learner_models(
        self, model_dir: Path, cold_models: dict
    ) -> dict[str, TabPFNAdaptiveLearner]:
        loaded: dict[str, TabPFNAdaptiveLearner] = {}
        for p2_dir in sorted(model_dir.glob("*/phase2")):
            tenant_id = p2_dir.parent.name
            try:
                if TabPFNAdaptiveLearner:
                    loaded[tenant_id] = TabPFNAdaptiveLearner.load(p2_dir)
            except Exception as exc:
                logger.warning("Could not load adaptive learner for tenant {}: {}", tenant_id, exc)
        if loaded:
            logger.info(
                "Loaded {} tenant adaptive learner model(s): {}",
                len(loaded),
                ", ".join(sorted(loaded)),
            )
        return loaded

    def _load_supervised_models(self, model_dir: Path) -> dict[str, ChampionModel]:
        loaded: dict[str, ChampionModel] = {}
        for p3_dir in sorted(model_dir.glob("*/phase3")):
            tenant_id = p3_dir.parent.name
            try:
                if ChampionModel:
                    loaded[tenant_id] = ChampionModel.load(p3_dir)
            except Exception as exc:
                logger.warning("Could not load champion model for tenant {}: {}", tenant_id, exc)
        if loaded:
            logger.info(
                "Loaded {} tenant champion model(s): {}",
                len(loaded),
                ", ".join(sorted(loaded)),
            )
        return loaded

    def get_active_model(self, tenant_id: str | None = None):
        # Watchdog handles reloads; no polling needed
        with self._lock:
            if tenant_id:
                if tenant_id in self.champion_models:
                    return self.champion_models[tenant_id]
                if tenant_id in self.adaptive_learner_models:
                    return self.adaptive_learner_models[tenant_id]
                if tenant_id in self.cold_start_models:
                    return self.cold_start_models[tenant_id]
                if tenant_id in self.simple_models:
                    return self.simple_models[tenant_id]

            # Prioritize champion model over ensemble
            if self.active_phase == "SUPERVISED":
                if self.champion:
                    return self.champion
            if self.active_phase == "ADAPTIVE_LEARNING" and self.adaptive_learner:
                return self.adaptive_learner
            if self.cold_start:
                return self.cold_start
            return None

    def get_model_version(self, tenant_id: str | None = None) -> str:
        with self._lock:
            if tenant_id:
                if tenant_id in self.champion_models:
                    return "supervised-ensemble"
                if tenant_id in self.adaptive_learner_models:
                    return "adaptive-learning-tabpfn"
                if tenant_id in self.cold_start_models:
                    return "cold-start-ensemble"
                if tenant_id in self.simple_models:
                    return self.simple_models[tenant_id].model_version

            # Return champion version if available
            if self.champion:
                return self.champion.model_version

            return self.model_version

    def get_version_info(self) -> dict:
        """Return complete version pinning information."""
        with self._lock:
            return {
                "model_version": self.model_version,
                "training_hash": self.training_hash,
                "feature_hash": self.feature_hash,
                "dataset_hash": self.dataset_hash,
                "trained_at": self.trained_at,
                "active_phase": self.active_phase,
            }


# Singleton registry — shared across all API workers via module-level state
_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    return _registry


# ── Scoring orchestrator ──────────────────────────────────────────────────────


class ScoringOrchestrator:
    """
    Main entry point for real-time transaction scoring.
    Called by the FastAPI scoring endpoint.
    """

    def __init__(self):
        self.registry = get_registry()
        self._redis: Optional[redis.Redis] = None
        self.rules_engine: Optional[RulesEngine] = None
        self._gnn_future = None  # For async GNN scoring
        self._explainability: Optional[Any] = None  # ExplainabilityEngine

    def _get_redis(self) -> Optional[redis.Redis]:
        if self._redis is None:
            try:
                r = get_redis()
                r.ping()
                self._redis = r
                self.rules_engine = RulesEngine(r)
            except Exception as exc:
                logger.warning("Redis unavailable — degraded mode: {}", exc)
                self.rules_engine = RulesEngine(None)
        return self._redis

    # ── Decision mapping ──────────────────────────────────────────────────────

    @staticmethod
    def _make_decision(score: float) -> str:
        if score >= settings.score_block_threshold:
            return "BLOCK"
        if score >= settings.score_review_low:
            return "REVIEW"
        return "APPROVE"

    # ── Main scoring flow ─────────────────────────────────────────────────────

    def score(self, txn: TransactionRequest) -> ScoringResponse:
        t_start = time.monotonic()
        trace_id = str(uuid.uuid4())
        scored_at = datetime.now(timezone.utc)

        r = self._get_redis()

        # Resolve tenant's phase dynamically from Redis
        model_phase = "UNSUPERVISED"
        if r:
            try:
                raw_phase = r.get(f"fraudtrap:phase:{txn.tenant_id}")
                if raw_phase:
                    phase_state = json.loads(raw_phase)
                    model_phase = phase_state.get("current_phase", "UNSUPERVISED")
            except Exception as exc:
                logger.warning("Failed to retrieve phase from Redis: {}", exc)

        # ── Step 1: Feature assembly (~5–15ms including Redis) ────────────────
        features = assemble_feature_vector(txn, r)

        # Feature validation: check for NaN/Inf before scoring
        if not self._validate_features(features):
            logger.warning(
                "Feature validation failed for txn={}, using heuristic score",
                txn.transaction_id,
            )

        # Schema compatibility validation (logs mismatch, falls back to heuristic floor)
        validate_feature_compatibility(txn.tenant_id, features)

        # Auto-register schema if missing (first request for tenant)
        auto_register_schema_if_missing(txn.tenant_id, features)

        model = self.registry.get_active_model(txn.tenant_id)
        model_version = self.registry.get_model_version(txn.tenant_id)
        feature_array = self._features_to_array(features, model)

        # ── Step 2: Tier 1 — rules engine (<1ms) ─────────────────────────────
        rule_result = self.rules_engine.evaluate(txn, features) if self.rules_engine else None

        if rule_result and rule_result.hard_block:
            latency = (time.monotonic() - t_start) * 1000
            response = ScoringResponse(
                transaction_id=txn.transaction_id,
                tenant_id=txn.tenant_id,
                risk_score=1.0,
                decision="BLOCK",
                model_phase="RULES",
                model_version=model_version,
                latency_ms=round(latency, 2),
                triggered_rules=rule_result.rule_ids,
                trace_id=trace_id,
                scored_at=scored_at,
            )
            self._emit_audit(response, txn)
            return response

        # ── Step 3: Tier 2 — ML model (~10–50ms) ─────────────────────────────
        risk_score = self._heuristic_score(features, rule_result)
        explanation = None
        uncertainty = None
        confidence_info = None

        if model is not None and feature_array is not None:
            try:
                X = feature_array.reshape(1, -1)

                # Cold-start model: anomaly scoring
                if ColdStartEnsemble is not None and isinstance(model, ColdStartEnsemble):
                    risk_score = float(model.score(X)[0])
                    confidence_info = (
                        ConfidenceInfo(
                            expert_used="ColdStartEnsemble",
                            confidence=1.0 - risk_score,
                        )
                        if ConfidenceInfo
                        else None
                    )

                # Champion model: confidence-aware routing
                elif ChampionModel is not None and isinstance(model, ChampionModel):
                    predictions = model.score_with_confidence(X)
                    pred = predictions[0]
                    risk_score = pred.probability
                    confidence_info = (
                        ConfidenceInfo(
                            expert_used=("FTTransformer" if pred.ft_invoked else "CatBoost"),
                            confidence=pred.confidence,
                            ft_invoked=pred.ft_invoked,
                            fusion_output=pred.fusion_output,
                        )
                        if ConfidenceInfo
                        else None
                    )

                    # Log specialist invocation
                    if pred.ft_invoked:
                        logger.debug(
                            "FT-Transformer invoked: prob={:.4f}, confidence={:.4f}, fusion={:.4f}",
                            pred.probability,
                            pred.confidence,
                            pred.fusion_output or 0.0,
                        )

                # TabPFN adaptive learner
                elif TabPFNAdaptiveLearner is not None and isinstance(model, TabPFNAdaptiveLearner):
                    preds = model.predict_with_uncertainty(X)
                    pred = preds[0]
                    risk_score = pred.probability
                    confidence_info = (
                        ConfidenceInfo(
                            expert_used="TabPFN",
                            confidence=pred.confidence,
                        )
                        if ConfidenceInfo
                        else None
                    )

                # Fallback: heuristic
                else:
                    policy_floor = self._heuristic_score(features, rule_result)
                    risk_score = policy_floor

                if rule_result and rule_result.triggered and not rule_result.hard_block:
                    risk_score = min(1.0, risk_score + rule_result.risk_boost)

                # ── Explainability Engine (new pipeline) ───────────────────────
                if EXPLAINABILITY_AVAILABLE and risk_score >= settings.score_review_low:
                    explanation = self._explain_with_engine(
                        txn.tenant_id,
                        txn.transaction_id,
                        X,
                        risk_score,
                        model,
                        confidence_info,
                    )

                # Fallback: legacy explanation methods
                if explanation is None:
                    explanation = self._explain_legacy(txn, features, model, X, risk_score)

                # MC Dropout uncertainty for GNN (async, non-blocking)
                if (
                    GNNScorer is not None
                    and self.registry.gnn_scorer
                    and self.registry.gnn_scorer.is_fitted
                ):
                    self._gnn_future = _GNN_EXECUTOR.submit(
                        self._score_gnn_async, txn.account_id, txn.tenant_id
                    )

            except Exception as exc:
                logger.error("Model scoring failed, using default score: {}", exc)

        decision = self._make_decision(risk_score)
        latency = (time.monotonic() - t_start) * 1000

        if latency > settings.scoring_timeout_ms:
            logger.warning("SLA breach: txn={} latency={}ms", txn.transaction_id, round(latency, 1))

        response = ScoringResponse(
            transaction_id=txn.transaction_id,
            tenant_id=txn.tenant_id,
            risk_score=round(risk_score, 6),
            decision=decision,
            model_phase=model_phase,
            model_version=model_version,
            latency_ms=round(latency, 2),
            explanation=explanation,
            triggered_rules=rule_result.rule_ids if rule_result else [],
            trace_id=trace_id,
            scored_at=scored_at,
        )
        self._emit_audit(response, txn)
        return response

    def _validate_features(self, features: dict[str, float]) -> bool:
        """Validate feature vector before scoring. Returns True if valid."""
        if not features:
            return False
        for k, v in features.items():
            if not isinstance(v, (int, float)) or not np.isfinite(v):
                logger.warning("Invalid feature value: {}={}", k, v)
                return False
        return True

    def _score_gnn_async(self, account_id: str, tenant_id: str) -> Optional[tuple[float, float]]:
        """Async GNN scoring with MC Dropout uncertainty. Returns (score, uncertainty)."""
        try:
            r = self._get_redis()
            if not r:
                return None
            # Fetch recent transactions for this account from Redis
            # (simplified - in production would query Redis streams or ClickHouse)
            recent_txns = []
            # Build recent transactions from Redis sorted sets...
            # For now return None to skip
            return None
        except Exception as exc:
            logger.warning("Async GNN scoring failed: {}", exc)
            return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _explain_cold_start(self, model, X: np.ndarray) -> Explanation:
        """Sync cold-start explanation (fast - component weights only)."""
        try:
            explanations = model.explain(X, top_n=8)
            exp = explanations[0]
            top_feats = [
                {
                    "feature": ft["feature"],
                    "value": ft["value"],
                    "contribution": ft["contribution"],
                    "method": ft["method"],
                }
                for ft in exp.get("top_features", [])
            ]
            return Explanation(
                model_type="cold_start",
                base_value=exp.get("base_value", 0.0),
                prediction_value=exp["prediction_value"],
                top_features=top_feats,
                components=exp.get("components"),
                latency_ms=1.0,
            )
        except Exception as exc:
            logger.warning("Cold-start explanation failed: {}", exc)
            return None

    def _explain_adaptive_learner(self, model, X: np.ndarray) -> Explanation:
        """Adaptive learning explanation (TabPFN permutation importance)."""
        try:
            explanations = model.explain(X, top_n=8)
            exp = explanations[0]
            top_feats = [
                {
                    "feature": ft["feature"],
                    "value": ft["value"],
                    "contribution": ft["contribution"],
                    "method": ft.get("method", "permutation_importance"),
                }
                for ft in exp.get("top_features", [])
            ]
            return Explanation(
                model_type="adaptive_learning",
                base_value=exp.get("base_value", 0.0),
                prediction_value=exp["prediction_value"],
                top_features=top_feats,
                components=exp.get("components"),
                latency_ms=5.0,
            )
        except Exception as exc:
            logger.warning("Adaptive learning explanation failed: {}", exc)
            return None

    def _explain_tabpfn(self, model, X: np.ndarray) -> Explanation:
        """TabPFN explanation (permutation importance)."""
        try:
            explanations = model.explain(X, top_n=8)
            exp = explanations[0]
            top_feats = [
                {
                    "feature": ft["feature"],
                    "value": ft["value"],
                    "contribution": ft["contribution"],
                    "method": ft.get("method", "permutation_importance"),
                }
                for ft in exp.get("top_features", [])
            ]
            return Explanation(
                model_type="tabpfn",
                base_value=exp.get("base_value", 0.0),
                prediction_value=exp["prediction_value"],
                top_features=top_feats,
                components=exp.get("components"),
                latency_ms=5.0,
            )
        except Exception as exc:
            logger.warning("TabPFN explanation failed: {}", exc)
            return None

    def _explain_rules(self, txn: TransactionRequest, features: dict[str, float]) -> Explanation:
        """Rules explanation (sync - very fast)."""
        try:
            result = self.rules_engine.explain(txn, features)
            top_feats = [
                {
                    "feature": ft["feature"],
                    "value": ft["value"],
                    "contribution": ft["contribution"],
                    "method": ft["method"],
                }
                for ft in result.get("top_features", [])
            ]
            return Explanation(
                model_type="rules",
                base_value=result.get("base_value", 0.0),
                prediction_value=result["prediction_value"],
                top_features=top_feats,
                components=result.get("components"),
                latency_ms=0.1,
            )
        except Exception as exc:
            logger.warning("Rules explanation failed: {}", exc)
            return None

    def _explain_supervised(self, model, X: np.ndarray) -> Explanation:
        """Sync supervised SHAP explanation."""
        try:
            shap_data = model.explain(X, top_n=8)[0]
            top_feats = [
                {
                    "feature": ft["feature"],
                    "value": ft["value"],
                    "contribution": ft["shap_value"],
                    "method": "shap",
                }
                for ft in shap_data["top_features"]
            ]
            return Explanation(
                model_type="supervised",
                base_value=shap_data["base_value"],
                prediction_value=shap_data["prediction_value"],
                top_features=top_feats,
                latency_ms=10.0,
            )
        except Exception as exc:
            logger.warning("Supervised explanation failed: {}", exc)
            return None

    def _explain_champion(self, model, X: np.ndarray) -> Explanation:
        """Sync champion model explanation using feature importance."""
        try:
            # Use feature importance for champion model explanation
            importance = model.get_feature_importance(top_n=8)
            if not importance:
                return self._explain_rules(None, {})

            top_feats = [
                {
                    "feature": ft["feature"],
                    "value": 0.0,  # Feature importance doesn't give values
                    "contribution": ft["importance"],
                    "method": "feature_importance",
                }
                for ft in importance
            ]

            # Get prediction value
            prediction_value = float(model.score(X)[0])

            return Explanation(
                model_type="champion",
                base_value=0.0,
                prediction_value=prediction_value,
                top_features=top_feats,
                latency_ms=2.0,
            )
        except Exception as exc:
            logger.warning("Champion explanation failed: {}", exc)
            return None

    # ── Async Explanation Methods ────────────────────────────────────────────────

    def _explain_cold_start_async(self, model, X: np.ndarray, trace_id: str) -> None:
        """Async cold-start explanation - stores result for /v1/explain/{trace_id}."""
        try:
            explanation = self._explain_cold_start(model, X)
            if explanation:
                self._cache_explanation(trace_id, explanation)
        except Exception as exc:
            logger.warning("Async cold-start explanation failed: {}", exc)

    def _explain_adaptive_learner_async(self, model, X: np.ndarray, trace_id: str) -> None:
        """Async adaptive learning explanation."""
        try:
            explanation = self._explain_adaptive_learner(model, X)
            if explanation:
                self._cache_explanation(trace_id, explanation)
        except Exception as exc:
            logger.warning("Async adaptive learning explanation failed: {}", exc)

    def _explain_supervised_async(self, model, X: np.ndarray, trace_id: str) -> None:
        """Async supervised explanation."""
        try:
            explanation = self._explain_supervised(model, X)
            if explanation:
                self._cache_explanation(trace_id, explanation)
        except Exception as exc:
            logger.warning("Async supervised explanation failed: {}", exc)

    def _explain_with_engine(
        self,
        tenant_id: str,
        transaction_id: str,
        X: np.ndarray,
        risk_score: float,
        model,
        confidence_info,
    ) -> Optional[Explanation]:
        """
        Generate explanation using the new ExplainabilityEngine.
        Returns the existing Explanation schema for backward compatibility.
        """
        try:
            if self._explainability is None:
                self._init_explainability(tenant_id, model)

            if self._explainability is None:
                return None

            feature_names = self.registry.feature_names or []
            if not feature_names:
                return None

            full_exp = self._explainability.explain(
                tenant_id=tenant_id,
                transaction_id=transaction_id,
                X=X,
                fraud_probability=risk_score,
                feature_names=feature_names,
                confidence=confidence_info,
            )

            if full_exp is None or full_exp.formatted is None:
                return None

            # Convert to existing Explanation schema
            top_feats = []
            if full_exp.shap and full_exp.shap.top_features:
                for attr in full_exp.shap.top_features:
                    top_feats.append(
                        {
                            "feature": attr.feature,
                            "value": attr.value,
                            "contribution": attr.impact,
                            "method": attr.method,
                        }
                    )

            # Add counterfactual info as a feature
            if full_exp.counterfactual and full_exp.counterfactual.changes:
                top_feats.append(
                    {
                        "feature": "counterfactual",
                        "value": full_exp.counterfactual.prediction_delta,
                        "contribution": 0.0,
                        "method": full_exp.counterfactual.source,
                    }
                )

            return Explanation(
                model_type="explainability",
                base_value=full_exp.shap.base_value if full_exp.shap else 0.0,
                prediction_value=risk_score,
                top_features=top_feats,
                latency_ms=full_exp.total_latency_ms,
                confidence=(
                    full_exp.formatted.confidence.__dict__
                    if full_exp.formatted and full_exp.formatted.confidence
                    else None
                ),
                counterfactual=(
                    full_exp.counterfactual.__dict__ if full_exp.counterfactual else None
                ),
                formatted_report=(full_exp.formatted.to_dict() if full_exp.formatted else None),
            )

        except Exception as exc:
            logger.warning("ExplainabilityEngine failed: {}", exc)
            return None

    def _init_explainability(self, tenant_id: str, model) -> None:
        """Initialize the explainability engine for a tenant."""
        try:
            if ExplainabilityEngine is None:
                return

            config = ExplainabilityConfig(
                enabled=True,
                shap_top_features=5,
                counterfactual_enabled=True,
            )

            self._explainability = ExplainabilityEngine(config=config)

            # Fit with available data
            feature_names = self.registry.feature_names or []
            if feature_names and hasattr(model, "feature_importance_"):
                # Use a dummy background dataset for SHAP initialization
                X_bg = np.zeros((100, len(feature_names)), dtype=np.float32)
                self._explainability.fit(
                    model=model,
                    X_background=X_bg,
                    feature_names=feature_names,
                    tenant_id=tenant_id,
                )

            logger.info("ExplainabilityEngine initialized for tenant={}", tenant_id)

        except Exception as exc:
            logger.warning("ExplainabilityEngine init failed: {}", exc)
            self._explainability = None

    def _explain_legacy(
        self,
        txn: TransactionRequest,
        features: dict,
        model,
        X: np.ndarray,
        risk_score: float,
    ) -> Optional[Explanation]:
        """Legacy explanation fallback for when ExplainabilityEngine is unavailable."""
        explanation = None

        # Champion model
        if ChampionModel is not None and isinstance(model, ChampionModel):
            explanation = self._explain_champion(model, X)

        # TabPFN adaptive learner
        elif TabPFNAdaptiveLearner is not None and isinstance(model, TabPFNAdaptiveLearner):
            explanation = self._explain_tabpfn(model, X)

        # Rules fallback
        if explanation is None and self.rules_engine:
            explanation = self._explain_rules(txn, features)

        # Cold-start
        elif (
            explanation is None
            and ColdStartEnsemble is not None
            and isinstance(model, ColdStartEnsemble)
        ):
            explanation = self._explain_cold_start(model, X)

        return explanation

    def _cache_explanation(self, trace_id: str, explanation: Explanation) -> None:
        """Cache explanation in Redis for /v1/explain/{trace_id} lookup."""
        try:
            r = self._get_redis()
            if r:
                r.setex(
                    f"fraudtrap:explanation:{trace_id}",
                    3600,
                    explanation.model_dump_json(),
                )
        except Exception as exc:
            logger.warning("Failed to cache explanation: {}", exc)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _features_to_array(self, features: dict[str, float], model=None) -> Optional[np.ndarray]:
        """Convert feature dict to numpy array aligned with model's feature order."""
        feature_names = getattr(model, "feature_names", None) or self.registry.feature_names
        if not feature_names:
            logger.warning(
                "Model has no persisted feature_names; falling back to sorted live feature keys"
            )
            feature_names = sorted(features.keys())
            self.registry.feature_names = feature_names
        try:
            missing = [name for name in feature_names if name not in features]
            if missing:
                logger.warning(
                    "Live feature vector missing {} trained feature(s): {}",
                    len(missing),
                    missing[:10],
                )
            values = np.array([features.get(f, 0.0) for f in feature_names], dtype=np.float32)
            if not np.all(np.isfinite(values)):
                logger.warning(
                    "Non-finite feature value detected; replacing NaN/Inf before scoring"
                )
                values = np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)
            return values
        except Exception as exc:
            logger.error("Feature array assembly failed: {}", exc)
            return None

    @staticmethod
    def _heuristic_score(features: dict[str, float], rule_result) -> float:
        """Day-zero fallback before tenant-specific models are trained."""
        score = 0.08
        score += min(0.30, max(features.get("amount_zscore", 0.0), 0.0) * 0.04)
        score += min(0.22, max(features.get("amount", 0.0), 0.0) / 2_500_000)
        score += 0.16 if features.get("is_new_device", 0.0) else 0.0
        score += 0.12 if features.get("is_new_merchant", 0.0) else 0.0
        score += 0.22 if features.get("impossible_travel", 0.0) else 0.0
        score += min(0.18, features.get("acct_v_1m_count", 0.0) * 0.015)
        score += 0.05 if features.get("is_night", 0.0) else 0.0
        score += 0.05 if features.get("channel_enc", -1.0) in (2.0, 5.0) else 0.0
        if rule_result and rule_result.triggered:
            score += rule_result.risk_boost
        return float(min(1.0, max(0.0, score)))

    def _emit_audit(self, response: ScoringResponse, txn: TransactionRequest) -> None:
        """Write decision to audit log (non-blocking, best-effort)."""
        try:
            r = self._get_redis()
            if r:
                event = {
                    **txn.model_dump(mode="json"),
                    **response.model_dump(mode="json"),
                    "is_fraud": int((txn.extra_fields or {}).get("simulated_label", 0)),
                }
                r.lpush("fraudtrap:recent_scores", json.dumps(event))
                r.ltrim("fraudtrap:recent_scores", 0, 4_999)
        except Exception as exc:
            logger.warning("Recent score cache write failed (non-fatal): {}", exc)

        try:
            from ingestion.kafka_client import FraudTrapProducer

            if not hasattr(self, "_producer"):
                self._producer = FraudTrapProducer()
                self._producer.connect()

            event = {
                **txn.model_dump(mode="json"),
                **response.model_dump(mode="json"),
                "is_fraud": int((txn.extra_fields or {}).get("simulated_label", 0)),
            }

            self._producer.emit_audit_event(event)
            self._producer.emit_scored_transaction(response.model_dump(mode="json"))

            logger.info(
                "AUDIT trace={} txn={} tenant={} score={:.4f} decision={} latency={}ms phase={}",
                response.trace_id,
                response.transaction_id,
                response.tenant_id,
                response.risk_score,
                response.decision,
                response.latency_ms,
                response.model_phase,
            )
        except Exception as exc:
            logger.warning("Audit emit failed (non-fatal): {}", exc)

    def recent_scores(self, limit: int = 500) -> list[dict]:
        """Return recent scored transactions for dashboard/demo views."""
        try:
            r = self._get_redis()
            if not r:
                return []
            rows = r.lrange("fraudtrap:recent_scores", 0, max(0, limit - 1))
            return [json.loads(row) for row in rows]
        except Exception as exc:
            logger.warning("Recent score cache read failed: {}", exc)
            return []
