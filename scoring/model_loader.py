"""
RiskLens — Model Loader
Handles loading models from disk with validation and warmup.
"""

from __future__ import annotations
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union
import numpy as np
import torch

from loguru import logger

from config.settings import get_settings
from models.cold_start.ensemble import ColdStartEnsemble, FraudVAE
from models.supervised.ensemble import SupervisedEnsemble
from models.supervised.semi_supervised import SemiSupervisedBridge
from models.gnn.gnn_scorer import GNNScorer

settings = get_settings()


@dataclass
class ModelLoadResult:
    """Result of model loading operation."""

    success: bool
    model: Optional[object] = None
    model_type: str = ""
    version: str = ""
    feature_names: list[str] = None
    error: str = ""
    load_time_ms: float = 0.0
    warmup_time_ms: float = 0.0


class ModelLoader:
    """
    Loads all model types from disk with validation and warmup.

    Supports:
    - SimpleFraudModel (logistic regression)
    - ColdStartEnsemble (VAE + IF + Tail)
    - SemiSupervisedBridge (XGBoost + cold-start)
    - SupervisedEnsemble (Stacked XGB/LGBM/CatBoost)
    - GNNScorer (Graph Neural Network)
    """

    def __init__(self, model_dir: Union[str, Path] = None):
        self.model_dir = Path(model_dir) if model_dir else Path(settings.model_dir)
        self._load_times: dict[str, float] = {}

    def load_simple_model(self, tenant_id: str) -> ModelLoadResult:
        """Load SimpleFraudModel for a tenant."""
        import time

        start = time.perf_counter()

        model_path = self.model_dir / tenant_id / "simple_model.pkl"
        if not model_path.exists():
            return ModelLoadResult(
                success=False,
                error=f"Model not found: {model_path}",
                load_time_ms=(time.perf_counter() - start) * 1000,
            )

        try:
            with open(model_path, "rb") as f:
                payload = pickle.load(f)

            # SimpleFraudModel doesn't have a load classmethod, recreate
            from scoring.simple_model import SimpleFraudModel

            model = SimpleFraudModel(
                feature_names=payload["feature_names"],
                weights=payload["weights"],
                bias=payload["bias"],
                mean=payload["mean"],
                scale=payload["scale"],
                model_version=payload.get("model_version", "unknown"),
                calibration_raw=payload.get("calibration_raw"),
                calibration_score=payload.get("calibration_score"),
            )

            load_time = (time.perf_counter() - start) * 1000
            warmup_time = self._warmup(model, payload["feature_names"])

            return ModelLoadResult(
                success=True,
                model=model,
                model_type="simple",
                version=payload.get("model_version", "unknown"),
                feature_names=payload["feature_names"],
                load_time_ms=load_time,
                warmup_time_ms=warmup_time,
            )

        except Exception as exc:
            return ModelLoadResult(
                success=False,
                error=f"Failed to load simple model: {exc}",
                load_time_ms=(time.perf_counter() - start) * 1000,
            )

    def load_cold_start(self, tenant_id: str) -> ModelLoadResult:
        """Load ColdStartEnsemble for a tenant."""
        import time

        start = time.perf_counter()

        model_path = self.model_dir / tenant_id / "phase1"
        if not model_path.exists():
            return ModelLoadResult(
                success=False,
                error=f"Cold start model not found: {model_path}",
                load_time_ms=(time.perf_counter() - start) * 1000,
            )

        try:
            model = ColdStartEnsemble.load(model_path, device="cpu")

            load_time = (time.perf_counter() - start) * 1000
            warmup_time = self._warmup(model, model.feature_names)

            return ModelLoadResult(
                success=True,
                model=model,
                model_type="cold_start",
                version=model.model_version,
                feature_names=model.feature_names,
                load_time_ms=load_time,
                warmup_time_ms=warmup_time,
            )

        except Exception as exc:
            return ModelLoadResult(
                success=False,
                error=f"Failed to load cold start: {exc}",
                load_time_ms=(time.perf_counter() - start) * 1000,
            )

    def load_semi_supervised(self, tenant_id: str) -> ModelLoadResult:
        """Load SemiSupervisedBridge for a tenant."""
        import time

        start = time.perf_counter()

        model_path = self.model_dir / tenant_id / "phase2"
        if not model_path.exists():
            return ModelLoadResult(
                success=False,
                error=f"Semi-supervised model not found: {model_path}",
                load_time_ms=(time.perf_counter() - start) * 1000,
            )

        try:
            # Need cold start model first
            cold_start = None
            cold_path = self.model_dir / tenant_id / "phase1"
            if cold_path.exists():
                cold_start = ColdStartEnsemble.load(cold_path, device="cpu")
            else:
                return ModelLoadResult(
                    success=False,
                    error="Semi-supervised requires cold start model (phase1)",
                    load_time_ms=(time.perf_counter() - start) * 1000,
                )

            from models.supervised.semi_supervised import SemiSupervisedBridge

            model = SemiSupervisedBridge.load(model_path, cold_start)

            load_time = (time.perf_counter() - start) * 1000
            warmup_time = self._warmup(model, cold_start.feature_names)

            return ModelLoadResult(
                success=True,
                model=model,
                model_type="semi_supervised",
                version=(
                    model.model_version
                    if hasattr(model, "model_version")
                    else "unknown"
                ),
                feature_names=cold_start.feature_names,
                load_time_ms=load_time,
                warmup_time_ms=warmup_time,
            )

        except Exception as exc:
            return ModelLoadResult(
                success=False,
                error=f"Failed to load semi-supervised: {exc}",
                load_time_ms=(time.perf_counter() - start) * 1000,
            )

    def load_supervised(self, tenant_id: str) -> ModelLoadResult:
        """Load SupervisedEnsemble for a tenant."""
        import time

        start = time.perf_counter()

        model_path = self.model_dir / tenant_id / "phase3"
        if not model_path.exists():
            return ModelLoadResult(
                success=False,
                error=f"Supervised model not found: {model_path}",
                load_time_ms=(time.perf_counter() - start) * 1000,
            )

        try:
            model = SupervisedEnsemble.load(model_path)

            load_time = (time.perf_counter() - start) * 1000
            warmup_time = self._warmup(model, model.feature_names)

            return ModelLoadResult(
                success=True,
                model=model,
                model_type="supervised",
                version=model.model_version,
                feature_names=model.feature_names,
                load_time_ms=load_time,
                warmup_time_ms=warmup_time,
            )

        except Exception as exc:
            return ModelLoadResult(
                success=False,
                error=f"Failed to load supervised: {exc}",
                load_time_ms=(time.perf_counter() - start) * 1000,
            )

    def load_gnn(self) -> ModelLoadResult:
        """Load GNN scorer (shared across tenants)."""
        import time

        start = time.perf_counter()

        model_path = self.model_dir / "gnn"
        if not model_path.exists():
            return ModelLoadResult(
                success=False,
                error=f"GNN model not found: {model_path}",
                load_time_ms=(time.perf_counter() - start) * 1000,
            )

        try:
            model = GNNScorer.load(model_path)

            load_time = (time.perf_counter() - start) * 1000
            # GNN warmup skipped (needs graph structure)

            return ModelLoadResult(
                success=True,
                model=model,
                model_type="gnn",
                version=(
                    model.model_version
                    if hasattr(model, "model_version")
                    else "unknown"
                ),
                feature_names=[],
                load_time_ms=load_time,
                warmup_time_ms=0.0,
            )

        except Exception as exc:
            return ModelLoadResult(
                success=False,
                error=f"Failed to load GNN: {exc}",
                load_time_ms=(time.perf_counter() - start) * 1000,
            )

    def load_all_tenants(self) -> dict[str, dict[str, ModelLoadResult]]:
        """Load all models for all tenants."""
        results = {}

        if not self.model_dir.exists():
            logger.warning("Model directory does not exist: {}", self.model_dir)
            return results

        for tenant_dir in self.model_dir.iterdir():
            if not tenant_dir.is_dir():
                continue

            tenant_id = tenant_dir.name
            results[tenant_id] = {}

            # Simple model (always load if exists)
            simple_result = self.load_simple_model(tenant_id)
            if simple_result.success:
                results[tenant_id]["simple"] = simple_result

            # Phase models
            cold_result = self.load_cold_start(tenant_id)
            if cold_result.success:
                results[tenant_id]["cold_start"] = cold_result

            semi_result = self.load_semi_supervised(tenant_id)
            if semi_result.success:
                results[tenant_id]["semi_supervised"] = semi_result

            supervised_result = self.load_supervised(tenant_id)
            if supervised_result.success:
                results[tenant_id]["supervised"] = supervised_result

        # GNN (shared)
        gnn_result = self.load_gnn()
        if gnn_result.success:
            results["_shared"] = {"gnn": gnn_result}

        return results

    def _warmup(self, model, feature_names: list[str]) -> float:
        """Run dummy inference to warm up model."""
        import time

        if not feature_names:
            return 0.0

        try:
            dummy = np.zeros((1, len(feature_names)), dtype=np.float32)

            start = time.perf_counter()
            _ = model.score(dummy)
            return (time.perf_counter() - start) * 1000
        except Exception as exc:
            logger.warning("Warmup failed: {}", exc)
            return 0.0

    def get_load_summary(self) -> dict:
        """Get summary of last load operation."""
        return {
            "total_models": len(self._load_times),
            "load_times_ms": self._load_times,
        }


def create_model_loader(model_dir: Union[str, Path] = None) -> ModelLoader:
    """Factory function to create ModelLoader."""
    return ModelLoader(model_dir)
