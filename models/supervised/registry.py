"""
FraudTrap — Model Registry for Champion-Challenger Architecture
Manages model registration, versioning, promotion, rollback, and archival.
Supports the Champion-Challenger supervised learning architecture.
"""
from __future__ import annotations
import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
from loguru import logger


class ModelStatus(str, Enum):
    """Model lifecycle status."""
    CHAMPION = "champion"
    CHALLENGER = "challenger"
    ARCHIVED = "archived"
    RETIRED = "retired"
    FAILED = "failed"


class PromotionStatus(str, Enum):
    """Promotion request status."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"


@dataclass
class ModelMetadata:
    """Model metadata for registry."""
    model_id: str
    version: str
    algorithm: str
    training_date: str
    dataset_version: str
    feature_version: str
    
    # Performance metrics
    pr_auc: float = 0.0
    roc_auc: float = 0.0
    f2_score: float = 0.0
    fpr: float = 0.0
    calibration_error: float = 0.0
    
    # Status
    status: ModelStatus = ModelStatus.CHALLENGER
    is_active: bool = False
    
    # Metadata
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    calibration_method: str = "isotonic"
    description: str = ""
    created_by: str = "system"
    
    # Timestamps
    created_at: str = ""
    promoted_at: Optional[str] = None
    archived_at: Optional[str] = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_id": self.model_id,
            "version": self.version,
            "algorithm": self.algorithm,
            "training_date": self.training_date,
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "f2_score": self.f2_score,
            "fpr": self.fpr,
            "calibration_error": self.calibration_error,
            "status": self.status.value,
            "is_active": self.is_active,
            "hyperparameters": self.hyperparameters,
            "calibration_method": self.calibration_method,
            "description": self.description,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "promoted_at": self.promoted_at,
            "archived_at": self.archived_at,
        }


@dataclass
class PromotionRequest:
    """Promotion request for challenger → champion."""
    request_id: str
    challenger_id: str
    champion_id: str
    challenger_metrics: Dict[str, float]
    champion_metrics: Dict[str, float]
    
    # Comparison results
    pr_auc_improvement: float = 0.0
    calibration_met: bool = False
    fpr_met: bool = False
    
    # Status
    status: PromotionStatus = PromotionStatus.PENDING
    reviewer: str = ""
    review_notes: str = ""
    created_at: str = ""
    reviewed_at: Optional[str] = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


class ModelRegistry:
    """
    Model Registry for Champion-Challenger architecture.
    
    Manages:
    - Model registration and versioning
    - Champion/Challenger status tracking
    - Promotion requests and approvals
    - Model archival and rollback
    - Feature/dataset version pinning
    """
    
    def __init__(self, registry_dir: Path = None):
        """
        Initialize model registry.
        
        Args:
            registry_dir: Directory to store registry data
        """
        self.registry_dir = Path(registry_dir) if registry_dir else Path("models/supervised/registry_data")
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        self._models: Dict[str, ModelMetadata] = {}
        self._promotion_requests: List[PromotionRequest] = []
        self._champion_id: Optional[str] = None
        
        # Load existing registry
        self._load_registry()
    
    def register(
        self,
        model_id: str,
        version: str,
        algorithm: str,
        training_date: str,
        dataset_version: str,
        feature_version: str,
        metrics: Dict[str, float],
        hyperparameters: Dict[str, Any] = None,
        calibration_method: str = "isotonic",
        description: str = "",
        created_by: str = "system",
    ) -> ModelMetadata:
        """
        Register a new model in the registry.
        
        Args:
            model_id: Unique model identifier
            version: Model version string
            algorithm: Model algorithm (catboost, xgboost, etc.)
            training_date: Date of training
            dataset_version: Dataset version used for training
            feature_version: Feature version used
            metrics: Performance metrics (pr_auc, roc_auc, f2_score, fpr, calibration_error)
            hyperparameters: Model hyperparameters
            calibration_method: Calibration method used
            description: Model description
            created_by: Who created this model
        
        Returns:
            ModelMetadata object
        """
        metadata = ModelMetadata(
            model_id=model_id,
            version=version,
            algorithm=algorithm,
            training_date=training_date,
            dataset_version=dataset_version,
            feature_version=feature_version,
            pr_auc=metrics.get("pr_auc", 0.0),
            roc_auc=metrics.get("roc_auc", 0.0),
            f2_score=metrics.get("f2_score", 0.0),
            fpr=metrics.get("fpr", 0.0),
            calibration_error=metrics.get("calibration_error", 0.0),
            status=ModelStatus.CHALLENGER,
            hyperparameters=hyperparameters or {},
            calibration_method=calibration_method,
            description=description,
            created_by=created_by,
        )
        
        self._models[model_id] = metadata
        self._save_registry()
        
        logger.info(
            "Model registered: {} (version={}, algorithm={}, pr_auc={:.4f})",
            model_id, version, algorithm, metadata.pr_auc
        )
        
        return metadata
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID."""
        return self._models.get(model_id)
    
    def list_models(
        self,
        status: Optional[ModelStatus] = None,
        algorithm: Optional[str] = None,
        limit: int = 100,
    ) -> List[ModelMetadata]:
        """
        List models with optional filtering.
        
        Args:
            status: Filter by status
            algorithm: Filter by algorithm
            limit: Maximum number of results
        
        Returns:
            List of ModelMetadata objects
        """
        models = list(self._models.values())
        
        if status:
            models = [m for m in models if m.status == status]
        
        if algorithm:
            models = [m for m in models if m.algorithm.lower() == algorithm.lower()]
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x.created_at, reverse=True)
        
        return models[:limit]
    
    def promote(
        self,
        model_id: str,
        champion_id: str = None,
        reviewer: str = "system",
        notes: str = "",
    ) -> Optional[PromotionRequest]:
        """
        Request promotion of a challenger to champion.
        
        Args:
            model_id: Challenger model ID
            champion_id: Current champion ID (if None, uses current champion)
            reviewer: Who approved the promotion
            notes: Review notes
        
        Returns:
            PromotionRequest object
        """
        challenger = self._models.get(model_id)
        if not challenger:
            logger.warning("Challenger model not found: {}", model_id)
            return None
        
        if champion_id:
            champion = self._models.get(champion_id)
        else:
            champion = self.get_champion()
        
        if not champion:
            # No current champion — auto-promote as first champion
            logger.info("No current champion, promoting {} as first champion", model_id)
            challenger.status = ModelStatus.CHAMPION
            challenger.is_active = True
            challenger.promoted_at = datetime.now(timezone.utc).isoformat()
            self._champion_id = model_id
            
            request = PromotionRequest(
                request_id=f"promo_{model_id}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                challenger_id=model_id,
                champion_id="none",
                challenger_metrics={
                    "pr_auc": challenger.pr_auc,
                    "roc_auc": challenger.roc_auc,
                    "f2_score": challenger.f2_score,
                    "fpr": challenger.fpr,
                    "calibration_error": challenger.calibration_error,
                },
                champion_metrics={},
                pr_auc_improvement=0.0,
                calibration_met=True,
                fpr_met=True,
                status=PromotionStatus.COMPLETED,
                reviewer=reviewer,
                review_notes=f"First champion promotion. {notes}",
            )
            request.reviewed_at = datetime.now(timezone.utc).isoformat()
            self._promotion_requests.append(request)
            self._save_registry()
            logger.info("First champion promoted: {}", model_id)
            return request
        
        # Compare metrics
        pr_auc_improvement = challenger.pr_auc - champion.pr_auc
        calibration_met = challenger.calibration_error <= champion.calibration_error
        fpr_met = challenger.fpr <= champion.fpr
        
        # Determine if promotion should be recommended
        should_promote = (
            pr_auc_improvement > 0 and
            calibration_met and
            fpr_met
        )
        
        request = PromotionRequest(
            request_id=f"promo_{model_id}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            challenger_id=model_id,
            champion_id=champion.model_id,
            challenger_metrics={
                "pr_auc": challenger.pr_auc,
                "roc_auc": challenger.roc_auc,
                "f2_score": challenger.f2_score,
                "fpr": challenger.fpr,
                "calibration_error": challenger.calibration_error,
            },
            champion_metrics={
                "pr_auc": champion.pr_auc,
                "roc_auc": champion.roc_auc,
                "f2_score": champion.f2_score,
                "fpr": champion.fpr,
                "calibration_error": champion.calibration_error,
            },
            pr_auc_improvement=pr_auc_improvement,
            calibration_met=calibration_met,
            fpr_met=fpr_met,
            status=PromotionStatus.APPROVED if should_promote else PromotionStatus.PENDING,
            reviewer=reviewer,
            review_notes=notes,
        )
        
        self._promotion_requests.append(request)
        
        # Auto-approve if metrics meet criteria
        if should_promote:
            self._execute_promotion(model_id, request)
        
        self._save_registry()
        
        logger.info(
            "Promotion request created: {} (pr_auc_improvement={:.4f}, approved={})",
            request.request_id, pr_auc_improvement, should_promote
        )
        
        return request
    
    def _execute_promotion(self, challenger_id: str, request: PromotionRequest) -> None:
        """Execute the promotion (challenger becomes champion)."""
        challenger = self._models.get(challenger_id)
        if not challenger:
            return
        
        # Demote current champion
        old_champion = self.get_champion()
        if old_champion:
            old_champion.status = ModelStatus.CHALLENGER
            old_champion.is_active = False
            logger.info("Old champion demoted: {}", old_champion.model_id)
        
        # Promote challenger
        challenger.status = ModelStatus.CHAMPION
        challenger.is_active = True
        challenger.promoted_at = datetime.now(timezone.utc).isoformat()
        self._champion_id = challenger_id
        
        request.status = PromotionStatus.COMPLETED
        request.reviewed_at = datetime.now(timezone.utc).isoformat()
        
        logger.info(
            "Model promoted to champion: {} (version={})",
            challenger_id, challenger.version
        )
    
    def rollback(self, champion_id: str = None, reviewer: str = "system", notes: str = "") -> bool:
        """
        Rollback champion to previous version.
        
        Args:
            champion_id: Champion ID to rollback (if None, uses current)
            reviewer: Who approved the rollback
            notes: Rollback reason
        
        Returns:
            True if rollback successful
        """
        if champion_id:
            current_champion = self._models.get(champion_id)
        else:
            current_champion = self.get_champion()
        
        if not current_champion:
            logger.warning("No champion to rollback")
            return False
        
        # Find previous champion (most recent promoted champion)
        previous_champions = [
            m for m in self._models.values()
            if m.status == ModelStatus.CHALLENGER and m.promoted_at is not None
        ]
        
        if not previous_champions:
            logger.warning("No previous champion to rollback to")
            return False
        
        # Sort by promoted_at (most recent first)
        previous_champions.sort(key=lambda x: x.promoted_at, reverse=True)
        previous_champion = previous_champions[0]
        
        # Demote current champion
        current_champion.status = ModelStatus.ARCHIVED
        current_champion.is_active = False
        current_champion.archived_at = datetime.now(timezone.utc).isoformat()
        
        # Restore previous champion
        previous_champion.status = ModelStatus.CHAMPION
        previous_champion.is_active = True
        self._champion_id = previous_champion.model_id
        
        logger.info(
            "Champion rolled back: {} → {}",
            current_champion.model_id, previous_champion.model_id
        )
        
        self._save_registry()
        return True
    
    def archive(self, model_id: str, notes: str = "") -> bool:
        """
        Archive a model.
        
        Args:
            model_id: Model ID to archive
            notes: Archive reason
        
        Returns:
            True if successful
        """
        model = self._models.get(model_id)
        if not model:
            logger.warning("Model not found: {}", model_id)
            return False
        
        if model.status == ModelStatus.CHAMPION:
            logger.warning("Cannot archive champion model. Promote another model first.")
            return False
        
        model.status = ModelStatus.ARCHIVED
        model.archived_at = datetime.now(timezone.utc).isoformat()
        
        logger.info("Model archived: {} ({})", model_id, notes)
        self._save_registry()
        return True
    
    def retire(self, model_id: str, notes: str = "") -> bool:
        """
        Retire a model (permanent archival).
        
        Args:
            model_id: Model ID to retire
            notes: Retirement reason
        
        Returns:
            True if successful
        """
        model = self._models.get(model_id)
        if not model:
            logger.warning("Model not found: {}", model_id)
            return False
        
        if model.status == ModelStatus.CHAMPION:
            logger.warning("Cannot retire champion model. Promote another model first.")
            return False
        
        model.status = ModelStatus.RETIRED
        model.archived_at = datetime.now(timezone.utc).isoformat()
        
        logger.info("Model retired: {} ({})", model_id, notes)
        self._save_registry()
        return True
    
    def get_champion(self) -> Optional[ModelMetadata]:
        """Get current champion model."""
        if self._champion_id:
            return self._models.get(self._champion_id)
        
        # Find champion by status
        for model in self._models.values():
            if model.status == ModelStatus.CHAMPION:
                self._champion_id = model.model_id
                return model
        
        return None
    
    def get_challengers(self, algorithm: Optional[str] = None) -> List[ModelMetadata]:
        """Get all challenger models."""
        return self.list_models(status=ModelStatus.CHALLENGER, algorithm=algorithm)
    
    def get_promotion_history(self, limit: int = 50) -> List[PromotionRequest]:
        """Get promotion request history."""
        return sorted(
            self._promotion_requests,
            key=lambda x: x.created_at,
            reverse=True
        )[:limit]
    
    def compare_models(
        self,
        model_id_1: str,
        model_id_2: str,
    ) -> Dict[str, Any]:
        """
        Compare two models side-by-side.
        
        Args:
            model_id_1: First model ID
            model_id_2: Second model ID
        
        Returns:
            Comparison dictionary
        """
        model1 = self._models.get(model_id_1)
        model2 = self._models.get(model_id_2)
        
        if not model1 or not model2:
            return {"error": "One or both models not found"}
        
        return {
            "model_1": {
                "model_id": model1.model_id,
                "algorithm": model1.algorithm,
                "version": model1.version,
                "pr_auc": model1.pr_auc,
                "roc_auc": model1.roc_auc,
                "f2_score": model1.f2_score,
                "fpr": model1.fpr,
                "calibration_error": model1.calibration_error,
                "status": model1.status.value,
            },
            "model_2": {
                "model_id": model2.model_id,
                "algorithm": model2.algorithm,
                "version": model2.version,
                "pr_auc": model2.pr_auc,
                "roc_auc": model2.roc_auc,
                "f2_score": model2.f2_score,
                "fpr": model2.fpr,
                "calibration_error": model2.calibration_error,
                "status": model2.status.value,
            },
            "differences": {
                "pr_auc_diff": model1.pr_auc - model2.pr_auc,
                "roc_auc_diff": model1.roc_auc - model2.roc_auc,
                "f2_diff": model1.f2_score - model2.f2_score,
                "fpr_diff": model1.fpr - model2.fpr,
                "calibration_error_diff": model1.calibration_error - model2.calibration_error,
            },
        }
    
    def _save_registry(self) -> None:
        """Save registry to disk."""
        registry_file = self.registry_dir / "registry.json"
        
        data = {
            "champion_id": self._champion_id,
            "models": {k: v.to_dict() for k, v in self._models.items()},
            "promotion_requests": [
                {
                    "request_id": r.request_id,
                    "challenger_id": r.challenger_id,
                    "champion_id": r.champion_id,
                    "status": r.status.value,
                    "created_at": r.created_at,
                    "reviewed_at": r.reviewed_at,
                }
                for r in self._promotion_requests
            ],
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        
        with open(registry_file, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.debug("Registry saved to {}", registry_file)
    
    def _load_registry(self) -> None:
        """Load registry from disk."""
        registry_file = self.registry_dir / "registry.json"
        
        if not registry_file.exists():
            logger.info("No existing registry found, starting fresh")
            return
        
        try:
            with open(registry_file, "r") as f:
                data = json.load(f)
            
            self._champion_id = data.get("champion_id")
            
            # Load models
            for model_id, model_data in data.get("models", {}).items():
                metadata = ModelMetadata(
                    model_id=model_data["model_id"],
                    version=model_data["version"],
                    algorithm=model_data["algorithm"],
                    training_date=model_data["training_date"],
                    dataset_version=model_data["dataset_version"],
                    feature_version=model_data["feature_version"],
                    pr_auc=model_data.get("pr_auc", 0.0),
                    roc_auc=model_data.get("roc_auc", 0.0),
                    f2_score=model_data.get("f2_score", 0.0),
                    fpr=model_data.get("fpr", 0.0),
                    calibration_error=model_data.get("calibration_error", 0.0),
                    status=ModelStatus(model_data.get("status", "challenger")),
                    is_active=model_data.get("is_active", False),
                    hyperparameters=model_data.get("hyperparameters", {}),
                    calibration_method=model_data.get("calibration_method", "isotonic"),
                    description=model_data.get("description", ""),
                    created_by=model_data.get("created_by", "system"),
                    created_at=model_data.get("created_at", ""),
                    promoted_at=model_data.get("promoted_at"),
                    archived_at=model_data.get("archived_at"),
                )
                self._models[model_id] = metadata
            
            # Load promotion requests
            for req_data in data.get("promotion_requests", []):
                request = PromotionRequest(
                    request_id=req_data["request_id"],
                    challenger_id=req_data["challenger_id"],
                    champion_id=req_data["champion_id"],
                    status=PromotionStatus(req_data.get("status", "pending")),
                    created_at=req_data.get("created_at", ""),
                    reviewed_at=req_data.get("reviewed_at"),
                )
                self._promotion_requests.append(request)
            
            logger.info(
                "Registry loaded: {} models, champion={}",
                len(self._models), self._champion_id
            )
        
        except Exception as exc:
            logger.error("Failed to load registry: {}", exc)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        champions = [m for m in self._models.values() if m.status == ModelStatus.CHAMPION]
        challengers = [m for m in self._models.values() if m.status == ModelStatus.CHALLENGER]
        archived = [m for m in self._models.values() if m.status == ModelStatus.ARCHIVED]
        retired = [m for m in self._models.values() if m.status == ModelStatus.RETIRED]
        
        algorithms = {}
        for model in self._models.values():
            alg = model.algorithm.lower()
            if alg not in algorithms:
                algorithms[alg] = 0
            algorithms[alg] += 1
        
        return {
            "total_models": len(self._models),
            "champions": len(champions),
            "challengers": len(challengers),
            "archived": len(archived),
            "retired": len(retired),
            "algorithms": algorithms,
            "current_champion": self._champion_id,
            "promotion_requests": len(self._promotion_requests),
        }
