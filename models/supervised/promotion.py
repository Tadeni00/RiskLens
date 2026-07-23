"""
FraudTrap — Champion Promotion Logic
Handles the promotion of challenger models to champion status.
Implements the Champion-Challenger promotion workflow.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
from loguru import logger

from models.supervised.registry import (
    ModelRegistry,
    ModelMetadata,
    ModelStatus,
    PromotionStatus,
)
from models.supervised.evaluator import (
    ModelEvaluator,
    EvaluationMetrics,
    EvaluationReport,
)


@dataclass
class PromotionCriteria:
    """Criteria for champion promotion."""

    # Minimum PR-AUC improvement over champion
    min_pr_auc_improvement: float = 0.01

    # Maximum allowed FPR
    max_fpr: float = 0.01

    # Maximum allowed calibration error
    max_calibration_error: float = 0.05

    # Maximum allowed latency ratio (challenger/champion)
    max_latency_ratio: float = 2.0

    # Minimum number of validation samples required
    min_validation_samples: int = 1000

    # Whether to require manual approval
    require_manual_approval: bool = False

    # Auto-approve if all criteria met
    auto_approve: bool = True


@dataclass
class PromotionDecision:
    """Result of a promotion decision."""

    challenger_id: str
    champion_id: str
    approved: bool
    criteria_met: List[str] = field(default_factory=list)
    criteria_failed: List[str] = field(default_factory=list)
    metrics_comparison: Dict[str, Any] = field(default_factory=dict)
    decision_date: str = ""
    reviewer: str = "system"
    notes: str = ""

    def __post_init__(self):
        if not self.decision_date:
            self.decision_date = datetime.now(timezone.utc).isoformat()


class ChampionPromoter:
    """
    Handles champion promotion workflow.

    Features:
    - Automated promotion based on criteria
    - Manual approval workflow
    - Rollback support
    - Audit logging
    """

    def __init__(
        self,
        registry: ModelRegistry,
        evaluator: ModelEvaluator = None,
        criteria: PromotionCriteria = None,
        output_dir: Path = None,
    ):
        """
        Initialize promoter.

        Args:
            registry: Model registry
            evaluator: Model evaluator
            criteria: Promotion criteria
            output_dir: Directory for promotion logs
        """
        self.registry = registry
        self.evaluator = evaluator or ModelEvaluator()
        self.criteria = criteria or PromotionCriteria()
        self.output_dir = (
            Path(output_dir) if output_dir else Path("models/supervised/promotions")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._promotion_history: List[PromotionDecision] = []

    def evaluate_and_promote(
        self,
        challenger_id: str,
        X_val: np.ndarray,
        y_val: np.ndarray,
        reviewer: str = "system",
        notes: str = "",
        force: bool = False,
    ) -> PromotionDecision:
        """
        Evaluate a challenger and potentially promote to champion.

        Args:
            challenger_id: Challenger model ID
            X_val: Validation features
            y_val: Validation labels
            reviewer: Who is reviewing
            notes: Review notes
            force: Force promotion even if criteria not met

        Returns:
            PromotionDecision object
        """
        # Get challenger from registry
        challenger = self.registry.get_model(challenger_id)
        if not challenger:
            return PromotionDecision(
                challenger_id=challenger_id,
                champion_id="",
                approved=False,
                criteria_failed=["challenger_not_found"],
                notes=notes,
            )

        # Get current champion
        champion = self.registry.get_champion()
        if not champion:
            # No current champion, promote directly
            return self._promote_first_champion(
                challenger_id, X_val, y_val, reviewer, notes
            )

        # Evaluate challenger
        challenger_metrics = self.evaluator.evaluate_model(
            model=self._load_model_artifact(challenger_id),
            X_val=X_val,
            y_val=y_val,
            model_id=challenger_id,
            algorithm=challenger.algorithm,
            version=challenger.version,
        )

        # Evaluate champion
        champion_metrics = self.evaluator.evaluate_model(
            model=self._load_model_artifact(champion.model_id),
            X_val=X_val,
            y_val=y_val,
            model_id=champion.model_id,
            algorithm=champion.algorithm,
            version=champion.version,
        )

        # Make promotion decision
        decision = self._make_decision(
            challenger_metrics,
            champion_metrics,
            reviewer,
            notes,
            force,
        )

        # Execute promotion if approved
        if decision.approved:
            self._execute_promotion(challenger_id, decision)

        # Save decision
        self._save_decision(decision)

        return decision

    def _promote_first_champion(
        self,
        challenger_id: str,
        X_val: np.ndarray,
        y_val: np.ndarray,
        reviewer: str,
        notes: str,
    ) -> PromotionDecision:
        """Promote the first champion when no champion exists."""
        challenger = self.registry.get_model(challenger_id)
        if not challenger:
            return PromotionDecision(
                challenger_id=challenger_id,
                champion_id="",
                approved=False,
                criteria_failed=["challenger_not_found"],
            )

        # Evaluate the model
        metrics = self.evaluator.evaluate_model(
            model=self._load_model_artifact(challenger_id),
            X_val=X_val,
            y_val=y_val,
            model_id=challenger_id,
            algorithm=challenger.algorithm,
            version=challenger.version,
        )

        # Auto-promote first champion
        decision = PromotionDecision(
            challenger_id=challenger_id,
            champion_id="none",
            approved=True,
            criteria_met=["first_champion"],
            metrics_comparison={
                "pr_auc": metrics.pr_auc,
                "roc_auc": metrics.roc_auc,
                "f2_score": metrics.f2_score,
                "fpr": metrics.fpr,
            },
            reviewer=reviewer,
            notes=f"First champion promotion. {notes}",
        )

        # Execute promotion
        self.registry.promote(challenger_id, reviewer=reviewer, notes=notes)

        # Save decision
        self._save_decision(decision)

        logger.info("First champion promoted: {}", challenger_id)

        return decision

    def _make_decision(
        self,
        challenger_metrics: EvaluationMetrics,
        champion_metrics: EvaluationMetrics,
        reviewer: str,
        notes: str,
        force: bool = False,
    ) -> PromotionDecision:
        """Make promotion decision based on criteria."""
        criteria_met = []
        criteria_failed = []

        # PR-AUC improvement
        pr_auc_improvement = challenger_metrics.pr_auc - champion_metrics.pr_auc
        if pr_auc_improvement >= self.criteria.min_pr_auc_improvement:
            criteria_met.append("pr_auc_improvement")
        else:
            criteria_failed.append(
                f"pr_auc_improvement ({pr_auc_improvement:.4f} < {self.criteria.min_pr_auc_improvement:.4f})"
            )

        # FPR check
        if challenger_metrics.fpr <= self.criteria.max_fpr:
            criteria_met.append("fpr_within_limit")
        else:
            criteria_failed.append(
                f"fpr ({challenger_metrics.fpr:.4f} > {self.criteria.max_fpr:.4f})"
            )

        # Calibration check
        if challenger_metrics.calibration_error <= self.criteria.max_calibration_error:
            criteria_met.append("calibration_within_limit")
        else:
            criteria_failed.append(
                f"calibration_error ({challenger_metrics.calibration_error:.4f} > {self.criteria.max_calibration_error:.4f})"
            )

        # Latency check
        latency_ratio = challenger_metrics.avg_latency_ms / max(
            champion_metrics.avg_latency_ms, 0.001
        )
        if latency_ratio <= self.criteria.max_latency_ratio:
            criteria_met.append("latency_acceptable")
        else:
            criteria_failed.append(
                f"latency ({latency_ratio:.1f}x > {self.criteria.max_latency_ratio:.1f}x)"
            )

        # Validation samples check
        if challenger_metrics.dataset_size >= self.criteria.min_validation_samples:
            criteria_met.append("sufficient_validation_samples")
        else:
            criteria_failed.append(
                f"validation_samples ({challenger_metrics.dataset_size} < {self.criteria.min_validation_samples})"
            )

        # Determine approval
        approved = False
        if force:
            approved = True
            criteria_met.append("force_approved")
        elif self.criteria.auto_approve and len(criteria_failed) == 0:
            approved = True
        elif not self.criteria.require_manual_approval and len(criteria_failed) == 0:
            approved = True

        return PromotionDecision(
            challenger_id=challenger_metrics.model_id,
            champion_id=champion_metrics.model_id,
            approved=approved,
            criteria_met=criteria_met,
            criteria_failed=criteria_failed,
            metrics_comparison={
                "champion": {
                    "pr_auc": champion_metrics.pr_auc,
                    "roc_auc": champion_metrics.roc_auc,
                    "f2_score": champion_metrics.f2_score,
                    "fpr": champion_metrics.fpr,
                    "calibration_error": champion_metrics.calibration_error,
                },
                "challenger": {
                    "pr_auc": challenger_metrics.pr_auc,
                    "roc_auc": challenger_metrics.roc_auc,
                    "f2_score": challenger_metrics.f2_score,
                    "fpr": challenger_metrics.fpr,
                    "calibration_error": challenger_metrics.calibration_error,
                },
                "improvement": {
                    "pr_auc": pr_auc_improvement,
                    "roc_auc": challenger_metrics.roc_auc - champion_metrics.roc_auc,
                    "f2_score": challenger_metrics.f2_score - champion_metrics.f2_score,
                },
            },
            reviewer=reviewer,
            notes=notes,
        )

    def _execute_promotion(
        self, challenger_id: str, decision: PromotionDecision
    ) -> None:
        """Execute the promotion."""
        self.registry.promote(
            challenger_id,
            reviewer=decision.reviewer,
            notes=f"Approved: {', '.join(decision.criteria_met)}",
        )

        logger.info(
            "Champion promoted: {} → {} (PR-AUC improvement: {:.4f})",
            decision.champion_id,
            challenger_id,
            decision.metrics_comparison.get("improvement", {}).get("pr_auc", 0),
        )

    def _load_model_artifact(self, model_id: str):
        """Load the actual model artifact from disk."""
        from pathlib import Path

        # Try to find the model file
        model_dir = Path("models/supervised/saved_models") / model_id

        if not model_dir.exists():
            logger.warning("Model artifact not found: {}", model_dir)
            return None

        # Load based on algorithm
        model = self.registry.get_model(model_id)
        if not model:
            return None

        if model.algorithm.lower() == "catboost":
            from models.supervised.champion import ChampionModel

            return ChampionModel.load(model_dir)
        else:
            from models.supervised.challengers import create_challenger

            return create_challenger(model.algorithm).load(model_dir)

    def _save_decision(self, decision: PromotionDecision) -> None:
        """Save promotion decision to disk."""
        filepath = (
            self.output_dir
            / f"decision_{decision.challenger_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(filepath, "w") as f:
            json.dump(
                {
                    "challenger_id": decision.challenger_id,
                    "champion_id": decision.champion_id,
                    "approved": decision.approved,
                    "criteria_met": decision.criteria_met,
                    "criteria_failed": decision.criteria_failed,
                    "metrics_comparison": decision.metrics_comparison,
                    "decision_date": decision.decision_date,
                    "reviewer": decision.reviewer,
                    "notes": decision.notes,
                },
                f,
                indent=2,
                default=str,
            )

        logger.debug("Promotion decision saved to {}", filepath)

    def rollback_champion(
        self,
        reviewer: str = "system",
        notes: str = "",
    ) -> bool:
        """
        Rollback champion to previous version.

        Args:
            reviewer: Who approved the rollback
            notes: Rollback reason

        Returns:
            True if rollback successful
        """
        return self.registry.rollback(reviewer=reviewer, notes=notes)

    def get_promotion_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get promotion history from disk."""
        decisions = []

        for filepath in sorted(self.output_dir.glob("decision_*.json"), reverse=True)[
            :limit
        ]:
            with open(filepath, "r") as f:
                decisions.append(json.load(f))

        return decisions

    def get_champion_status(self) -> Dict[str, Any]:
        """Get current champion status."""
        champion = self.registry.get_champion()

        if not champion:
            return {
                "has_champion": False,
                "champion_id": None,
                "challengers": len(self.registry.get_challengers()),
            }

        return {
            "has_champion": True,
            "champion_id": champion.model_id,
            "algorithm": champion.algorithm,
            "version": champion.version,
            "pr_auc": champion.pr_auc,
            "fpr": champion.fpr,
            "calibration_error": champion.calibration_error,
            "trained_at": champion.trained_at,
            "promoted_at": champion.promoted_at,
            "challengers": len(self.registry.get_challengers()),
        }

    def should_promote(
        self,
        challenger_id: str,
        champion_metrics: Dict[str, float],
        challenger_metrics: Dict[str, float],
    ) -> tuple[bool, List[str], List[str]]:
        """
        Check if a challenger should be promoted.

        Args:
            challenger_id: Challenger model ID
            champion_metrics: Champion metrics dict
            challenger_metrics: Challenger metrics dict

        Returns:
            Tuple of (should_promote, criteria_met, criteria_failed)
        """
        criteria_met = []
        criteria_failed = []

        # PR-AUC improvement
        pr_auc_improvement = challenger_metrics.get("pr_auc", 0) - champion_metrics.get(
            "pr_auc", 0
        )
        if pr_auc_improvement >= self.criteria.min_pr_auc_improvement:
            criteria_met.append("pr_auc_improvement")
        else:
            criteria_failed.append("pr_auc_improvement")

        # FPR check
        if challenger_metrics.get("fpr", 1) <= self.criteria.max_fpr:
            criteria_met.append("fpr_within_limit")
        else:
            criteria_failed.append("fpr_within_limit")

        # Calibration check
        if (
            challenger_metrics.get("calibration_error", 1)
            <= self.criteria.max_calibration_error
        ):
            criteria_met.append("calibration_within_limit")
        else:
            criteria_failed.append("calibration_within_limit")

        should_promote = len(criteria_failed) == 0

        return should_promote, criteria_met, criteria_failed


def create_promoter(
    registry: ModelRegistry = None,
    criteria: PromotionCriteria = None,
) -> ChampionPromoter:
    """
    Factory function to create a ChampionPromoter.

    Args:
        registry: Model registry (creates new if None)
        criteria: Promotion criteria

    Returns:
        ChampionPromoter instance
    """
    if registry is None:
        registry = ModelRegistry()

    return ChampionPromoter(registry=registry, criteria=criteria)
