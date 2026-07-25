"""
RiskLens — Explanation Formatter
Converts raw SHAP and counterfactual outputs into analyst-friendly language.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
from loguru import logger

from models.explainability.types import (
    SHAPExplanation,
    CounterfactualExplanation,
    ConfidenceInfo,
    FormattedReport,
)


class ExplanationFormatter:
    """
    Produces concise, analyst-friendly reports from raw explanation data.

    Every output answers three questions:
    1. Why was this classified this way?
    2. Which features contributed most?
    3. What minimal changes would flip the decision?
    """

    def __init__(self, max_drivers: int = 5):
        self.max_drivers = max_drivers

    def format(
        self,
        fraud_probability: float,
        shap: Optional[SHAPExplanation] = None,
        counterfactual: Optional[CounterfactualExplanation] = None,
        confidence: Optional[ConfidenceInfo] = None,
    ) -> FormattedReport:
        """
        Format a complete explanation into an analyst-friendly report.

        Args:
            fraud_probability: Final fraud probability
            shap: SHAP explanation (feature attributions)
            counterfactual: Counterfactual explanation
            confidence: Model confidence metadata

        Returns:
            FormattedReport with human-readable strings
        """
        confidence = confidence or ConfidenceInfo(expert_used="Unknown")

        risk_drivers = self._format_risk_drivers(shap)
        counterfactual_summary = self._format_counterfactual_summary(counterfactual)
        nearest_legitimate = self._format_nearest_legitimate(counterfactual)
        minimal_changes = self._format_minimal_changes(counterfactual)

        return FormattedReport(
            fraud_probability=fraud_probability,
            confidence=confidence,
            risk_drivers=tuple(risk_drivers),
            counterfactual_summary=counterfactual_summary,
            nearest_legitimate=nearest_legitimate,
            minimal_changes=tuple(minimal_changes),
            raw_shap=shap,
            raw_counterfactual=counterfactual,
        )

    def _format_risk_drivers(self, shap: Optional[SHAPExplanation]) -> list[str]:
        """Convert SHAP attributions into natural language risk drivers."""
        if shap is None or not shap.top_features:
            return ["Insufficient feature attribution data"]

        drivers = []
        for attr in shap.top_features[: self.max_drivers]:
            direction = "increases" if attr.direction == "increase" else "decreases"

            # Format value for readability
            value_str = self._format_value(attr.value)

            if abs(attr.impact) > 0.1:
                strength = "significantly"
            elif abs(attr.impact) > 0.05:
                strength = "moderately"
            else:
                strength = "slightly"

            driver = f"{attr.feature} is {value_str}, which {strength} {direction} fraud risk"
            drivers.append(driver)

        return drivers

    def _format_counterfactual_summary(
        self, counterfactual: Optional[CounterfactualExplanation]
    ) -> Optional[str]:
        """Generate a summary of what would need to change."""
        if counterfactual is None or not counterfactual.changes:
            return None

        if counterfactual.source == "nearest_neighbor":
            return (
                f"Based on the nearest legitimate transaction "
                f"({counterfactual.nearest_neighbor.transaction_id if counterfactual.nearest_neighbor else 'N/A'}), "
                f"the following changes would reduce fraud risk:"
            )
        elif counterfactual.source == "dice":
            return "Optimal changes to reduce fraud risk:"
        else:
            return "Suggested changes to reduce fraud risk:"

    def _format_nearest_legitimate(
        self, counterfactual: Optional[CounterfactualExplanation]
    ) -> Optional[str]:
        """Format the nearest legitimate transaction reference."""
        if counterfactual is None or counterfactual.nearest_neighbor is None:
            return None

        nn = counterfactual.nearest_neighbor
        parts = [f"Nearest legitimate behaviour (txn {nn.transaction_id}):"]

        # Show top 3 most different features
        top_changes = sorted(
            counterfactual.changes,
            key=lambda c: abs(c.current_value - c.counterfactual_value),
            reverse=True,
        )[:3]

        for change in top_changes:
            current = self._format_value(change.current_value)
            target = self._format_value(change.counterfactual_value)
            parts.append(f"  {change.feature}: {current} → {target}")

        return "\n".join(parts)

    def _format_minimal_changes(
        self, counterfactual: Optional[CounterfactualExplanation]
    ) -> list[str]:
        """List the minimal changes required to flip the decision."""
        if counterfactual is None or not counterfactual.changes:
            return []

        changes = []
        for change in counterfactual.changes[: self.max_drivers]:
            current = self._format_value(change.current_value)
            target = self._format_value(change.counterfactual_value)
            changes.append(f"Change {change.feature} from {current} to {target}")

        return changes

    @staticmethod
    def _format_value(val: float) -> str:
        """Format a numeric value for human readability."""
        if abs(val) >= 1_000_000:
            return f"{val/1_000_000:.1f}M"
        if abs(val) >= 1_000:
            return f"{val/1_000:.1f}K"
        if abs(val) < 0.01:
            return f"{val:.4f}"
        return f"{val:.2f}"
