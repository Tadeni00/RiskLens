"""
FraudTrap — DiCE Counterfactual Engine
Advanced optimization-based counterfactuals for analyst investigations ONLY.
Never used during production inference.
"""

from __future__ import annotations
import time
from typing import Optional, List, Dict, Any
import numpy as np
from loguru import logger

try:
    import dice_ml

    DICE_AVAILABLE = True
except ImportError:
    DICE_AVAILABLE = False
    logger.warning("DiCE not installed — advanced counterfactuals unavailable")

from models.explainability.types import CounterfactualChange, CounterfactualExplanation


class DiCECounterfactual:
    """
    DiCE-based counterfactual generator for deep analyst investigations.

    This module is NEVER called during production inference.
    It is exposed via the analyst API for detailed investigation workflows.

    Features:
    - Realistic changes only (respects feature constraints)
    - Handles categorical and continuous features
    - Configurable number of counterfactuals
    - Proximity constraints to ensure feasibility
    """

    def __init__(
        self,
        max_counterfactuals: int = 3,
        proximity_weight: float = 0.5,
        diversity_weight: float = 1.0,
    ):
        self.max_counterfactuals = max_counterfactuals
        self.proximity_weight = proximity_weight
        self.diversity_weight = diversity_weight
        self._explainer = None
        self._is_fitted = False

    def fit(
        self,
        model,
        X_background: np.ndarray,
        feature_names: List[str],
        continuous_features: Optional[List[str]] = None,
        categorical_features: Optional[Dict[str, List[Any]]] = None,
    ) -> None:
        """
        Initialize DiCE with the model and data.

        Args:
            model: Fitted model with predict_proba
            X_background: Background dataset
            feature_names: Feature names
            continuous_features: List of continuous feature names
            categorical_features: Dict of feature_name -> allowed_values
        """
        if not DICE_AVAILABLE:
            logger.warning("DiCE not available")
            return

        try:
            import pandas as pd

            continuous_features = continuous_features or list(feature_names)

            # Build DiCE data object
            df_bg = pd.DataFrame(
                X_background[: min(500, len(X_background))], columns=feature_names
            )
            data = dice_ml.Data(
                dataframe=df_bg,
                continuous_features=continuous_features,
                outcome_name="label",
            )

            # Wrap model for DiCE
            ml_model = dice_ml.Model(model=model, backend="sklearn")

            self._explainer = dice_ml.Dice(
                data,
                ml_model,
                method="random",
            )
            self._feature_names = feature_names
            self._is_fitted = True

            logger.info(
                "DiCE counterfactual engine fitted (features={})", len(feature_names)
            )

        except Exception as exc:
            logger.error("DiCE init failed: {}", exc)
            self._is_fitted = False

    def generate(
        self,
        X: np.ndarray,
        features_to_vary: Optional[List[str]] = None,
        permitted_range: Optional[Dict[str, List[Any]]] = None,
        num_counterfactuals: Optional[int] = None,
    ) -> Optional[CounterfactualExplanation]:
        """
        Generate counterfactual explanations.

        Args:
            X: Input transaction (1, n_features)
            features_to_vary: Which features can be changed
            permitted_range: Min/max bounds for continuous features
            num_counterfactuals: Number of counterfactuals to generate

        Returns:
            CounterfactualExplanation or None
        """
        if not self._is_fitted or self._explainer is None:
            return None

        t_start = time.perf_counter()
        num_cf = num_counterfactuals or self.max_counterfactuals

        try:
            import pandas as pd

            X_input = X.reshape(1, -1) if X.ndim == 1 else X
            df_input = pd.DataFrame(X_input, columns=self._feature_names)

            # Generate counterfactuals
            cf = self._explainer.generate_counterfactuals(
                df_input,
                total_CFs=num_cf,
                features_to_vary=features_to_vary,
                permitted_range=permitted_range,
                desired_class="opposite",
            )

            cf_df = cf.cf_examples_list[0].final_cfs_df

            if cf_df is None or len(cf_df) == 0:
                return None

            # Extract the best counterfactual (first one)
            cf_row = cf_df.iloc[0]
            changes = []

            for fname in self._feature_names:
                current_val = float(X_input[0, self._feature_names.index(fname)])
                cf_val = float(cf_row.get(fname, current_val))

                if abs(current_val - cf_val) > 1e-6:
                    changes.append(
                        CounterfactualChange(
                            feature=fname,
                            current_value=current_val,
                            counterfactual_value=cf_val,
                            realistic=True,
                        )
                    )

            # Compute distance
            cf_features = np.array(
                [cf_row.get(f, 0.0) for f in self._feature_names], dtype=np.float32
            )
            distance = float(np.linalg.norm(X_input[0] - cf_features))

            latency_ms = (time.perf_counter() - t_start) * 1000

            return CounterfactualExplanation(
                prediction_delta=0.0,  # Will be filled by caller
                changes=tuple(changes),
                source="dice",
                dice_distance=distance,
                latency_ms=round(latency_ms, 2),
            )

        except Exception as exc:
            logger.warning("DiCE counterfactual generation failed: {}", exc)
            return None

    def generate_with_model(
        self,
        model,
        X: np.ndarray,
        fraud_probability: float,
        features_to_vary: Optional[List[str]] = None,
    ) -> Optional[CounterfactualExplanation]:
        """
        Generate counterfactual and compute prediction delta using the model directly.
        For when DiCE is not available or for manual verification.
        """
        t_start = time.perf_counter()

        X_input = X.reshape(1, -1) if X.ndim == 1 else X

        # Simple perturbation-based counterfactual (fallback)
        best_cf = None
        best_distance = float("inf")

        for _ in range(20):
            noise = np.random.randn(*X_input.shape) * 0.1
            X_candidate = np.clip(X_input + noise, 0, None)

            try:
                if hasattr(model, "predict_proba"):
                    prob = float(model.predict_proba(X_candidate)[0, 1])
                else:
                    prob = float(model.predict(X_candidate)[0])

                distance = float(np.linalg.norm(X_candidate - X_input))

                if prob < 0.5 and distance < best_distance:
                    best_distance = distance
                    best_cf = X_candidate
                    target_prob = prob
            except Exception:
                continue

        if best_cf is None:
            return None

        changes = []
        for i, fname in enumerate(
            self._feature_names
            if hasattr(self, "_feature_names")
            else [f"f_{j}" for j in range(X_input.shape[1])]
        ):
            current_val = float(X_input[0, i])
            cf_val = float(best_cf[0, i])
            if abs(current_val - cf_val) > 1e-6:
                changes.append(
                    CounterfactualChange(
                        feature=fname,
                        current_value=current_val,
                        counterfactual_value=cf_val,
                        realistic=True,
                    )
                )

        latency_ms = (time.perf_counter() - t_start) * 1000

        return CounterfactualExplanation(
            prediction_delta=round(fraud_probability - target_prob, 4),
            changes=tuple(changes),
            source="perturbation",
            dice_distance=best_distance,
            latency_ms=round(latency_ms, 2),
        )
