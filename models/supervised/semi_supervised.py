"""
RiskLens — Phase 2: Semi-Supervised Bridge
Pseudo-labelling + label propagation via transaction graph.
Transitions when readiness gates are met.

Key improvements:
- Adaptive pseudo-label thresholds that evolve as confirmed labels grow
- Dynamic blending weights based on label quality
- Uncertainty-aware pseudo-labeling
"""

from __future__ import annotations
import pickle
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from loguru import logger

from models.cold_start.ensemble import ColdStartEnsemble


class SemiSupervisedBridge:
    """
    Phase 2 model.
    Uses confirmed labels + high-confidence pseudo-labels from Phase 1
    to train a lightweight XGBoost that progressively replaces the
    unsupervised ensemble.

    Adaptive thresholds:
    - Week 1 (few labels): 0.99 / 0.05 (very conservative)
    - Week 5: 0.96 / 0.10
    - Week 12+: 0.93 / 0.15 (more aggressive as model improves)
    """

    # Base thresholds (will be adapted based on confirmed label count)
    BASE_HIGH_CONF_START = 0.99
    BASE_HIGH_CONF_FLOOR = 0.93
    BASE_LOW_CONF_START = 0.05
    BASE_LOW_CONF_CEILING = 0.15

    # Pseudo-label thresholds (adaptive)
    PSEUDO_HIGH_CONF_BASE = 0.92  # above → pseudo-fraud label
    PSEUDO_LOW_CONF_BASE = 0.15  # below → pseudo-legit label
    # 0.15–0.92 = uncertainty zone → routed to human review queue

    def __init__(self, cold_start: ColdStartEnsemble):
        self.cold_start = cold_start
        self.scaler = StandardScaler()
        self.xgb: Optional[XGBClassifier] = None
        self.calibrator: Optional[CalibratedClassifierCV] = None
        self.is_fitted: bool = False
        self.pseudo_label_count: int = 0
        self.confirmed_label_count: int = 0
        self.training_iteration: int = 0

    def _adaptive_thresholds(self) -> tuple[float, float, float, float]:
        """
        Compute adaptive thresholds based on confirmed label count.
        As we get more confirmed labels, we can be more aggressive with pseudo-labeling.
        """
        # Progress from 0 to 1 as confirmed labels grow from 0 to 10000
        progress = min(1.0, self.confirmed_label_count / 10000.0)

        high_conf = self.BASE_HIGH_CONF_START - progress * (
            self.BASE_HIGH_CONF_START - self.BASE_HIGH_CONF_FLOOR
        )
        low_conf = self.BASE_LOW_CONF_START + progress * (
            self.BASE_LOW_CONF_CEILING - self.BASE_LOW_CONF_START
        )

        # Pseudo-label thresholds also adapt
        pseudo_high = self.PSEUDO_HIGH_CONF_BASE - progress * 0.05  # 0.92 -> 0.87
        pseudo_low = self.PSEUDO_LOW_CONF_BASE + progress * 0.05  # 0.15 -> 0.20

        return high_conf, low_conf, pseudo_high, pseudo_low

    # ── Pseudo-label generation ───────────────────────────────────────────────

    def generate_pseudo_labels(
        self,
        X_unlabelled: np.ndarray,
        transaction_ids: Optional[list[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Score unlabelled data with Phase 1 ensemble.
        Returns (X_pseudo, y_pseudo, review_queue_ids).
        Only high/low confidence samples are pseudo-labelled.
        Uncertain samples are returned as review_queue_ids for human labelling.
        Uses adaptive thresholds based on confirmed label count.
        """
        scores = self.cold_start.score(X_unlabelled)

        # Get adaptive thresholds
        _, _, pseudo_high, pseudo_low = self._adaptive_thresholds()

        high_mask = scores >= pseudo_high
        low_mask = scores <= pseudo_low
        confident_mask = high_mask | low_mask

        X_pseudo = X_unlabelled[confident_mask]
        y_pseudo = (scores[confident_mask] >= pseudo_high).astype(int)

        review_idx = np.where(~confident_mask)[0]
        review_ids = (
            [transaction_ids[i] for i in review_idx]
            if transaction_ids
            else list(review_idx.tolist())
        )

        self.pseudo_label_count = int(confident_mask.sum())
        logger.info(
            "Pseudo-labels (adaptive thresholds: high={:.2f}, low={:.2f}): "
            "{} fraud, {} legit, {} sent to review queue",
            pseudo_high,
            pseudo_low,
            int(high_mask.sum()),
            int(low_mask.sum()),
            len(review_ids),
        )
        return X_pseudo, y_pseudo, review_ids

    # ── Label propagation (graph-based) ──────────────────────────────────────

    @staticmethod
    def propagate_labels(
        confirmed_fraud_ids: set[str],
        adjacency: dict[str, list[str]],
        hop: int = 2,
        decay: float = 0.5,
    ) -> dict[str, float]:
        """
        BFS label propagation from confirmed fraud nodes.
        Returns {transaction_id: soft_label_weight}.
        Each hop decays the label strength by `decay`.
        """
        propagated: dict[str, float] = {}
        queue = [(fid, 1.0) for fid in confirmed_fraud_ids]
        visited = set(confirmed_fraud_ids)

        for _ in range(hop):
            next_queue = []
            for node, weight in queue:
                for neighbour in adjacency.get(node, []):
                    if neighbour not in visited:
                        new_weight = weight * decay
                        propagated[neighbour] = max(
                            propagated.get(neighbour, 0.0), new_weight
                        )
                        next_queue.append((neighbour, new_weight))
                        visited.add(neighbour)
            queue = next_queue

        return propagated

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X_confirmed: np.ndarray,
        y_confirmed: np.ndarray,
        X_unlabelled: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> "SemiSupervisedBridge":
        """
        Combine confirmed labels + pseudo-labels, then train XGBoost.
        """
        self.confirmed_label_count = len(y_confirmed)
        self.training_iteration += 1

        if X_unlabelled is not None and len(X_unlabelled) > 0:
            X_pseudo, y_pseudo, _ = self.generate_pseudo_labels(X_unlabelled)
            if len(X_pseudo) > 0:
                X_train = np.vstack([X_confirmed, X_pseudo])
                # Confirmed labels weighted 3x pseudo-labels
                w_conf = np.ones(len(y_confirmed)) * 3.0
                w_pseudo = np.ones(len(y_pseudo)) * 1.0
                weights = np.concatenate([w_conf, w_pseudo])
                if sample_weights is not None:
                    weights[: len(y_confirmed)] *= sample_weights
                y_train = np.concatenate([y_confirmed, y_pseudo])
            else:
                X_train, y_train, weights = X_confirmed, y_confirmed, sample_weights
        else:
            X_train, y_train, weights = X_confirmed, y_confirmed, sample_weights

        X_scaled = self.scaler.fit_transform(X_train)

        fraud_rate = y_train.mean()
        scale_pos_weight = (1 - fraud_rate) / max(fraud_rate, 1e-6)

        self.xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
        )
        self.xgb.fit(X_scaled, y_train, sample_weight=weights)

        # Platt scaling calibration
        self.calibrator = CalibratedClassifierCV(
            self.xgb, method="sigmoid", cv="prefit"
        )
        self.calibrator.fit(X_scaled, y_train, sample_weight=weights)

        self.is_fitted = True
        logger.info(
            "SemiSupervisedBridge fitted (iter={}) — {} confirmed + {} pseudo labels",
            self.training_iteration,
            self.confirmed_label_count,
            self.pseudo_label_count,
        )
        return self

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Blended score: calibrated XGBoost (70%) + cold-start ensemble (30%).
        Blend weight shifts more toward XGBoost as confirmed labels grow.
        Also considers pseudo-label quality.
        """
        cold_scores = self.cold_start.score(X)

        if not self.is_fitted:
            return cold_scores

        X_scaled = self.scaler.transform(X)
        xgb_proba = self.calibrator.predict_proba(X_scaled)[:, 1]

        # Dynamic blending — more confirmed labels = more trust in XGBoost
        # Also factor in pseudo-label quality
        label_quality = self.confirmed_label_count / max(
            1, self.confirmed_label_count + self.pseudo_label_count
        )
        xgb_weight = min(0.70, 0.30 + 0.40 * label_quality)
        cold_weight = 1.0 - xgb_weight

        return xgb_weight * xgb_proba + cold_weight * cold_scores

    def explain(self, X: np.ndarray, top_n: int = 8) -> list[dict]:
        """
        Returns per-sample explanation blending cold-start + XGBoost contributions.
        Uses SHAP on XGBoost if available.
        """
        if not self.is_fitted:
            # Return cold-start only explanation
            return self.cold_start.explain(X, top_n=top_n)

        n_samples = X.shape[0]
        X_scaled = self.scaler.transform(X)

        # Get cold-start explanation
        cold_explanations = self.cold_start.explain(X, top_n=top_n)

        # Get XGBoost SHAP values
        try:
            import shap

            explainer = shap.TreeExplainer(self.xgb)
            shap_values = explainer.shap_values(X_scaled)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Class 1 (fraud)

            base_value = float(explainer.expected_value)
            if isinstance(base_value, (list, np.ndarray)):
                base_value = float(
                    base_value[1] if len(base_value) > 1 else base_value[0]
                )
        except Exception as exc:
            logger.warning("SHAP explanation failed for XGBoost: {}", exc)
            shap_values = np.zeros((n_samples, X_scaled.shape[1]))
            base_value = 0.0

        # Dynamic blend weights
        label_quality = self.confirmed_label_count / max(
            1, self.confirmed_label_count + self.pseudo_label_count
        )
        xgb_weight = min(0.70, 0.30 + 0.40 * label_quality)
        cold_weight = 1.0 - xgb_weight

        explanations = []
        for i in range(n_samples):
            # Cold-start component
            cold_comp = cold_explanations[i]
            cold_contrib = cold_weight * cold_comp["prediction_value"]

            # XGBoost component
            xgb_contrib = xgb_weight * (base_value + shap_values[i].sum())

            # Combined prediction
            combined = cold_contrib + xgb_contrib

            # Feature-level attributions (combine cold-start + SHAP)
            # Cold-start components
            cold_components = cold_comp.get("top_features", [])

            # XGBoost SHAP features
            xgb_feature_names = (
                self.xgb.feature_names_in_
                if hasattr(self.xgb, "feature_names_in_")
                else [f"f{j}" for j in range(X_scaled.shape[1])]
            )

            shap_pairs = list(zip(xgb_feature_names, shap_values[i]))
            shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

            top_features = []

            # Add cold-start components (weighted by cold_weight)
            for cf in cold_components[: top_n // 2]:
                top_features.append(
                    {
                        "feature": cf.get("feature", "cold_start_component"),
                        "value": cf.get("value", 0.0),
                        "contribution": cf.get("contribution", 0.0) * cold_weight,
                        "method": "cold_start_component",
                    }
                )

            # Add XGBoost SHAP features (weighted by xgb_weight)
            for feat, shap_val in shap_pairs[: top_n // 2]:
                top_features.append(
                    {
                        "feature": feat,
                        "value": (
                            float(X_scaled[i, list(xgb_feature_names).index(feat)])
                            if feat in xgb_feature_names
                            else 0.0
                        ),
                        "contribution": float(shap_val) * xgb_weight,
                        "method": "shap",
                    }
                )

            explanations.append(
                {
                    "model_type": "semi_supervised",
                    "base_value": base_value * xgb_weight,
                    "prediction_value": float(combined),
                    "top_features": top_features[:top_n],
                    "components": {
                        "cold_start": {
                            "weight": cold_weight,
                            "prediction": float(cold_comp["prediction_value"]),
                            "top_features": cold_comp.get("top_features", []),
                        },
                        "xgboost": {
                            "weight": xgb_weight,
                            "prediction": (
                                float(xgb_contrib / xgb_weight)
                                if xgb_weight > 0
                                else 0.0
                            ),
                            "base_value": base_value,
                            "shap_values": {
                                feat: float(shap_val)
                                for feat, shap_val in shap_pairs[:top_n]
                            },
                        },
                        "blend_weights": {
                            "cold_start": cold_weight,
                            "xgboost": xgb_weight,
                        },
                        "label_quality": label_quality,
                    },
                }
            )

        return explanations

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "semi_supervised.pkl", "wb") as f:
            pickle.dump(
                {
                    "scaler": self.scaler,
                    "xgb": self.xgb,
                    "calibrator": self.calibrator,
                    "is_fitted": self.is_fitted,
                    "pseudo_label_count": self.pseudo_label_count,
                    "confirmed_label_count": self.confirmed_label_count,
                    "training_iteration": self.training_iteration,
                },
                f,
            )
        logger.info("SemiSupervisedBridge saved to {}", path)

    @classmethod
    def load(cls, path: Path, cold_start: ColdStartEnsemble) -> "SemiSupervisedBridge":
        path = Path(path)
        with open(path / "semi_supervised.pkl", "rb") as f:
            payload = pickle.load(f)
        obj = cls(cold_start=cold_start)
        obj.scaler = payload["scaler"]
        obj.xgb = payload["xgb"]
        obj.calibrator = payload["calibrator"]
        obj.is_fitted = payload["is_fitted"]
        obj.pseudo_label_count = payload["pseudo_label_count"]
        obj.confirmed_label_count = payload["confirmed_label_count"]
        obj.training_iteration = payload.get("training_iteration", 0)
        logger.info("SemiSupervisedBridge loaded from {}", path)
        return obj
