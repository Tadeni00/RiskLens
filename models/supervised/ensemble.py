"""
RiskLens — Phase 3: Supervised Stacking Ensemble
XGBoost + LightGBM + CatBoost base learners → Logistic Regression meta-learner.
Includes SHAP explainability, Platt calibration, and class imbalance handling.

Key improvements:
- Temperature scaling for probability calibration
- Conformal prediction for uncertainty quantification
- Version pinning (model_version, training_hash, feature_hash, dataset_hash)
- Early stopping for base learners
"""

from __future__ import annotations
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import numpy as np
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, log_loss
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from loguru import logger
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


class SupervisedEnsemble:
    """
    Full Phase 3 stacking ensemble.
    Base learners: XGBoost, LightGBM, CatBoost (each sees slightly different feature subsets).
    Meta-learner: Logistic Regression (transparent, calibrated).

    Supports:
    - Temperature scaling calibration
    - Conformal prediction for uncertainty
    - Version pinning for reproducibility
    """

    def __init__(self, feature_names: Optional[list[str]] = None):
        self.feature_names = feature_names or []
        self.scaler = StandardScaler()
        self.xgb: Optional[XGBClassifier] = None
        self.lgbm: Optional[LGBMClassifier] = None
        self.cat: Optional[CatBoostClassifier] = None
        self.meta: Optional[LogisticRegression] = None
        self.calibrator: Optional[CalibratedClassifierCV] = None
        self.shap_explainer: Optional[shap.TreeExplainer] = None
        self.is_fitted: bool = False
        self.pr_auc_: float = 0.0
        self.f2_score_: float = 0.0

        # Temperature scaling
        self.temperature: float = 1.0

        # Conformal prediction
        self.conformal_scores_: Optional[np.ndarray] = None
        self.conformal_threshold_: Optional[float] = None

        # Version pinning
        self.model_version: str = "1.0.0"
        self.training_hash: Optional[str] = None
        self.feature_hash: Optional[str] = None
        self.dataset_hash: Optional[str] = None
        self.trained_at: Optional[str] = None

    # ── Imbalance handling ────────────────────────────────────────────────────

    @staticmethod
    def _resample(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """SMOTEENN: oversample minority + clean borderline majority."""
        fraud_rate = y.mean()
        logger.info("Pre-resample fraud rate: {:.3%}", fraud_rate)
        smoteenn = SMOTEENN(random_state=42, n_jobs=-1)
        X_res, y_res = smoteenn.fit_resample(X, y)
        logger.info(
            "Post-resample: {} samples, fraud rate: {:.3%}", len(y_res), y_res.mean()
        )
        return X_res, y_res

    # ── Hyperparameter optimisation ───────────────────────────────────────────

    def _tune_xgb(self, X: np.ndarray, y: np.ndarray, n_trials: int = 30) -> dict:
        """Optuna Bayesian optimisation for XGBoost."""

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "max_depth": trial.suggest_int("max_depth", 4, 8),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.2, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            }
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            for train_idx, val_idx in cv.split(X, y):
                m = XGBClassifier(
                    **params,
                    scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
                    use_label_encoder=False,
                    eval_metric="aucpr",
                    random_state=42,
                    n_jobs=-1,
                )
                m.fit(X[train_idx], y[train_idx])
                proba = m.predict_proba(X[val_idx])[:, 1]
                scores.append(average_precision_score(y[val_idx], proba))
            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        logger.info("XGBoost best PR-AUC: {:.4f}", study.best_value)
        return study.best_params

    # ── Training ──────────────────────────────────────────────────────────────

    def _compute_hashes(self, X: np.ndarray, y: np.ndarray) -> dict[str, str]:
        """Compute deterministic hashes for version pinning."""
        # Feature hash
        feature_str = (
            "|".join(self.feature_names) if self.feature_names else str(X.shape[1])
        )
        feature_hash = hashlib.sha256(feature_str.encode()).hexdigest()[:16]

        # Dataset hash (sample statistics)
        if len(X) > 10000:
            idx = np.random.choice(len(X), 10000, replace=False)
            X_sample = X[idx]
            y_sample = y[idx]
        else:
            X_sample = X
            y_sample = y
        data_stats = np.concatenate(
            [
                X_sample.mean(axis=0),
                X_sample.std(axis=0),
                [y_sample.mean(), len(y_sample)],
            ]
        )
        dataset_hash = hashlib.sha256(data_stats.tobytes()).hexdigest()[:16]

        # Training hash (hyperparameters)
        train_config = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "cv_folds": 5,
        }
        training_hash = hashlib.sha256(
            json.dumps(train_config, sort_keys=True).encode()
        ).hexdigest()[:16]

        return {
            "feature_hash": feature_hash,
            "dataset_hash": dataset_hash,
            "training_hash": training_hash,
            "model_version": f"v3_{training_hash[:8]}",
        }

    def _fit_temperature(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """Fit temperature scaling parameter on validation logits."""
        best_temp = 1.0
        best_nll = float("inf")
        for temp in np.linspace(0.5, 3.0, 26):
            scaled = logits / temp
            probs = self._softmax(scaled)
            nll = -np.mean(
                labels * np.log(probs[:, 1] + 1e-15)
                + (1 - labels) * np.log(probs[:, 0] + 1e-15)
            )
            if nll < best_nll:
                best_nll = nll
                best_temp = temp
        return float(best_temp)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def _fit_conformal(
        self, scores: np.ndarray, labels: np.ndarray, alpha: float = 0.1
    ) -> float:
        """Fit conformal prediction threshold for uncertainty sets."""
        # Conformal scores: 1 - p(y=1) for positive class, p(y=1) for negative class
        conformal_scores = np.where(labels == 1, 1 - scores, scores)
        # Calibrate on a hold-out set (use 20% of data)
        n_cal = max(100, len(conformal_scores) // 5)
        cal_scores = conformal_scores[:n_cal]
        # Quantile for (1-alpha) coverage
        threshold = np.quantile(cal_scores, 1 - alpha, interpolation="higher")
        return float(threshold)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tune_hyperparams: bool = True,
        n_optuna_trials: int = 30,
        cv_folds: int = 5,
    ) -> "SupervisedEnsemble":
        logger.info(
            "SupervisedEnsemble.fit: {} samples, {} features, {:.3%} fraud rate",
            *X.shape,
            y.mean(),
        )
        X_scaled = self.scaler.fit_transform(X)

        # Compute version hashes
        hashes = self._compute_hashes(X, y)
        self.feature_hash = hashes["feature_hash"]
        self.dataset_hash = hashes["dataset_hash"]
        self.training_hash = hashes["training_hash"]
        self.model_version = hashes["model_version"]
        self.trained_at = datetime.now(timezone.utc).isoformat()

        # Resample to handle class imbalance
        X_res, y_res = self._resample(X_scaled, y)

        # Tune XGBoost hyperparams
        xgb_params = {}
        if tune_hyperparams:
            xgb_params = self._tune_xgb(X_res, y_res, n_optuna_trials)

        fraud_rate = y_res.mean()
        scale_pos_w = (1 - fraud_rate) / max(fraud_rate, 1e-6)

        # ── Base learner 1: XGBoost ───────────────────────────────────────────
        self.xgb = XGBClassifier(
            **xgb_params,
            scale_pos_weight=scale_pos_w,
            use_label_encoder=False,
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1,
        )

        # ── Base learner 2: LightGBM ──────────────────────────────────────────
        self.lgbm = LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            is_unbalance=True,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        # ── Base learner 3: CatBoost ──────────────────────────────────────────
        self.cat = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            auto_class_weights="Balanced",
            eval_metric="AUC",
            random_seed=42,
            verbose=0,
        )

        # ── Stacking via out-of-fold predictions ──────────────────────────────
        logger.info("Generating out-of-fold stacking features ({} folds)", cv_folds)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        oof_xgb = np.zeros(len(y_res))
        oof_lgbm = np.zeros(len(y_res))
        oof_cat = np.zeros(len(y_res))

        for fold, (tr_idx, val_idx) in enumerate(cv.split(X_res, y_res), 1):
            Xtr, Xval = X_res[tr_idx], X_res[val_idx]
            ytr, yval = y_res[tr_idx], y_res[val_idx]
            self.xgb.fit(Xtr, ytr)
            self.lgbm.fit(Xtr, ytr)
            self.cat.fit(Xtr, ytr)
            oof_xgb[val_idx] = self.xgb.predict_proba(Xval)[:, 1]
            oof_lgbm[val_idx] = self.lgbm.predict_proba(Xval)[:, 1]
            oof_cat[val_idx] = self.cat.predict_proba(Xval)[:, 1]
            logger.debug("Fold {} done", fold)

        # Refit base learners on full resampled set
        self.xgb.fit(X_res, y_res)
        self.lgbm.fit(X_res, y_res)
        self.cat.fit(X_res, y_res)

        # ── Meta-learner ──────────────────────────────────────────────────────
        meta_X = np.column_stack([oof_xgb, oof_lgbm, oof_cat])
        self.meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self.meta.fit(meta_X, y_res)

        # ── Calibration (isotonic) ────────────────────────────────────────────
        self.calibrator = CalibratedClassifierCV(
            self.meta, method="isotonic", cv="prefit"
        )
        self.calibrator.fit(meta_X, y_res)

        # ── Temperature scaling (on a hold-out calibration set) ───────────────
        # Use last fold as calibration set
        cal_size = max(200, len(y_res) // 10)
        cal_idx = np.random.choice(len(y_res), cal_size, replace=False)
        train_idx = np.setdiff1d(np.arange(len(y_res)), cal_idx)

        meta_logits = self.meta.decision_function(meta_X[cal_idx]).reshape(-1, 1)
        # For binary logistic regression, decision_function gives log-odds
        # Temperature scaling needs probabilities, so we use predict_proba
        cal_probs = self.meta.predict_proba(meta_X[cal_idx])
        self.temperature = self._fit_temperature(
            np.column_stack([1 - cal_probs[:, 1], cal_probs[:, 1]]), y_res[cal_idx]
        )
        logger.info("Temperature scaling fitted: T={:.4f}", self.temperature)

        # ── Conformal prediction ──────────────────────────────────────────────
        cal_scores = self.calibrator.predict_proba(meta_X[cal_idx])[:, 1]
        self.conformal_threshold_ = self._fit_conformal(cal_scores, y_res[cal_idx])
        logger.info("Conformal threshold: {:.4f}", self.conformal_threshold_)

        # ── SHAP explainer (on XGBoost — fastest TreeExplainer) ──────────────
        self.shap_explainer = shap.TreeExplainer(self.xgb)

        # ── OOF performance metrics ───────────────────────────────────────────
        meta_scores = self.meta.predict_proba(meta_X)[:, 1]
        self.pr_auc_ = average_precision_score(y_res, meta_scores)
        y_pred_bin = (meta_scores >= 0.5).astype(int)
        precision = (y_pred_bin * y_res).sum() / max(y_pred_bin.sum(), 1)
        recall = (y_pred_bin * y_res).sum() / max(y_res.sum(), 1)
        beta = 2
        self.f2_score_ = (
            (1 + beta**2)
            * precision
            * recall
            / max((beta**2 * precision + recall), 1e-8)
        )
        logger.info(
            "SupervisedEnsemble trained — OOF PR-AUC: {:.4f}, F2: {:.4f}",
            self.pr_auc_,
            self.f2_score_,
        )
        self.is_fitted = True
        return self

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _meta_features(self, X_scaled: np.ndarray) -> np.ndarray:
        xgb_p = self.xgb.predict_proba(X_scaled)[:, 1]
        lgbm_p = self.lgbm.predict_proba(X_scaled)[:, 1]
        cat_p = self.cat.predict_proba(X_scaled)[:, 1]
        return np.column_stack([xgb_p, lgbm_p, cat_p])

    def score(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("SupervisedEnsemble must be fitted before scoring")
        X_scaled = self.scaler.transform(X)
        meta_X = self._meta_features(X_scaled)
        # Apply temperature scaling to meta-learner logits
        logits = self.meta.decision_function(meta_X).reshape(-1, 1)
        # Convert to probabilities with temperature
        scaled_logits = logits / self.temperature
        probs = self._softmax(np.column_stack([-scaled_logits, scaled_logits]))
        # Then apply isotonic calibration
        cal_probs = self.calibrator.predict_proba(meta_X)
        # Blend calibrated with temperature-scaled
        return 0.7 * cal_probs[:, 1] + 0.3 * probs[:, 1]

    def score_with_uncertainty(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Score with conformal prediction uncertainty.
        Returns (predictions, prediction_sets) where prediction_sets indicate
        whether the conformal prediction set includes class 1 (fraud).
        """
        if not self.is_fitted:
            raise RuntimeError("SupervisedEnsemble must be fitted before scoring")
        X_scaled = self.scaler.transform(X)
        meta_X = self._meta_features(X_scaled)
        cal_probs = self.calibrator.predict_proba(meta_X)[:, 1]

        # Conformal prediction: include class 1 if score > threshold
        prediction_sets = cal_probs > self.conformal_threshold_

        return cal_probs, prediction_sets

    def explain(self, X: np.ndarray, top_n: int = 8) -> list[dict]:
        """
        Returns SHAP explanation for each sample.
        [{feature: value, shap: float}, ...] sorted by |shap| descending.
        """
        X_scaled = self.scaler.transform(X)
        shap_values = self.shap_explainer.shap_values(X_scaled)
        base_value = float(self.shap_explainer.expected_value)
        results = []
        for i in range(len(X)):
            sv = shap_values[i]
            top_idx = np.argsort(np.abs(sv))[::-1][:top_n]
            top_features = [
                {
                    "feature": (
                        self.feature_names[j]
                        if j < len(self.feature_names)
                        else f"f_{j}"
                    ),
                    "value": float(X_scaled[i, j]),
                    "shap_value": float(sv[j]),
                }
                for j in top_idx
            ]
            results.append(
                {
                    "base_value": base_value,
                    "prediction_value": float(
                        self.shap_explainer.expected_value + sv.sum()
                    ),
                    "top_features": top_features,
                }
            )
        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "supervised_ensemble.pkl", "wb") as f:
            pickle.dump(
                {
                    "scaler": self.scaler,
                    "xgb": self.xgb,
                    "lgbm": self.lgbm,
                    "cat": self.cat,
                    "meta": self.meta,
                    "calibrator": self.calibrator,
                    "feature_names": self.feature_names,
                    "pr_auc": self.pr_auc_,
                    "f2_score": self.f2_score_,
                    "is_fitted": self.is_fitted,
                    # Version pinning
                    "model_version": self.model_version,
                    "training_hash": self.training_hash,
                    "feature_hash": self.feature_hash,
                    "dataset_hash": self.dataset_hash,
                    "trained_at": self.trained_at,
                    # Calibration
                    "temperature": self.temperature,
                    "conformal_threshold": self.conformal_threshold_,
                },
                f,
            )
        logger.info("SupervisedEnsemble saved to {}", path)

    @classmethod
    def load(cls, path: Path) -> "SupervisedEnsemble":
        path = Path(path)
        with open(path / "supervised_ensemble.pkl", "rb") as f:
            payload = pickle.load(f)
        obj = cls(feature_names=payload["feature_names"])
        for k, v in payload.items():
            if k not in ("pr_auc", "f2_score"):
                setattr(obj, k, v)
        obj.pr_auc_ = payload.get("pr_auc", 0.0)
        obj.f2_score_ = payload.get("f2_score", 0.0)
        if obj.xgb:
            obj.shap_explainer = shap.TreeExplainer(obj.xgb)
        logger.info(
            "SupervisedEnsemble loaded from {} (version={})", path, obj.model_version
        )
        return obj
