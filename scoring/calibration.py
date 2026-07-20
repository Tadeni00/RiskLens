"""
FraudTrap — Probability Calibration
Isotonic Regression and Platt Scaling for probability calibration.
Supports the Champion-Challenger model architecture.
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Optional
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from loguru import logger


class ProbabilityCalibrator:
    """
    Probability calibrator for fraud scores.
    
    Supports:
    - Isotonic Regression (non-parametric, flexible)
    - Platt Scaling (parametric, faster inference)
    
    Usage:
        calibrator = ProbabilityCalibrator(method="isotonic")
        calibrator.fit(raw_probs, y_true)
        calibrated = calibrator.transform(new_raw_probs)
    """
    
    def __init__(self, method: str = "isotonic"):
        """
        Initialize calibrator.
        
        Args:
            method: "isotonic" or "platt"
        """
        if method not in ("isotonic", "platt"):
            raise ValueError(f"Unknown calibration method: {method}. Use 'isotonic' or 'platt'")
        
        self.method = method
        self._scaler: Optional[StandardScaler] = None
        self._isotonic: Optional[IsotonicRegression] = None
        self._platt_a: Optional[float] = None
        self._platt_b: Optional[float] = None
        self._fitted = False
    
    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        """
        Fit the calibrator on validation set predictions.
        
        Args:
            raw_probs: Raw model probabilities (uncalibrated)
            y_true: True binary labels (0=legit, 1=fraud)
        
        Returns:
            self
        """
        logger.info(
            "Fitting {} calibrator on {} samples, fraud rate: {:.3%}",
            self.method, len(y_true), y_true.mean()
        )
        
        # Clip probabilities to avoid numerical issues
        raw_probs = np.clip(raw_probs, 1e-7, 1 - 1e-7)
        
        if self.method == "isotonic":
            self._isotonic = IsotonicRegression(
                out_of_bounds="clip",
                y_min=0.0,
                y_max=1.0,
                increasing="auto"
            )
            self._isotonic.fit(raw_probs, y_true)
        elif self.method == "platt":
            self._fit_platt(raw_probs, y_true)
        
        self._fitted = True
        logger.info("Calibrator fitted successfully")
        return self
    
    def _fit_platt(self, raw_probs: np.ndarray, y_true: np.ndarray) -> None:
        """
        Fit Platt scaling: sigmoid(raw_prob) = 1 / (1 + exp(A * raw_prob + B))
        """
        # Convert to logits for numerical stability
        logits = np.log(raw_probs / (1 - raw_probs))
        logits = np.clip(logits, -10, 10)
        
        # Fit logistic regression on logits
        X = logits.reshape(-1, 1)
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        
        lr = LogisticRegression(C=1.0, random_state=42)
        lr.fit(X_scaled, y_true)
        
        self._platt_a = float(lr.coef_[0, 0])
        self._platt_b = float(lr.intercept_[0])
    
    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        """
        Transform raw probabilities to calibrated probabilities.
        
        Args:
            raw_probs: Raw model probabilities
        
        Returns:
            Calibrated probabilities
        """
        if not self._fitted:
            raise RuntimeError("Calibrator must be fitted before transform")
        
        raw_probs = np.clip(raw_probs, 1e-7, 1 - 1e-7)
        
        if self.method == "isotonic":
            return self._isotonic.predict(raw_probs).astype(np.float64)
        elif self.method == "platt":
            return self._transform_platt(raw_probs)
        
        raise ValueError(f"Unknown method: {self.method}")
    
    def _transform_platt(self, raw_probs: np.ndarray) -> np.ndarray:
        """Apply Platt scaling transformation."""
        logits = np.log(raw_probs / (1 - raw_probs))
        logits = np.clip(logits, -10, 10)
        
        X = logits.reshape(-1, 1)
        X_scaled = self._scaler.transform(X)
        
        # Apply linear transformation and sigmoid
        scaled_logits = self._platt_a * X_scaled + self._platt_b
        calibrated = 1.0 / (1.0 + np.exp(-scaled_logits))
        
        return calibrated.ravel().astype(np.float64)
    
    def fit_transform(self, raw_probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Fit calibrator and transform probabilities in one step."""
        self.fit(raw_probs, y_true)
        return self.transform(raw_probs)
    
    def save(self, path: Path) -> None:
        """Save calibrator to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        with open(path / "calibrator.pkl", "wb") as f:
            pickle.dump({
                "method": self.method,
                "scaler": self._scaler,
                "isotonic": self._isotonic,
                "platt_a": self._platt_a,
                "platt_b": self._platt_b,
                "fitted": self._fitted,
            }, f)
        
        logger.info("Calibrator saved to {}", path)
    
    @classmethod
    def load(cls, path: Path) -> "ProbabilityCalibrator":
        """Load calibrator from disk."""
        path = Path(path)
        
        with open(path / "calibrator.pkl", "rb") as f:
            payload = pickle.load(f)
        
        obj = cls(method=payload["method"])
        obj._scaler = payload["scaler"]
        obj._isotonic = payload["isotonic"]
        obj._platt_a = payload["platt_a"]
        obj._platt_b = payload["platt_b"]
        obj._fitted = payload["fitted"]
        
        logger.info("Calibrator loaded from {} (method={})", path, obj.method)
        return obj
    
    def get_params(self) -> dict:
        """Get calibrator parameters."""
        return {
            "method": self.method,
            "fitted": self._fitted,
            "platt_a": self._platt_a if self.method == "platt" else None,
            "platt_b": self._platt_b if self.method == "platt" else None,
        }
    
    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return f"ProbabilityCalibrator(method={self.method}, status={status})"


def calibrate_model(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: str = "isotonic"
) -> ProbabilityCalibrator:
    """
    Convenience function to calibrate an already-trained model.
    
    Args:
        model: Model with predict_proba or score method
        X_val: Validation features
        y_val: Validation labels
        method: "isotonic" or "platt"
    
    Returns:
        Fitted calibrator
    """
    # Get raw probabilities
    if hasattr(model, "predict_proba"):
        raw_probs = model.predict_proba(X_val)[:, 1]
    elif hasattr(model, "score"):
        raw_probs = model.score(X_val)
    else:
        raise ValueError("Model must have predict_proba or score method")
    
    calibrator = ProbabilityCalibrator(method=method)
    calibrator.fit(raw_probs, y_val)
    
    return calibrator


def evaluate_calibration(
    raw_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10
) -> dict:
    """
    Evaluate calibration quality using reliability diagram metrics.
    
    Args:
        raw_probs: Uncalibrated probabilities
        calibrated_probs: Calibrated probabilities
        y_true: True labels
        n_bins: Number of bins for reliability diagram
    
    Returns:
        Dictionary with calibration metrics
    """
    from sklearn.calibration import calibration_curve
    
    # Reliability diagram data
    raw_fraction_pos, raw_mean_predicted = calibration_curve(
        y_true, raw_probs, n_bins=n_bins, strategy="uniform"
    )
    cal_fraction_pos, cal_mean_predicted = calibration_curve(
        y_true, calibrated_probs, n_bins=n_bins, strategy="uniform"
    )
    
    # Expected Calibration Error (ECE)
    raw_ece = np.mean(np.abs(raw_fraction_pos - raw_mean_predicted))
    cal_ece = np.mean(np.abs(cal_fraction_pos - cal_mean_predicted))
    
    # Maximum Calibration Error (MCE)
    raw_mce = np.max(np.abs(raw_fraction_pos - raw_mean_predicted))
    cal_mce = np.max(np.abs(cal_fraction_pos - cal_mean_predicted))
    
    # Brier Score
    raw_brier = np.mean((raw_probs - y_true) ** 2)
    cal_brier = np.mean((calibrated_probs - y_true) ** 2)
    
    return {
        "raw_ece": float(raw_ece),
        "calibrated_ece": float(cal_ece),
        "raw_mce": float(raw_mce),
        "calibrated_mce": float(cal_mce),
        "raw_brier": float(raw_brier),
        "calibrated_brier": float(cal_brier),
        "improvement_ece": float(raw_ece - cal_ece),
        "improvement_brier": float(raw_brier - cal_brier),
        "raw_reliability": {
            "fraction_pos": raw_fraction_pos.tolist(),
            "mean_predicted": raw_mean_predicted.tolist(),
        },
        "calibrated_reliability": {
            "fraction_pos": cal_fraction_pos.tolist(),
            "mean_predicted": cal_mean_predicted.tolist(),
        },
    }
