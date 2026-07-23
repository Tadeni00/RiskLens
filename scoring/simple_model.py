from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class SimpleFraudModel:
    """Small numpy-only logistic model for the serving runtime."""

    def __init__(
        self,
        feature_names: list[str],
        weights: np.ndarray,
        bias: float,
        mean: np.ndarray,
        scale: np.ndarray,
        model_version: str,
        calibration_raw: np.ndarray | None = None,
        calibration_score: np.ndarray | None = None,
    ):
        self.feature_names = feature_names
        self.weights = weights.astype(np.float32)
        self.bias = float(bias)
        self.mean = mean.astype(np.float32)
        self.scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
        self.model_version = model_version
        self.calibration_raw = (
            calibration_raw.astype(np.float32) if calibration_raw is not None else None
        )
        self.calibration_score = (
            calibration_score.astype(np.float32)
            if calibration_score is not None
            else None
        )
        self.is_fitted = True

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        return (X.astype(np.float32) - self.mean) / self.scale

    def score(self, X: np.ndarray) -> np.ndarray:
        Xs = self._prepare(X)
        logits = Xs @ self.weights + self.bias
        raw = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        if self.calibration_raw is None or self.calibration_score is None:
            return raw
        return np.interp(raw, self.calibration_raw, self.calibration_score).astype(
            np.float32
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "feature_names": self.feature_names,
                    "weights": self.weights,
                    "bias": self.bias,
                    "mean": self.mean,
                    "scale": self.scale,
                    "model_version": self.model_version,
                    "calibration_raw": self.calibration_raw,
                    "calibration_score": self.calibration_score,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "SimpleFraudModel":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        payload.setdefault("calibration_raw", None)
        payload.setdefault("calibration_score", None)
        return cls(**payload)
