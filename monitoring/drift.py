"""
FraudTrap — Drift Detection
Computes population stability, KL divergence, embedding drift, and concept drift.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import kl_div
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger


@dataclass
class DriftResult:
    """Result of drift computation for a single feature."""
    feature: str
    psi: float
    kl_divergence: float
    mean_shift: float
    std_shift: float
    drift_detected: bool
    baseline_stats: dict
    current_stats: dict


@dataclass
class EmbeddingDriftResult:
    """Result of embedding drift computation."""
    tenant_id: str
    centroid_distance: float
    max_distance: float
    mean_distance: float
    drift_detected: bool
    baseline_centroid: Optional[np.ndarray] = None
    current_centroid: Optional[np.ndarray] = None


@dataclass
class ConceptDriftResult:
    """Result of concept drift detection."""
    tenant_id: str
    label_rate_baseline: float
    label_rate_current: float
    rate_change: float
    drift_detected: bool
    prediction_rate_baseline: Optional[float] = None
    prediction_rate_current: Optional[float] = None


def compute_psi(
    baseline: np.ndarray, 
    current: np.ndarray, 
    bins: int = 10
) -> float:
    """
    Population Stability Index (PSI).
    
    PSI = sum((current% - baseline%) * ln(current% / baseline%))
    
    Args:
        baseline: Reference distribution values
        current: Current distribution values
        bins: Number of quantile bins
        
    Returns:
        PSI value (0 = no drift, >0.1 = moderate, >0.25 = significant)
    """
    # Remove NaN
    baseline = baseline[~np.isnan(baseline)]
    current = current[~np.isnan(current)]
    
    if len(baseline) < 10 or len(current) < 10:
        return 0.0
    
    # Create bins from baseline quantiles
    try:
        bin_edges = np.percentile(baseline, np.linspace(0, 100, bins + 1))
        # Ensure unique edges
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 3:
            return 0.0
        
        # Adjust edges slightly to avoid boundary issues
        bin_edges[0] -= 1e-6
        bin_edges[-1] += 1e-6
        
        # Histogram counts
        base_counts, _ = np.histogram(baseline, bins=bin_edges)
        curr_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Convert to percentages
        base_pct = base_counts / len(baseline)
        curr_pct = curr_counts / len(current)
        
        # Avoid log(0) and division by zero
        base_pct = np.clip(base_pct, 1e-6, 1.0)
        curr_pct = np.clip(curr_pct, 1e-6, 1.0)
        
        psi = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))
        return float(max(0.0, psi))
    except Exception as exc:
        logger.warning("PSI computation failed: {}", exc)
        return 0.0


def compute_kl_divergence(
    baseline: np.ndarray, 
    current: np.ndarray, 
    bins: int = 20
) -> float:
    """
    KL Divergence between baseline and current distributions.
    
    KL(P||Q) = sum(P(x) * log(P(x)/Q(x)))
    
    Args:
        baseline: Reference distribution
        current: Current distribution
        bins: Number of histogram bins
        
    Returns:
        KL divergence (0 = identical, higher = more divergence)
    """
    baseline = baseline[~np.isnan(baseline)]
    current = current[~np.isnan(current)]
    
    if len(baseline) < 20 or len(current) < 20:
        return 0.0
    
    try:
        # Use histogram-based density estimation
        bin_edges = np.linspace(
            min(baseline.min(), current.min()),
            max(baseline.max(), current.max()),
            bins + 1
        )
        
        base_hist, _ = np.histogram(baseline, bins=bin_edges, density=True)
        curr_hist, _ = np.histogram(current, bins=bin_edges, density=True)
        
        # Avoid zero bins
        base_hist = np.clip(base_hist, 1e-10, None)
        curr_hist = np.clip(curr_hist, 1e-10, None)
        
        # Normalize
        base_hist = base_hist / base_hist.sum()
        curr_hist = curr_hist / curr_hist.sum()
        
        kl = np.sum(kl_div(base_hist, curr_hist))
        return float(max(0.0, kl))
    except Exception as exc:
        logger.warning("KL divergence computation failed: {}", exc)
        return 0.0


def compute_feature_drift(
    feature: str,
    baseline_vals: np.ndarray,
    current_vals: np.ndarray,
    psi_threshold: float = 0.1,
    kl_threshold: float = 0.1
) -> DriftResult:
    """
    Compute comprehensive drift metrics for a single feature.
    
    Args:
        feature: Feature name
        baseline_vals: Baseline values
        current_vals: Current values
        psi_threshold: PSI threshold for drift detection
        kl_threshold: KL threshold for drift detection
        
    Returns:
        DriftResult with all metrics
    """
    baseline_clean = baseline_vals[~np.isnan(baseline_vals)]
    current_clean = current_vals[~np.isnan(current_vals)]
    
    if len(baseline_clean) < 10 or len(current_clean) < 10:
        return DriftResult(
            feature=feature,
            psi=0.0,
            kl_divergence=0.0,
            mean_shift=0.0,
            std_shift=0.0,
            drift_detected=False,
            baseline_stats={},
            current_stats={}
        )
    
    psi = compute_psi(baseline_clean, current_clean)
    kl = compute_kl_divergence(baseline_clean, current_clean)
    
    mean_shift = float(np.mean(current_clean) - np.mean(baseline_clean))
    std_shift = float(np.std(current_clean) - np.std(baseline_clean))
    
    drift_detected = psi > psi_threshold or kl > kl_threshold
    
    return DriftResult(
        feature=feature,
        psi=psi,
        kl_divergence=kl,
        mean_shift=mean_shift,
        std_shift=std_shift,
        drift_detected=drift_detected,
        baseline_stats={
            "mean": float(np.mean(baseline_clean)),
            "std": float(np.std(baseline_clean)),
            "min": float(np.min(baseline_clean)),
            "max": float(np.max(baseline_clean)),
            "count": len(baseline_clean)
        },
        current_stats={
            "mean": float(np.mean(current_clean)),
            "std": float(np.std(current_clean)),
            "min": float(np.min(current_clean)),
            "max": float(np.max(current_clean)),
            "count": len(current_clean)
        }
    )


def compute_all_feature_drift(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_list: list[str],
    psi_threshold: float = 0.1,
    kl_threshold: float = 0.1
) -> dict[str, DriftResult]:
    """
    Compute drift for all specified features.
    
    Args:
        baseline_df: DataFrame with baseline period data
        current_df: DataFrame with current period data
        feature_list: List of features to check
        psi_threshold: PSI threshold
        kl_threshold: KL threshold
        
    Returns:
        Dict mapping feature name to DriftResult
    """
    results = {}
    
    for feature in feature_list:
        if feature not in baseline_df.columns or feature not in current_df.columns:
            continue
        
        baseline_vals = baseline_df[feature].values
        current_vals = current_df[feature].values
        
        results[feature] = compute_feature_drift(
            feature=feature,
            baseline_vals=baseline_vals,
            current_vals=current_vals,
            psi_threshold=psi_threshold,
            kl_threshold=kl_threshold
        )
    
    return results


def compute_embedding_drift(
    tenant_id: str,
    baseline_embeddings: np.ndarray,
    current_embeddings: np.ndarray,
    threshold: float = 0.3
) -> EmbeddingDriftResult:
    """
    Compute embedding drift using centroid distance.
    
    For GNN models: compares current account/transaction embeddings
    to baseline centroid.
    
    Args:
        tenant_id: Tenant identifier
        baseline_embeddings: [N, D] baseline embedding matrix
        current_embeddings: [M, D] current embedding matrix
        threshold: Cosine distance threshold for drift
        
    Returns:
        EmbeddingDriftResult with drift metrics
    """
    if len(baseline_embeddings) == 0 or len(current_embeddings) == 0:
        return EmbeddingDriftResult(
            tenant_id=tenant_id,
            centroid_distance=0.0,
            max_distance=0.0,
            mean_distance=0.0,
            drift_detected=False
        )
    
    # Compute centroids
    baseline_centroid = np.mean(baseline_embeddings, axis=0)
    current_centroid = np.mean(current_embeddings, axis=0)
    
    # Cosine distance between centroids
    norm_b = np.linalg.norm(baseline_centroid)
    norm_c = np.linalg.norm(current_centroid)
    
    if norm_b > 0 and norm_c > 0:
        cosine_sim = np.dot(baseline_centroid, current_centroid) / (norm_b * norm_c)
        centroid_distance = 1.0 - cosine_sim
    else:
        centroid_distance = 1.0
    
    # Per-embedding distances to baseline centroid
    if len(current_embeddings) > 0:
        # Normalize
        curr_norm = current_embeddings / np.linalg.norm(current_embeddings, axis=1, keepdims=True)
        base_norm = baseline_centroid / np.linalg.norm(baseline_centroid)
        
        distances = 1.0 - np.dot(curr_norm, base_norm)
        max_distance = float(np.max(distances))
        mean_distance = float(np.mean(distances))
    else:
        max_distance = 0.0
        mean_distance = 0.0
    
    drift_detected = centroid_distance > threshold
    
    return EmbeddingDriftResult(
        tenant_id=tenant_id,
        centroid_distance=centroid_distance,
        max_distance=max_distance,
        mean_distance=mean_distance,
        drift_detected=drift_detected,
        baseline_centroid=baseline_centroid,
        current_centroid=current_centroid
    )


def compute_concept_drift(
    tenant_id: str,
    baseline_labels: np.ndarray,
    current_labels: np.ndarray,
    baseline_predictions: Optional[np.ndarray] = None,
    current_predictions: Optional[np.ndarray] = None,
    threshold: float = 0.2
) -> ConceptDriftResult:
    """
    Compute concept drift by comparing label and prediction distributions.
    
    Concept drift = change in P(y|x) or P(y).
    
    Args:
        tenant_id: Tenant identifier
        baseline_labels: Binary labels from baseline period
        current_labels: Binary labels from current period
        baseline_predictions: Model predictions (scores) from baseline
        current_predictions: Model predictions (scores) from current
        threshold: Absolute rate change threshold
        
    Returns:
        ConceptDriftResult
    """
    # Filter valid labels
    base_labels = baseline_labels[~np.isnan(baseline_labels)]
    curr_labels = current_labels[~np.isnan(current_labels)]
    
    if len(base_labels) == 0 or len(curr_labels) == 0:
        return ConceptDriftResult(
            tenant_id=tenant_id,
            label_rate_baseline=0.0,
            label_rate_current=0.0,
            rate_change=0.0,
            drift_detected=False
        )
    
    label_rate_base = float(np.mean(base_labels))
    label_rate_curr = float(np.mean(curr_labels))
    rate_change = abs(label_rate_curr - label_rate_base)
    
    pred_rate_base = None
    pred_rate_curr = None
    
    if baseline_predictions is not None and current_predictions is not None:
        base_preds = baseline_predictions[~np.isnan(baseline_predictions)]
        curr_preds = current_predictions[~np.isnan(current_predictions)]
        
        if len(base_preds) > 0 and len(curr_preds) > 0:
            # Using 0.5 threshold for positive prediction rate
            pred_rate_base = float(np.mean(base_preds >= 0.5))
            pred_rate_curr = float(np.mean(curr_preds >= 0.5))
    
    drift_detected = rate_change > threshold
    
    return ConceptDriftResult(
        tenant_id=tenant_id,
        label_rate_baseline=label_rate_base,
        label_rate_current=label_rate_curr,
        rate_change=rate_change,
        drift_detected=drift_detected,
        prediction_rate_baseline=pred_rate_base,
        prediction_rate_current=pred_rate_curr
    )


def detect_feature_outliers(
    values: np.ndarray,
    method: str = "iqr",
    threshold: float = 3.0
) -> np.ndarray:
    """
    Detect outlier values in feature distribution.
    
    Args:
        values: Feature values
        method: 'iqr' or 'zscore'
        threshold: Multiplier for IQR or z-score threshold
        
    Returns:
        Boolean mask of outliers
    """
    values = values[~np.isnan(values)]
    
    if len(values) < 4:
        return np.zeros(len(values), dtype=bool)
    
    if method == "iqr":
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        return (values < lower) | (values > upper)
    
    elif method == "zscore":
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return np.zeros(len(values), dtype=bool)
        z = np.abs((values - mean) / std)
        return z > threshold
    
    return np.zeros(len(values), dtype=bool)


# Default monitored features (can be extended per tenant)
DEFAULT_MONITORED_FEATURES = [
    "amount",
    "amount_log",
    "amount_zscore",
    "acct_v_1m_count",
    "acct_v_1h_count",
    "acct_v_24h_count",
    "acct_v_24h_total_amt",
    "geo_speed_kmh",
    "impossible_travel",
    "device_account_count",
    "is_new_device",
    "is_new_merchant",
    "typing_zscore",
    "cross_country_flag",
    "hour_sin",
    "hour_cos",
    "is_night",
    "is_weekend",
]