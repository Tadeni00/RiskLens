"""
RiskLens Behavioral Intelligence Layer
Feature Generation Package
"""

from behavior.feature_generation.generator import generate_behavioral_features
from behavior.feature_generation.velocity import compute_velocity_features

__all__ = [
    "generate_behavioral_features",
    "compute_velocity_features",
]
