"""
Face feature framework for domain mesh modifications.

This module provides a unified framework for creating features on domain faces,
such as ridges, grooves, and ports.
"""

from .face_feature import (
    FaceFeature,
    FeatureType,
    FaceId,
    FeatureConstraints,
    FeatureResult,
    RidgeFeature,
    RidgeFeatureSpec,
    create_ridge_with_constraints,
)
from .ridge_helpers import (
    create_annular_ridge,
    create_frame_ridge,
)

__all__ = [
    "FaceFeature",
    "FeatureType",
    "FaceId",
    "FeatureConstraints",
    "FeatureResult",
    "RidgeFeature",
    "RidgeFeatureSpec",
    "create_ridge_with_constraints",
    "create_annular_ridge",
    "create_frame_ridge",
]
