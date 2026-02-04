"""
Face feature framework for domain mesh modifications.

This module provides a unified framework for creating features on domain faces,
such as ridges, grooves, and ports.

Re-exports both new framework API and legacy API for backward compatibility.
"""

# New framework API
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

# Legacy API (from features_legacy.py)
# Note: FaceId, create_annular_ridge, create_frame_ridge are already exported above
# Import additional legacy exports
from ..features_legacy import (
    RidgeSpec,
    add_raised_ridge,
)

__all__ = [
    # New framework API
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
    # Legacy API
    "RidgeSpec",
    "add_raised_ridge",
]
