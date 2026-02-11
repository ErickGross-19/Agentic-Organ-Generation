"""
Hierarchical tissue point system for Optimized Directed Colonization (ODC).

This package provides data structures and utilities for defining tissue points
organized by priority levels, where the vascular network must reach
higher-priority points before lower-priority ones.
"""

from .hierarchical import TissueLevel, HierarchicalTissueSpec
from .coverage import (
    LevelCoverageResult,
    HierarchicalCoverageResult,
    compute_hierarchical_coverage,
)
from .samplers import (
    sample_hierarchical_tissue_points,
    generate_hierarchical_from_strategy,
)

__all__ = [
    "TissueLevel",
    "HierarchicalTissueSpec",
    "LevelCoverageResult",
    "HierarchicalCoverageResult",
    "compute_hierarchical_coverage",
    "sample_hierarchical_tissue_points",
    "generate_hierarchical_from_strategy",
]
