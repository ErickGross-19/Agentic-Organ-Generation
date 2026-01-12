"""
Optimization module for vascular network refinement.

Provides Jessen-style global geometry optimization via NLP and
discrete topology optimization with subtree swaps.

This module implements:
1. Global geometry NLP optimization (node positions, radii, pressures)
2. Discrete topology optimization via subtree swapping with SA acceptance

Note: The library uses METERS internally for all geometry.
"""

from .nlp_geometry import (
    NLPConfig,
    NLPResult,
    optimize_geometry,
    build_nlp_problem,
)
from .topology_swaps import (
    TopologyConfig,
    TopologyResult,
    optimize_topology,
    perform_subtree_swap,
)
from .refine import (
    refine_geometry,
    refine_topology,
    refine_network,
    RefinementConfig,
    RefinementResult,
)

__all__ = [
    "NLPConfig",
    "NLPResult",
    "optimize_geometry",
    "build_nlp_problem",
    "TopologyConfig",
    "TopologyResult",
    "optimize_topology",
    "perform_subtree_swap",
    "refine_geometry",
    "refine_topology",
    "refine_network",
    "RefinementConfig",
    "RefinementResult",
]
