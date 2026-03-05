"""
AOG (Agentic Organ Generation) vascular network integration for MorphoStruct.

This package provides integration between AOG's advanced vascular generation
algorithms and MorphoStruct's web-based scaffold design platform.
"""

from .space_colonization import generate_space_colonization_from_dict
from .bifurcating_tree import generate_bifurcating_tree_from_dict

__all__ = [
    "generate_space_colonization_from_dict",
    "generate_bifurcating_tree_from_dict",
]
