"""
Enhanced embedding operations with port-preserving options.

This module provides embedding functionality for creating domain-with-void
meshes, with support for port preservation and feature constraints.

Re-exports both new enhanced API and legacy API for backward compatibility.
"""

# New enhanced API
from .enhanced_embedding import (
    embed_with_port_preservation,
    embed_void_mesh_as_negative_space,  # C1 FIX: Real mesh-based function, not alias
    get_port_constraints,
    EnhancedEmbeddingPolicy,
    PortPreservationPolicy,
    EmbeddingReport,
)

# Legacy API (from embedding_legacy.py)
from ..embedding_legacy import (
    embed_tree_as_negative_space,  # Takes STL path, not in-memory mesh
    VoxelBudgetExceededError,
)

__all__ = [
    # New enhanced API (mesh-based)
    "embed_with_port_preservation",
    "embed_void_mesh_as_negative_space",  # C1 FIX: Takes in-memory mesh
    "get_port_constraints",
    "EnhancedEmbeddingPolicy",
    "PortPreservationPolicy",
    "EmbeddingReport",
    # Legacy API (file-based)
    "embed_tree_as_negative_space",  # Takes STL path
    "VoxelBudgetExceededError",
]
