"""
Enhanced embedding operations with port-preserving options.

This module provides embedding functionality for creating domain-with-void
meshes, with support for port preservation and feature constraints.

Re-exports both new enhanced API and legacy API for backward compatibility.
"""

# New enhanced API
from .enhanced_embedding import (
    embed_with_port_preservation,
    get_port_constraints,
    EnhancedEmbeddingPolicy,
    PortPreservationPolicy,
    EmbeddingReport,
)

# Legacy API (from embedding_legacy.py)
from ..embedding_legacy import (
    embed_tree_as_negative_space,
    VoxelBudgetExceededError,
)

# Alias for new primary function
embed_void_mesh_as_negative_space = embed_tree_as_negative_space

__all__ = [
    # New enhanced API
    "embed_with_port_preservation",
    "get_port_constraints",
    "EnhancedEmbeddingPolicy",
    "PortPreservationPolicy",
    "EmbeddingReport",
    # Legacy API
    "embed_tree_as_negative_space",
    "VoxelBudgetExceededError",
    # New primary alias
    "embed_void_mesh_as_negative_space",
]
