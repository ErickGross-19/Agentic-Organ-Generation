"""
Enhanced embedding operations with port-preserving options.

This module provides embedding functionality for creating domain-with-void
meshes, with support for port preservation and feature constraints.
"""

from .enhanced_embedding import (
    embed_with_port_preservation,
    get_port_constraints,
    EnhancedEmbeddingPolicy,
    PortPreservationPolicy,
    EmbeddingReport,
)

__all__ = [
    "embed_with_port_preservation",
    "get_port_constraints",
    "EnhancedEmbeddingPolicy",
    "PortPreservationPolicy",
    "EmbeddingReport",
]
