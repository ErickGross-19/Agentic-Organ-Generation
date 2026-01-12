"""
Backend interfaces for vascular network generation.

This module provides a unified interface for different generation methods:
- CCO hybrid backend (Sexton-style accelerated)
- Space colonization backend (wrapper for existing ops)
"""

from .base import GenerationBackend, BackendConfig
from .cco_hybrid_backend import CCOHybridBackend, CCOConfig
from .space_colonization_backend import SpaceColonizationBackend, SpaceColonizationConfig

__all__ = [
    "GenerationBackend",
    "BackendConfig",
    "CCOHybridBackend",
    "CCOConfig",
    "SpaceColonizationBackend",
    "SpaceColonizationConfig",
]
