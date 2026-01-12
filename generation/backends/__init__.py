"""
Backend interfaces for vascular network generation.

This module provides a unified interface for different generation methods:
- CCO hybrid backend (Sexton-style accelerated)
- Space colonization backend (wrapper for existing ops)
"""

from .base import GenerationBackend, BackendConfig
from .cco_hybrid_backend import CCOHybridBackend, CCOConfig

# Space colonization backend import is optional - it may fail if ops API has changed
try:
    from .space_colonization_backend import SpaceColonizationBackend, SpaceColonizationConfig
except Exception:
    SpaceColonizationBackend = None
    SpaceColonizationConfig = None

__all__ = [
    "GenerationBackend",
    "BackendConfig",
    "CCOHybridBackend",
    "CCOConfig",
    "SpaceColonizationBackend",
    "SpaceColonizationConfig",
]
