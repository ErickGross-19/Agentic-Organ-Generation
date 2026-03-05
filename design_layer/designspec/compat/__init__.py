"""
Compatibility layer for DesignSpec.

This module provides backward-compatible alias mappings for legacy field names
and version-specific transformations.
"""

from .v1_aliases import (
    V1_TOP_LEVEL_ALIASES,
    V1_POLICY_ALIASES,
    V1_COMPONENT_ALIASES,
    apply_aliases,
    apply_all_aliases,
)

__all__ = [
    "V1_TOP_LEVEL_ALIASES",
    "V1_POLICY_ALIASES",
    "V1_COMPONENT_ALIASES",
    "apply_aliases",
    "apply_all_aliases",
]
