"""
Unified collision detection and resolution for vascular networks.

This module provides collision detection between network segments, meshes,
and domain boundaries, with configurable resolution strategies.
"""

from .unified import (
    detect_collisions,
    resolve_collisions,
    UnifiedCollisionPolicy,
    CollisionResult,
    ResolutionResult,
    CollisionType,
)

__all__ = [
    "detect_collisions",
    "resolve_collisions",
    "UnifiedCollisionPolicy",
    "CollisionResult",
    "ResolutionResult",
    "CollisionType",
]
