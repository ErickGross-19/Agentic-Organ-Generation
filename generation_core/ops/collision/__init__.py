"""
Unified collision detection and resolution for vascular networks.

This module provides collision detection between network segments, meshes,
and domain boundaries, with configurable resolution strategies.

Re-exports both new unified API and legacy API for backward compatibility.
"""

# New unified API
from .unified import (
    detect_collisions,
    resolve_collisions,
    UnifiedCollisionPolicy,
    CollisionResult,
    ResolutionResult,
    CollisionType,
)

# Legacy API (from collision_legacy.py)
from ..collision_legacy import (
    get_collisions,
    avoid_collisions,
    RepairReport,
    capsule_collision_check,
    check_segment_collision_swept,
    check_domain_boundary_clearance,
)

__all__ = [
    # New unified API
    "detect_collisions",
    "resolve_collisions",
    "UnifiedCollisionPolicy",
    "CollisionResult",
    "ResolutionResult",
    "CollisionType",
    # Legacy API
    "get_collisions",
    "avoid_collisions",
    "RepairReport",
    "capsule_collision_check",
    "check_segment_collision_swept",
    "check_domain_boundary_clearance",
]
