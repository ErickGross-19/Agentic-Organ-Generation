"""
Unified Collision System for all growth backends.

This module provides a single, shared collision checking and spatial indexing
layer that ALL growth backends use. Instead of each backend reimplementing
collision detection, spatial hashing, and boundary checking, they all
instantiate and use this unified system.

The key insight: in the growth engine, all that changes between tree types
is the growth algorithm. Domain collision checking, spatial hashing, etc.
all stay the same across backends.

Architecture
------------
CollisionSystem : The main entry point. Wraps spatial indexing + collision
    detection + boundary checking into one coherent object.

    - create_collision_system() : Factory function to create a system
    - CollisionSystem.check_segment_collision() : Online collision check
    - CollisionSystem.check_point_clearance() : Point clearance check
    - CollisionSystem.check_polyline_collision() : Polyline collision check
    - CollisionSystem.check_insertion_collision() : CCO-style insertion check
    - CollisionSystem.check_boundary_clearance() : Domain boundary check
    - CollisionSystem.insert_segment() : Register a new segment
    - CollisionSystem.detect_all_collisions() : Post-pass full network check
    - CollisionSystem.resolve_collisions() : Attempt to resolve collisions

Re-exports from spatial indexing and collision detection:
    - DynamicSpatialIndex : Incremental spatial index
    - SpatialIndex : Static spatial index for existing networks
    - SpatialHash : Fixed-grid spatial hash for point queries
    - detect_collisions : Post-pass collision detection
    - resolve_collisions : Collision resolution
    - CollisionResult, ResolutionResult, CollisionType : Result types
"""

from .system import (
    CollisionSystem,
    create_collision_system,
)

# Re-export spatial indexing primitives
from ..spatial.grid_index import (
    DynamicSpatialIndex,
    SpatialIndex,
    segment_segment_distance_exact,
    polyline_segment_distance,
)

# Re-export the spatial hash
from ..ops._spatial_hash import SpatialHash

# Re-export post-pass collision detection and resolution
from ..ops.collision.unified import (
    detect_collisions,
    resolve_collisions,
    CollisionResult,
    ResolutionResult,
    CollisionType,
    Collision,
    ResolutionStrategy,
)

__all__ = [
    # Unified system
    "CollisionSystem",
    "create_collision_system",
    # Spatial indexing
    "DynamicSpatialIndex",
    "SpatialIndex",
    "SpatialHash",
    "segment_segment_distance_exact",
    "polyline_segment_distance",
    # Post-pass detection/resolution
    "detect_collisions",
    "resolve_collisions",
    "CollisionResult",
    "ResolutionResult",
    "CollisionType",
    "Collision",
    "ResolutionStrategy",
]
