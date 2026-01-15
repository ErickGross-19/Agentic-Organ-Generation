"""
Pathfinding operations for vascular network generation.

This module provides pathfinding algorithms for routing vascular paths
through domains while avoiding obstacles.

Re-exports both new A* API and legacy API for backward compatibility.
"""

# New A* voxel API
from .astar_voxel import (
    find_path,
    find_path_through_waypoints,
    PathfindingPolicy,
    PathfindingResult,
)

# Legacy API (from pathfinding_legacy.py)
from ..pathfinding_legacy import (
    CostWeights,
    grow_toward_targets,
)

__all__ = [
    # New A* voxel API
    "find_path",
    "find_path_through_waypoints",
    "PathfindingPolicy",
    "PathfindingResult",
    # Legacy API
    "CostWeights",
    "grow_toward_targets",
]
