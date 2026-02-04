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
    WaypointPolicy,
    VoxelObstacleMap,
)

# Hierarchical pathfinding for scale-aware routing
from .hierarchical_astar import (
    find_path_hierarchical,
    HierarchicalPathfindingPolicy,
    HierarchicalPathfindingResult,
    CorridorVoxelMap,
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
    "WaypointPolicy",
    "VoxelObstacleMap",
    # Hierarchical pathfinding
    "find_path_hierarchical",
    "HierarchicalPathfindingPolicy",
    "HierarchicalPathfindingResult",
    "CorridorVoxelMap",
    # Legacy API
    "CostWeights",
    "grow_toward_targets",
]
