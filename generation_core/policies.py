"""
Policy dataclasses for parameterizing generation operations.

All public functions in the generation library accept policy objects
that control their behavior. This enables JSON-serializable configuration
and clear documentation of all parameters.

This module re-exports policies from the centralized aog_policies package
for backward compatibility. New code should import directly from aog_policies.

Each policy includes:
- enabled: bool gate field
- Default values defined here
- JSON schema docstring
- validate_policy() helper
"""

# Re-export from centralized aog_policies package
from aog_policies import (
    # Base utilities
    OperationReport,
    validate_policy,
    # Generation policies
    PortPlacementPolicy,
    ChannelPolicy,
    GrowthPolicy,
    TissueSamplingPolicy,
    CollisionPolicy,
    NetworkCleanupPolicy,
    MeshSynthesisPolicy,
    MeshMergePolicy,
    EmbeddingPolicy,
    OutputPolicy,
    # Validity policies (also available here for convenience)
    ValidationPolicy,
    RepairPolicy,
    # Pathfinding policies
    PathfindingPolicy,
    WaypointPolicy,
    HierarchicalPathfindingPolicy,
)


# Export all policies
__all__ = [
    "validate_policy",
    "PortPlacementPolicy",
    "ChannelPolicy",
    "GrowthPolicy",
    "TissueSamplingPolicy",
    "CollisionPolicy",
    "NetworkCleanupPolicy",
    "MeshSynthesisPolicy",
    "MeshMergePolicy",
    "EmbeddingPolicy",
    "ValidationPolicy",
    "RepairPolicy",
    "OutputPolicy",
    "OperationReport",
    # Pathfinding policies
    "PathfindingPolicy",
    "WaypointPolicy",
    "HierarchicalPathfindingPolicy",
]
