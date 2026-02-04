"""
AOG Policies - Centralized policy definitions for Agentic Organ Generation.

This package provides all policy dataclasses used by the generation and validity
modules. All policies are JSON-serializable and support the "requested vs effective"
pattern for tracking runtime adjustments.

Usage:
    from aog_policies import PortPlacementPolicy, ValidationPolicy, OperationReport
    from aog_policies.generation import GrowthPolicy, ChannelPolicy
    from aog_policies.validity import RepairPolicy
"""

from .base import (
    OperationReport,
    validate_policy,
    coerce_float,
    coerce_vec3,
    alias_fields,
)

from .generation import (
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
    ProgramPolicy,
)

from .validity import (
    ValidationPolicy,
    RepairPolicy,
    OpenPortPolicy,
)

from .resolution import (
    PitchLimits,
    ResolutionPolicy,
)

from .pathfinding import (
    PathfindingPolicy,
    WaypointPolicy,
    HierarchicalPathfindingPolicy,
)

from .composition import (
    ComposePolicy,
)

from .domain import (
    PrimitiveMeshingPolicy,
    MeshDomainPolicy,
    ImplicitMeshingPolicy,
    DomainMeshingPolicy,
)

from .collision import (
    UnifiedCollisionPolicy,
    RadiusPolicy,
    RetryPolicy,
)

from .features import (
    RidgePolicy,
    PortPreservationPolicy,
)

from .space_colonization import (
    SpaceColonizationPolicy,
)

__all__ = [
    # Base
    "OperationReport",
    "validate_policy",
    "coerce_float",
    "coerce_vec3",
    "alias_fields",
    # Generation policies
    "PortPlacementPolicy",
    "ChannelPolicy",
    "GrowthPolicy",
    "TissueSamplingPolicy",
    "CollisionPolicy",
    "NetworkCleanupPolicy",
    "MeshSynthesisPolicy",
    "MeshMergePolicy",
    "EmbeddingPolicy",
    "OutputPolicy",
    "ProgramPolicy",
    # Validity policies
    "ValidationPolicy",
    "RepairPolicy",
    "OpenPortPolicy",
    # Resolution policies
    "PitchLimits",
    "ResolutionPolicy",
    # Pathfinding policies
    "PathfindingPolicy",
    "WaypointPolicy",
    "HierarchicalPathfindingPolicy",
    # Composition policies
    "ComposePolicy",
    # Domain meshing policies
    "PrimitiveMeshingPolicy",
    "MeshDomainPolicy",
    "ImplicitMeshingPolicy",
    "DomainMeshingPolicy",
    # Collision policies
    "UnifiedCollisionPolicy",
    "RadiusPolicy",
    "RetryPolicy",
    # Feature policies
    "RidgePolicy",
    "PortPreservationPolicy",
    # Space colonization policy
    "SpaceColonizationPolicy",
]
