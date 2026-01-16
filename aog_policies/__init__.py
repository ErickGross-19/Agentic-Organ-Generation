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
)

from .validity import (
    ValidationPolicy,
    RepairPolicy,
)

from .resolution import (
    PitchLimits,
    ResolutionPolicy,
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
    # Validity policies
    "ValidationPolicy",
    "RepairPolicy",
    # Resolution policies
    "PitchLimits",
    "ResolutionPolicy",
]
