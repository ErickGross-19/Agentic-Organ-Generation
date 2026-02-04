"""
Mesh-level operations for vascular network meshes.

This module provides operations for synthesizing, merging, and repairing
mesh representations of vascular networks.
"""

from .synthesis import (
    synthesize_mesh,
    MeshSynthesisPolicy,
)
from .merge import (
    merge_meshes,
    MeshMergePolicy,
)
from .repair import (
    repair_mesh,
    MeshRepairPolicy,
)

__all__ = [
    "synthesize_mesh",
    "MeshSynthesisPolicy",
    "merge_meshes",
    "MeshMergePolicy",
    "repair_mesh",
    "MeshRepairPolicy",
]
