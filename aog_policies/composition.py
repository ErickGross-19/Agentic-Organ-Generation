"""
Composition policies for AOG.

This module contains policy dataclasses for multi-component composition.
All policies are JSON-serializable and support the "requested vs effective" pattern.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .generation import MeshSynthesisPolicy, MeshMergePolicy
from .validity import RepairPolicy


@dataclass
class ComposePolicy:
    """
    Policy for multi-component composition.
    
    Controls how multiple components are merged into a single void mesh.
    
    JSON Schema:
    {
        "synthesis_policy": MeshSynthesisPolicy,
        "merge_policy": MeshMergePolicy,
        "repair_policy": RepairPolicy,
        "repair_enabled": bool,
        "union_before_embed": bool,
        "repair_fill_holes": bool,
        "keep_largest_component": bool,
        "min_component_volume": float (cubic meters)
    }
    
    Note: repair_voxel_pitch is deprecated; use repair_policy.voxel_pitch instead.
    """
    synthesis_policy: Optional[MeshSynthesisPolicy] = None
    merge_policy: Optional[MeshMergePolicy] = None
    repair_policy: Optional[RepairPolicy] = None
    repair_enabled: bool = True
    union_before_embed: bool = True  # Runner contract: union voids before embedding
    repair_fill_holes: bool = True  # Runner contract: fill holes during repair
    repair_voxel_pitch: float = 5e-5  # 50um (kept for backward compatibility)
    keep_largest_component: bool = True
    min_component_volume: float = 1e-12  # 1 cubic mm
    
    def __post_init__(self):
        if self.synthesis_policy is None:
            self.synthesis_policy = MeshSynthesisPolicy()
        if self.merge_policy is None:
            self.merge_policy = MeshMergePolicy()
        if self.repair_policy is None:
            self.repair_policy = RepairPolicy(
                voxel_pitch=self.repair_voxel_pitch,
                min_component_volume=self.min_component_volume,
            )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "synthesis_policy": self.synthesis_policy.to_dict() if self.synthesis_policy else None,
            "merge_policy": self.merge_policy.to_dict() if self.merge_policy else None,
            "repair_policy": self.repair_policy.to_dict() if self.repair_policy else None,
            "repair_enabled": self.repair_enabled,
            "union_before_embed": self.union_before_embed,
            "repair_fill_holes": self.repair_fill_holes,
            "repair_voxel_pitch": self.repair_voxel_pitch,
            "keep_largest_component": self.keep_largest_component,
            "min_component_volume": self.min_component_volume,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ComposePolicy":
        synthesis_policy = None
        merge_policy = None
        repair_policy = None
        
        if d.get("synthesis_policy"):
            synthesis_policy = MeshSynthesisPolicy.from_dict(d["synthesis_policy"])
        if d.get("merge_policy"):
            merge_policy = MeshMergePolicy.from_dict(d["merge_policy"])
        if d.get("repair_policy"):
            repair_policy = RepairPolicy.from_dict(d["repair_policy"])
        
        return cls(
            synthesis_policy=synthesis_policy,
            merge_policy=merge_policy,
            repair_policy=repair_policy,
            repair_enabled=d.get("repair_enabled", True),
            union_before_embed=d.get("union_before_embed", True),
            repair_fill_holes=d.get("repair_fill_holes", True),
            repair_voxel_pitch=d.get("repair_voxel_pitch", 5e-5),
            keep_largest_component=d.get("keep_largest_component", True),
            min_component_volume=d.get("min_component_volume", 1e-12),
        )


__all__ = [
    "ComposePolicy",
]
