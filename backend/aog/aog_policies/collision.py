"""
Unified collision policies for AOG.

This module contains the comprehensive collision detection and resolution policies
used by the generation module. It consolidates collision handling into a single
policy surface that covers:
- Strategy order (reroute/shrink/terminate)
- Radius floors for shrink strategy
- Clearance requirements
- Obstacle inflation rules

All policies are JSON-serializable and support the "requested vs effective" pattern.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal


@dataclass
class UnifiedCollisionPolicy:
    """
    Unified policy for collision detection and resolution.
    
    This is the canonical collision policy that covers all collision handling
    behaviors including detection parameters and resolution strategy order.
    
    JSON Schema:
    {
        "enabled": bool,
        "min_clearance": float (meters),
        "strategy_order": ["reroute", "shrink", "terminate", "voxel_merge_fallback"],
        "min_radius": float (meters),
        "check_segment_segment": bool,
        "check_segment_mesh": bool,
        "check_segment_boundary": bool,
        "check_node_boundary": bool,
        "reroute_max_attempts": int,
        "shrink_factor": float,
        "shrink_max_iterations": int,
        "inflate_by_radius": bool
    }
    
    Strategy Order:
    - "reroute": Use pathfinding to find alternative route
    - "shrink": Reduce segment radii to eliminate overlap
    - "terminate": Mark collision as unresolvable (with warning)
    - "voxel_merge_fallback": Accept collision for voxel-based merge
    
    Obstacle Inflation Rule:
    When inflate_by_radius is True, obstacles are inflated by:
        clearance + local_radius
    This ensures proper separation accounting for vessel thickness.
    """
    enabled: bool = True
    min_clearance: float = 0.0002  # 0.2mm
    strategy_order: List[str] = field(
        default_factory=lambda: ["reroute", "shrink", "terminate"]
    )
    min_radius: float = 0.0001  # 0.1mm - floor for shrink strategy
    check_segment_segment: bool = True
    check_segment_mesh: bool = True
    check_segment_boundary: bool = True
    check_node_boundary: bool = True
    reroute_max_attempts: int = 3
    shrink_factor: float = 0.9
    shrink_max_iterations: int = 5
    inflate_by_radius: bool = True
    check_after_each_step: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "min_clearance": self.min_clearance,
            "strategy_order": self.strategy_order,
            "min_radius": self.min_radius,
            "check_segment_segment": self.check_segment_segment,
            "check_segment_mesh": self.check_segment_mesh,
            "check_segment_boundary": self.check_segment_boundary,
            "check_node_boundary": self.check_node_boundary,
            "reroute_max_attempts": self.reroute_max_attempts,
            "shrink_factor": self.shrink_factor,
            "shrink_max_iterations": self.shrink_max_iterations,
            "inflate_by_radius": self.inflate_by_radius,
            "check_after_each_step": self.check_after_each_step,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UnifiedCollisionPolicy":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RadiusPolicy:
    """
    Policy for radius handling during generation.
    
    Controls how radii are computed at bifurcations and along paths,
    including Murray's Law compliance.
    
    JSON Schema:
    {
        "mode": "constant" | "taper" | "murray",
        "murray_exponent": float,
        "taper_factor": float,
        "min_radius": float (meters),
        "max_radius": float (meters)
    }
    
    Modes:
    - "constant": Maintain constant radius along path
    - "taper": Reduce radius by taper_factor at each bifurcation
    - "murray": Apply Murray's Law (r_parent^n = r_child1^n + r_child2^n)
    """
    mode: Literal["constant", "taper", "murray"] = "murray"
    murray_exponent: float = 3.0
    taper_factor: float = 0.8
    min_radius: float = 0.0001  # 0.1mm
    max_radius: float = 0.005  # 5mm
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "murray_exponent": self.murray_exponent,
            "taper_factor": self.taper_factor,
            "min_radius": self.min_radius,
            "max_radius": self.max_radius,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RadiusPolicy":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RetryPolicy:
    """
    Policy for retrying failed operations.
    
    Controls retry behavior for pathfinding and collision resolution.
    
    JSON Schema:
    {
        "max_retries": int,
        "backoff_factor": float,
        "retry_with_larger_clearance": bool,
        "clearance_increase_factor": float
    }
    """
    max_retries: int = 3
    backoff_factor: float = 1.5
    retry_with_larger_clearance: bool = True
    clearance_increase_factor: float = 1.2
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_retries": self.max_retries,
            "backoff_factor": self.backoff_factor,
            "retry_with_larger_clearance": self.retry_with_larger_clearance,
            "clearance_increase_factor": self.clearance_increase_factor,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RetryPolicy":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


__all__ = [
    "UnifiedCollisionPolicy",
    "RadiusPolicy",
    "RetryPolicy",
]
