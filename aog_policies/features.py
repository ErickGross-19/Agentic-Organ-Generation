"""
Feature policies for AOG.

This module contains policy dataclasses for face features such as ridges,
grooves, and ports. Features return constraint information that can be used
by placement, pathfinding, and channel generation.

All policies are JSON-serializable and support the "requested vs effective" pattern.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RidgePolicy:
    """
    Policy for ridge features on domain faces.
    
    A ridge is a raised ring (for cylinders) or frame (for boxes) around
    the perimeter of a face. Ridge constraints inform:
    - Port placement clamp (effective radius)
    - Fang-hook enforcement (max depth)
    - Channel routing constraints (exclusion zones)
    
    JSON Schema:
    {
        "height": float (meters),
        "thickness": float (meters),
        "inset": float (meters),
        "overlap": float (meters) | null,
        "resolution": int
    }
    """
    height: float = 0.001  # 1mm
    thickness: float = 0.001  # 1mm
    inset: float = 0.0
    overlap: Optional[float] = None
    resolution: int = 64
    
    def __post_init__(self):
        if self.overlap is None:
            self.overlap = 0.5 * self.height
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "height": self.height,
            "thickness": self.thickness,
            "inset": self.inset,
            "overlap": self.overlap,
            "resolution": self.resolution,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RidgePolicy":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PortPreservationPolicy:
    """
    Policy for preserving port geometry during embedding.
    
    Controls how inlet/outlet ports are handled to ensure they remain
    accessible after the embedding process.
    
    NOTE: Only "recarve" mode is supported. The "mask" mode has been deprecated
    as it does not properly preserve port geometry.
    
    JSON Schema:
    {
        "enabled": bool,
        "mode": "recarve",
        "cylinder_radius_factor": float,
        "cylinder_depth": float (meters),
        "min_clearance": float (meters)
    }
    """
    enabled: bool = True
    mode: str = "recarve"
    cylinder_radius_factor: float = 1.2
    cylinder_depth: float = 0.002  # 2mm
    min_clearance: float = 0.0001  # 0.1mm
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "cylinder_radius_factor": self.cylinder_radius_factor,
            "cylinder_depth": self.cylinder_depth,
            "min_clearance": self.min_clearance,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PortPreservationPolicy":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


__all__ = [
    "RidgePolicy",
    "PortPreservationPolicy",
]
