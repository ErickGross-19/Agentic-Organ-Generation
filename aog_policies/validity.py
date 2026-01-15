"""
Validity policies for AOG.

This module contains all policy dataclasses used by the validity module.
All policies are JSON-serializable and support the "requested vs effective" pattern.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Literal


@dataclass
class ValidationPolicy:
    """
    Policy for mesh/network validation.
    
    Controls which validation checks are enabled and their thresholds.
    
    JSON Schema:
    {
        "check_watertight": bool,
        "check_components": bool,
        "check_min_diameter": bool,
        "check_open_ports": bool,
        "check_bounds": bool,
        "min_diameter_threshold": float (meters),
        "max_components": int
    }
    
    Note: check_open_ports is currently not implemented and will be
    ignored with a warning if enabled.
    """
    check_watertight: bool = True
    check_components: bool = True
    check_min_diameter: bool = True
    check_open_ports: bool = False  # Not yet implemented
    check_bounds: bool = True
    min_diameter_threshold: float = 0.0005  # 0.5mm
    max_components: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ValidationPolicy":
        return ValidationPolicy(**{k: v for k, v in d.items() if k in ValidationPolicy.__dataclass_fields__})


@dataclass
class RepairPolicy:
    """
    Policy for mesh repair operations.
    
    Controls which repair steps are enabled and their parameters.
    
    JSON Schema:
    {
        "voxel_repair_enabled": bool,
        "voxel_pitch": float (meters),
        "auto_adjust_pitch": bool,
        "max_pitch_steps": int,
        "pitch_step_factor": float,
        "fill_voxels": bool,
        "remove_small_components_enabled": bool,
        "min_component_faces": int,
        "min_component_volume": float (cubic meters),
        "fill_holes_enabled": bool,
        "smooth_enabled": bool,
        "smooth_iterations": int
    }
    """
    voxel_repair_enabled: bool = True
    voxel_pitch: float = 1e-4  # 0.1mm
    auto_adjust_pitch: bool = True
    max_pitch_steps: int = 4
    pitch_step_factor: float = 1.5
    fill_voxels: bool = True
    remove_small_components_enabled: bool = True
    min_component_faces: int = 500
    min_component_volume: float = 1e-12  # 1 cubic mm
    fill_holes_enabled: bool = True
    smooth_enabled: bool = False
    smooth_iterations: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RepairPolicy":
        return RepairPolicy(**{k: v for k, v in d.items() if k in RepairPolicy.__dataclass_fields__})


__all__ = [
    "ValidationPolicy",
    "RepairPolicy",
]
