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
        "check_void_inside_domain": bool,
        "min_diameter_threshold": float (meters),
        "max_components": int
    }
    
    When check_open_ports is enabled, use OpenPortPolicy to configure
    the open-port validation behavior.
    """
    check_watertight: bool = True
    check_components: bool = True
    check_min_diameter: bool = True
    check_open_ports: bool = False
    check_bounds: bool = True
    check_void_inside_domain: bool = True  # Added for backward compatibility
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


@dataclass
class OpenPortPolicy:
    """
    Policy for open-port validation.
    
    Controls how ports are checked for connectivity to the outside.
    Uses port-axis ROI voxel patches with strict budgeting for performance.
    
    JSON Schema:
    {
        "enabled": bool,
        "probe_radius_factor": float,
        "probe_length": float (meters),
        "min_connected_volume_voxels": int,
        "mode": str,
        "validation_pitch": float or null (meters),
        "local_region_size": float (meters),
        "max_voxels_roi": int,
        "auto_relax_pitch": bool,
        "roi_size_factor": float (deprecated alias for probe_radius_factor),
        "roi_min_size": float (deprecated alias for local_region_size),
        "roi_max_size": float (deprecated, ignored)
    }
    """
    enabled: bool = True
    probe_radius_factor: float = 1.2
    probe_length: float = 0.002  # 2mm probe length
    min_connected_volume_voxels: int = 10
    mode: str = "voxel_connectivity"
    validation_pitch: Optional[float] = None
    local_region_size: float = 0.005  # 5mm local region around port
    max_voxels_roi: int = 1_000_000  # 1M voxels max per port ROI
    auto_relax_pitch: bool = True  # Relax pitch if ROI exceeds budget
    
    # Alias fields for backward compatibility (not stored, just for constructor)
    roi_size_factor: Optional[float] = field(default=None, repr=False)
    roi_min_size: Optional[float] = field(default=None, repr=False)
    roi_max_size: Optional[float] = field(default=None, repr=False)
    
    def __post_init__(self):
        import logging
        logger = logging.getLogger(__name__)
        
        # Handle alias: roi_size_factor -> probe_radius_factor
        # Keep the alias field readable for backward compatibility
        if self.roi_size_factor is not None:
            self.probe_radius_factor = self.roi_size_factor
            logger.warning("OpenPortPolicy: 'roi_size_factor' is deprecated, use 'probe_radius_factor' instead.")
        
        # Handle alias: roi_min_size -> local_region_size
        # Keep the alias field readable for backward compatibility
        if self.roi_min_size is not None:
            self.local_region_size = self.roi_min_size
            logger.warning("OpenPortPolicy: 'roi_min_size' is deprecated, use 'local_region_size' instead.")
        
        # Handle alias: roi_max_size (deprecated, ignored but kept readable)
        if self.roi_max_size is not None:
            logger.warning("OpenPortPolicy: 'roi_max_size' is deprecated and ignored.")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "probe_radius_factor": self.probe_radius_factor,
            "probe_length": self.probe_length,
            "min_connected_volume_voxels": self.min_connected_volume_voxels,
            "mode": self.mode,
            "validation_pitch": self.validation_pitch,
            "local_region_size": self.local_region_size,
            "max_voxels_roi": self.max_voxels_roi,
            "auto_relax_pitch": self.auto_relax_pitch,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OpenPortPolicy":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


__all__ = [
    "ValidationPolicy",
    "RepairPolicy",
    "OpenPortPolicy",
]
