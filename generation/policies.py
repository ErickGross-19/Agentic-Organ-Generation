"""
Policy dataclasses for parameterizing generation operations.

All public functions in the generation library accept policy objects
that control their behavior. This enables JSON-serializable configuration
and clear documentation of all parameters.

Each policy includes:
- enabled: bool gate field
- Default values defined here
- JSON schema docstring
- validate_policy() helper
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Literal, Tuple
import json


def validate_policy(policy: Any, required_fields: Optional[List[str]] = None) -> List[str]:
    """
    Validate a policy object.
    
    Parameters
    ----------
    policy : Any
        Policy dataclass instance to validate
    required_fields : List[str], optional
        List of field names that must be non-None
        
    Returns
    -------
    List[str]
        List of validation error messages (empty if valid)
    """
    errors = []
    
    if required_fields:
        for field_name in required_fields:
            if not hasattr(policy, field_name):
                errors.append(f"Missing required field: {field_name}")
            elif getattr(policy, field_name) is None:
                errors.append(f"Required field is None: {field_name}")
    
    return errors


@dataclass
class PortPlacementPolicy:
    """
    Policy for port placement on domain surfaces.
    
    Controls how inlet/outlet ports are positioned on domain faces,
    accounting for ridge geometry and clearance requirements.
    
    JSON Schema:
    {
        "enabled": bool,
        "ridge_width": float (meters),
        "ridge_clearance": float (meters),
        "port_margin": float (meters),
        "placement_pattern": "circle" | "grid" | "center_rings",
        "placement_fraction": float (0-1),
        "angular_offset": float (radians)
    }
    """
    enabled: bool = True
    ridge_width: float = 0.0001  # 0.1mm
    ridge_clearance: float = 0.0001  # 0.1mm
    port_margin: float = 0.0005  # 0.5mm
    placement_pattern: Literal["circle", "grid", "center_rings"] = "circle"
    placement_fraction: float = 0.7
    angular_offset: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PortPlacementPolicy":
        return PortPlacementPolicy(**{k: v for k, v in d.items() if k in PortPlacementPolicy.__dataclass_fields__})


@dataclass
class ChannelPolicy:
    """
    Policy for channel primitive creation.
    
    Controls the geometry of individual channel segments including
    tapering and curved hook shapes.
    
    JSON Schema:
    {
        "enabled": bool,
        "channel_type": "straight" | "tapered" | "fang_hook",
        "taper_factor": float (0-1),
        "hook_depth": float (meters),
        "hook_angle_deg": float (degrees),
        "segments_per_curve": int,
        "enforce_effective_radius": bool
    }
    """
    enabled: bool = True
    channel_type: Literal["straight", "tapered", "fang_hook"] = "straight"
    taper_factor: float = 0.5
    hook_depth: float = 0.001  # 1mm
    hook_angle_deg: float = 90.0
    segments_per_curve: int = 16
    enforce_effective_radius: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ChannelPolicy":
        return ChannelPolicy(**{k: v for k, v in d.items() if k in ChannelPolicy.__dataclass_fields__})


@dataclass
class GrowthPolicy:
    """
    Policy for network growth/generation.
    
    Controls the generation backend and its parameters for growing
    vascular networks.
    
    JSON Schema:
    {
        "enabled": bool,
        "backend": "space_colonization" | "kary_tree" | "cco_hybrid",
        "target_terminals": int,
        "terminal_tolerance": float (fraction),
        "max_iterations": int,
        "seed": int | null
    }
    """
    enabled: bool = True
    backend: Literal["space_colonization", "kary_tree", "cco_hybrid"] = "space_colonization"
    target_terminals: Optional[int] = None
    terminal_tolerance: float = 0.1  # 10% tolerance
    max_iterations: int = 500
    seed: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GrowthPolicy":
        return GrowthPolicy(**{k: v for k, v in d.items() if k in GrowthPolicy.__dataclass_fields__})


@dataclass
class CollisionPolicy:
    """
    Policy for collision detection during generation.
    
    JSON Schema:
    {
        "enabled": bool,
        "check_collisions": bool,
        "collision_clearance": float (meters)
    }
    """
    enabled: bool = True
    check_collisions: bool = True
    collision_clearance: float = 0.0002  # 0.2mm
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CollisionPolicy":
        return CollisionPolicy(**{k: v for k, v in d.items() if k in CollisionPolicy.__dataclass_fields__})


@dataclass
class NetworkCleanupPolicy:
    """
    Policy for network cleanup operations.
    
    Controls node snapping, duplicate merging, and segment pruning.
    
    JSON Schema:
    {
        "enable_snap": bool,
        "snap_tol": float (meters),
        "enable_prune": bool,
        "min_segment_length": float (meters),
        "enable_merge": bool,
        "merge_tol": float (meters)
    }
    """
    enable_snap: bool = True
    snap_tol: float = 0.0001  # 0.1mm
    enable_prune: bool = True
    min_segment_length: float = 0.0001  # 0.1mm
    enable_merge: bool = True
    merge_tol: float = 0.0001  # 0.1mm
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "NetworkCleanupPolicy":
        return NetworkCleanupPolicy(**{k: v for k, v in d.items() if k in NetworkCleanupPolicy.__dataclass_fields__})


@dataclass
class MeshSynthesisPolicy:
    """
    Policy for mesh synthesis from networks.
    
    Controls how vascular networks are converted to triangle meshes.
    
    JSON Schema:
    {
        "add_node_spheres": bool,
        "cap_ends": bool,
        "radius_clamp_min": float (meters),
        "radius_clamp_max": float (meters),
        "voxel_repair_synthesis": bool,
        "segments_per_circle": int
    }
    """
    add_node_spheres: bool = True
    cap_ends: bool = True
    radius_clamp_min: Optional[float] = None
    radius_clamp_max: Optional[float] = None
    voxel_repair_synthesis: bool = False
    segments_per_circle: int = 16
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MeshSynthesisPolicy":
        return MeshSynthesisPolicy(**{k: v for k, v in d.items() if k in MeshSynthesisPolicy.__dataclass_fields__})


@dataclass
class MeshMergePolicy:
    """
    Policy for mesh merging operations.
    
    Controls how multiple meshes are combined, with voxel-first strategy.
    
    JSON Schema:
    {
        "mode": "auto" | "voxel" | "boolean",
        "voxel_pitch": float (meters),
        "auto_adjust_pitch": bool,
        "max_pitch_steps": int,
        "pitch_step_factor": float,
        "fallback_boolean": bool
    }
    """
    mode: Literal["auto", "voxel", "boolean"] = "auto"
    voxel_pitch: float = 5e-5  # 50um
    auto_adjust_pitch: bool = True
    max_pitch_steps: int = 4
    pitch_step_factor: float = 1.5
    fallback_boolean: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MeshMergePolicy":
        return MeshMergePolicy(**{k: v for k, v in d.items() if k in MeshMergePolicy.__dataclass_fields__})


@dataclass
class EmbeddingPolicy:
    """
    Policy for embedding voids into domains.
    
    Controls the voxelization and carving process for creating
    domain-with-void meshes.
    
    JSON Schema:
    {
        "voxel_pitch": float (meters),
        "shell_thickness": float (meters),
        "auto_adjust_pitch": bool,
        "max_pitch_steps": int,
        "fallback": "auto" | "voxel_subtraction" | "none"
    }
    """
    voxel_pitch: float = 3e-4  # 0.3mm
    shell_thickness: float = 2e-3  # 2mm
    auto_adjust_pitch: bool = True
    max_pitch_steps: int = 4
    fallback: Literal["auto", "voxel_subtraction", "none"] = "auto"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "EmbeddingPolicy":
        return EmbeddingPolicy(**{k: v for k, v in d.items() if k in EmbeddingPolicy.__dataclass_fields__})


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
    """
    check_watertight: bool = True
    check_components: bool = True
    check_min_diameter: bool = True
    check_open_ports: bool = True
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
        "remove_small_components_enabled": bool,
        "min_component_faces": int,
        "fill_holes_enabled": bool,
        "smooth_enabled": bool,
        "smooth_iterations": int
    }
    """
    voxel_repair_enabled: bool = True
    voxel_pitch: float = 1e-4  # 0.1mm
    remove_small_components_enabled: bool = True
    min_component_faces: int = 500
    fill_holes_enabled: bool = True
    smooth_enabled: bool = False
    smooth_iterations: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RepairPolicy":
        return RepairPolicy(**{k: v for k, v in d.items() if k in RepairPolicy.__dataclass_fields__})


@dataclass
class OutputPolicy:
    """
    Policy for output file generation.
    
    Controls output directory, units, and naming conventions.
    
    JSON Schema:
    {
        "output_dir": str,
        "output_units": "mm" | "m",
        "naming_convention": "default" | "timestamped",
        "save_intermediates": bool,
        "save_reports": bool
    }
    """
    output_dir: str = "./output"
    output_units: Literal["mm", "m"] = "mm"
    naming_convention: Literal["default", "timestamped"] = "default"
    save_intermediates: bool = False
    save_reports: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "OutputPolicy":
        return OutputPolicy(**{k: v for k, v in d.items() if k in OutputPolicy.__dataclass_fields__})


@dataclass
class OperationReport:
    """
    Standard report structure for all operations.
    
    Every operation returns a report with requested vs effective policy,
    warnings, and operation-specific metadata.
    """
    operation: str
    success: bool
    requested_policy: Dict[str, Any]
    effective_policy: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# Export all policies
__all__ = [
    "validate_policy",
    "PortPlacementPolicy",
    "ChannelPolicy",
    "GrowthPolicy",
    "CollisionPolicy",
    "NetworkCleanupPolicy",
    "MeshSynthesisPolicy",
    "MeshMergePolicy",
    "EmbeddingPolicy",
    "ValidationPolicy",
    "RepairPolicy",
    "OutputPolicy",
    "OperationReport",
]
