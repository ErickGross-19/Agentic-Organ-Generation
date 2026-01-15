"""
Generation policies for AOG.

This module contains all policy dataclasses used by the generation module.
All policies are JSON-serializable and support the "requested vs effective" pattern.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Literal, Union
from .base import alias_fields


# Field aliases for backward compatibility
PORT_PLACEMENT_ALIASES = {
    "placement_pattern": "pattern",  # Legacy name -> canonical name
}


@dataclass
class PortPlacementPolicy:
    """
    Policy for port placement on domain surfaces.
    
    Controls how inlet/outlet ports are positioned on domain faces,
    accounting for ridge geometry and clearance requirements.
    
    JSON Schema:
    {
        "enabled": bool,
        "face": "top" | "bottom" | "+x" | "-x" | "+y" | "-y" | "+z" | "-z",
        "pattern": "circle" | "grid" | "center_rings" | "explicit",
        "pattern_params": dict,
        "projection_mode": "clamp_to_face" | "project_to_boundary",
        "ridge_width": float (meters),
        "ridge_clearance": float (meters),
        "port_margin": float (meters),
        "disk_constraint_enabled": bool,
        "ridge_constraint_enabled": bool,
        "placement_fraction": float (0-1),
        "angular_offset": float (radians)
    }
    
    Effective radius convention:
        ridge_inner_radius = R - ridge_width - ridge_clearance
        effective_radius = ridge_inner_radius - port_margin
    """
    enabled: bool = True
    face: Literal["top", "bottom", "+x", "-x", "+y", "-y", "+z", "-z"] = "top"
    pattern: Literal["circle", "grid", "center_rings", "explicit"] = "circle"
    pattern_params: Dict[str, Any] = field(default_factory=dict)
    projection_mode: Literal["clamp_to_face", "project_to_boundary"] = "clamp_to_face"
    ridge_width: float = 0.0001  # 0.1mm
    ridge_clearance: float = 0.0001  # 0.1mm
    port_margin: float = 0.0005  # 0.5mm
    disk_constraint_enabled: bool = True
    ridge_constraint_enabled: bool = True
    placement_fraction: float = 0.7
    angular_offset: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PortPlacementPolicy":
        # Apply aliases for backward compatibility
        d = alias_fields(d, PORT_PLACEMENT_ALIASES)
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
        "profile": "cylinder" | "taper" | "fang_hook",
        "length_mode": "explicit" | "to_center_fraction" | "to_depth",
        "length": float (meters),
        "length_fraction": float (0-1),
        "start_offset": float (meters),
        "stop_before_boundary": float (meters),
        "taper_factor": float (0-1),
        "radius_end": float (meters) | null,
        "bend_mode": "radial_out" | "arbitrary",
        "hook_depth": float (meters),
        "hook_strength": float (0-1),
        "hook_angle_deg": float (degrees),
        "straight_fraction": float (0-1),
        "curve_fraction": float (0-1),
        "bend_shape": "quadratic" | "cubic",
        "segments_per_curve": int,
        "radial_sections": int,
        "path_samples": int,
        "enforce_effective_radius": bool,
        "constraint_strategy": "reduce_depth" | "rotate" | "both"
    }
    """
    enabled: bool = True
    profile: Literal["cylinder", "taper", "fang_hook"] = "cylinder"
    length_mode: Literal["explicit", "to_center_fraction", "to_depth"] = "explicit"
    length: Optional[float] = None
    length_fraction: float = 0.5
    start_offset: float = 0.0
    stop_before_boundary: float = 0.0
    taper_factor: float = 0.5
    radius_end: Optional[float] = None
    bend_mode: Literal["radial_out", "arbitrary"] = "radial_out"
    hook_depth: float = 0.001  # 1mm
    hook_strength: float = 0.5
    hook_angle_deg: float = 90.0
    straight_fraction: float = 0.3
    curve_fraction: float = 0.4
    bend_shape: Literal["quadratic", "cubic"] = "quadratic"
    segments_per_curve: int = 16
    radial_sections: int = 16
    path_samples: int = 32
    enforce_effective_radius: bool = True
    constraint_strategy: Literal["reduce_depth", "rotate", "both"] = "reduce_depth"
    
    # Legacy field alias
    channel_type: Optional[str] = None  # Maps to profile
    
    def __post_init__(self):
        # Handle legacy channel_type field
        if self.channel_type is not None and self.profile == "cylinder":
            type_map = {"straight": "cylinder", "tapered": "taper", "fang_hook": "fang_hook"}
            self.profile = type_map.get(self.channel_type, self.profile)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Remove legacy field from output
        d.pop("channel_type", None)
        return d
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ChannelPolicy":
        return ChannelPolicy(**{k: v for k, v in d.items() if k in ChannelPolicy.__dataclass_fields__})


@dataclass
class TissueSamplingPolicy:
    """
    Policy for tissue/attractor point sampling in space colonization.
    
    Controls how tissue points are distributed within the domain for
    guiding vascular network growth.
    
    JSON Schema:
    {
        "enabled": bool,
        "n_points": int,
        "seed": int | null,
        "strategy": "uniform" | "depth_biased" | "radial_biased" | "boundary_shell" | "gaussian" | "mixture",
        "depth_reference": {"mode": "face", "face": "top|bottom|+x|..."},
        "depth_distribution": "linear" | "power" | "exponential" | "beta",
        "depth_min": float (meters),
        "depth_max": float (meters),
        "depth_power": float,
        "depth_lambda": float,
        "depth_alpha": float,
        "depth_beta": float,
        "radial_reference": {"mode": "face", "face": "...", "center": "face_center"},
        "radial_distribution": "center_heavy" | "edge_heavy" | "ring",
        "r_min": float (meters),
        "r_max": float (meters),
        "radial_power": float,
        "ring_r0": float (meters),
        "ring_sigma": float (meters),
        "shell_thickness": float (meters),
        "shell_mode": "near_boundary" | "near_center",
        "gaussian_mean": [x, y, z] (meters),
        "gaussian_sigma": [sx, sy, sz] (meters),
        "mixture_components": List[{weight, policy_subobject}],
        "min_distance_to_ports": float (meters),
        "exclude_spheres": List[{center, radius}]
    }
    """
    enabled: bool = True
    n_points: int = 1000
    seed: Optional[int] = None
    strategy: Literal["uniform", "depth_biased", "radial_biased", "boundary_shell", "gaussian", "mixture"] = "uniform"
    
    # Depth-biased parameters
    depth_reference: Dict[str, Any] = field(default_factory=lambda: {"mode": "face", "face": "top"})
    depth_distribution: Literal["linear", "power", "exponential", "beta"] = "power"
    depth_min: float = 0.0
    depth_max: Optional[float] = None  # None = full domain depth
    depth_power: float = 2.0  # For power distribution: more points deeper
    depth_lambda: float = 1.0  # For exponential distribution
    depth_alpha: float = 2.0  # For beta distribution
    depth_beta: float = 5.0  # For beta distribution
    
    # Radial-biased parameters
    radial_reference: Dict[str, Any] = field(default_factory=lambda: {"mode": "face", "face": "top", "center": "face_center"})
    radial_distribution: Literal["center_heavy", "edge_heavy", "ring"] = "center_heavy"
    r_min: float = 0.0
    r_max: Optional[float] = None  # None = domain radius
    radial_power: float = 2.0
    ring_r0: float = 0.0  # Ring center radius
    ring_sigma: float = 0.001  # Ring width (1mm)
    
    # Boundary shell parameters
    shell_thickness: float = 0.002  # 2mm
    shell_mode: Literal["near_boundary", "near_center"] = "near_boundary"
    
    # Gaussian parameters
    gaussian_mean: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    gaussian_sigma: List[float] = field(default_factory=lambda: [0.001, 0.001, 0.001])
    
    # Mixture parameters
    mixture_components: List[Dict[str, Any]] = field(default_factory=list)
    
    # Exclusions
    min_distance_to_ports: float = 0.0005  # 0.5mm
    exclude_spheres: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TissueSamplingPolicy":
        return TissueSamplingPolicy(**{k: v for k, v in d.items() if k in TissueSamplingPolicy.__dataclass_fields__})


@dataclass
class GrowthPolicy:
    """
    Policy for network growth/generation.
    
    Controls the generation backend and its parameters for growing
    vascular networks.
    
    JSON Schema:
    {
        "enabled": bool,
        "backend": "space_colonization" | "kary_tree" | "cco_hybrid" | "programmatic",
        "target_terminals": int,
        "terminal_tolerance": float (fraction),
        "max_iterations": int,
        "seed": int | null,
        "min_segment_length": float (meters),
        "max_segment_length": float (meters),
        "step_size": float (meters)
    }
    """
    enabled: bool = True
    backend: Literal["space_colonization", "kary_tree", "cco_hybrid", "programmatic"] = "space_colonization"
    target_terminals: Optional[int] = None
    terminal_tolerance: float = 0.1  # 10% tolerance
    max_iterations: int = 500
    seed: Optional[int] = None
    min_segment_length: float = 0.0002  # 0.2mm
    max_segment_length: float = 0.002  # 2mm
    step_size: float = 0.0003  # 0.3mm
    
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
        "segments_per_circle": int,
        "mutate_network_in_place": bool,
        "radius_clamp_mode": "copy" | "mutate"
    }
    """
    add_node_spheres: bool = True
    cap_ends: bool = True
    radius_clamp_min: Optional[float] = None
    radius_clamp_max: Optional[float] = None
    voxel_repair_synthesis: bool = False
    segments_per_circle: int = 16
    mutate_network_in_place: bool = False
    radius_clamp_mode: Literal["copy", "mutate"] = "copy"
    
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
        "fallback_boolean": bool,
        "keep_largest_component": bool,
        "min_component_faces": int,
        "min_component_volume": float (cubic meters),
        "fill_voxels": bool,
        "max_voxels": int
    }
    """
    mode: Literal["auto", "voxel", "boolean"] = "auto"
    voxel_pitch: float = 5e-5  # 50um
    auto_adjust_pitch: bool = True
    max_pitch_steps: int = 4
    pitch_step_factor: float = 1.5
    fallback_boolean: bool = True
    keep_largest_component: bool = True
    min_component_faces: int = 100
    min_component_volume: float = 1e-12  # 1 cubic mm
    fill_voxels: bool = True
    max_voxels: int = 100_000_000  # 100M voxels budget
    
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
        "fallback": "auto" | "voxel_subtraction" | "none",
        "preserve_ports_enabled": bool,
        "preserve_mode": "recarve" | "mask",
        "carve_radius_factor": float,
        "carve_depth": float (meters)
    }
    """
    voxel_pitch: float = 3e-4  # 0.3mm
    shell_thickness: float = 2e-3  # 2mm
    auto_adjust_pitch: bool = True
    max_pitch_steps: int = 4
    fallback: Literal["auto", "voxel_subtraction", "none"] = "auto"
    preserve_ports_enabled: bool = True
    preserve_mode: Literal["recarve", "mask"] = "recarve"
    carve_radius_factor: float = 1.2
    carve_depth: float = 0.002  # 2mm
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "EmbeddingPolicy":
        return EmbeddingPolicy(**{k: v for k, v in d.items() if k in EmbeddingPolicy.__dataclass_fields__})


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


__all__ = [
    "PortPlacementPolicy",
    "ChannelPolicy",
    "TissueSamplingPolicy",
    "GrowthPolicy",
    "CollisionPolicy",
    "NetworkCleanupPolicy",
    "MeshSynthesisPolicy",
    "MeshMergePolicy",
    "EmbeddingPolicy",
    "OutputPolicy",
]
