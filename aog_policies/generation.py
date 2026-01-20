"""
Generation policies for AOG.

This module contains all policy dataclasses used by the generation module.
All policies are JSON-serializable and support the "requested vs effective" pattern.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Literal
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
    hook_depth: float = 0.002  # 2mm (default, not hardcoded 1mm)
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

    # Legacy field aliases (not stored, just for constructor)
    channel_type: Optional[str] = field(default=None, repr=False)  # Maps to profile
    min_length: Optional[float] = field(default=None, repr=False)  # Alias for length (when length_mode="fixed")
    hook_strategy: Optional[str] = field(default=None, repr=False)  # Alias for constraint_strategy
    inlet_radius: Optional[float] = field(default=None, repr=False)  # Alias for radius_end (taper start)
    outlet_radius: Optional[float] = field(default=None, repr=False)  # Alias for radius_end (taper end)
    target_depth: Optional[float] = field(default=None, repr=False)  # Alias for hook_depth
    hook_angle: Optional[float] = field(default=None, repr=False)  # Alias for hook_angle_deg

    def __post_init__(self):
        import logging
        logger = logging.getLogger(__name__)
        
        # Handle legacy channel_type field
        if self.channel_type is not None and self.profile == "cylinder":
            type_map = {"straight": "cylinder", "tapered": "taper", "fang_hook": "fang_hook"}
            self.profile = type_map.get(self.channel_type, self.profile)
        
        # Handle alias: min_length -> length (with length_mode="explicit")
        # Keep the alias field readable for backward compatibility
        if self.min_length is not None:
            if self.length is None:
                self.length = self.min_length
            logger.warning("ChannelPolicy: 'min_length' is deprecated, use 'length' instead.")
        
        # Handle alias: hook_strategy -> constraint_strategy
        # Keep the alias field readable for backward compatibility
        if self.hook_strategy is not None:
            self.constraint_strategy = self.hook_strategy
            logger.warning("ChannelPolicy: 'hook_strategy' is deprecated, use 'constraint_strategy' instead.")
        
        # Handle alias: inlet_radius/outlet_radius -> radius_end
        if self.inlet_radius is not None or self.outlet_radius is not None:
            if self.radius_end is None and self.outlet_radius is not None:
                self.radius_end = self.outlet_radius
            logger.warning("ChannelPolicy: 'inlet_radius'/'outlet_radius' are deprecated, use 'radius_end' instead.")
        
        # Handle alias: target_depth -> hook_depth
        # Keep the alias field readable for backward compatibility
        if self.target_depth is not None:
            self.hook_depth = self.target_depth
            logger.warning("ChannelPolicy: 'target_depth' is deprecated, use 'hook_depth' instead.")
        
        # Handle alias: hook_angle -> hook_angle_deg
        if self.hook_angle is not None:
            self.hook_angle_deg = self.hook_angle
            logger.warning("ChannelPolicy: 'hook_angle' is deprecated, use 'hook_angle_deg' instead.")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Include alias fields in output if they have values (for backward compatibility)
        # but remove None values for cleaner output
        for alias in ["channel_type", "min_length", "inlet_radius", "outlet_radius"]:
            if d.get(alias) is None:
                d.pop(alias, None)
        # Keep hook_strategy, target_depth, hook_angle if they have values
        if d.get("hook_strategy") is None:
            d.pop("hook_strategy", None)
        if d.get("target_depth") is None:
            d.pop("target_depth", None)
        if d.get("hook_angle") is None:
            d.pop("hook_angle", None)
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
    radial_reference: Dict[str, Any] = field(
        default_factory=lambda: {"mode": "face", "face": "top", "center": "face_center"}
    )
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
        "min_radius": float (meters),
        "step_size": float (meters),
        "backend_params": dict (JSON-serializable backend-specific configuration)
    }

    backend_params Structure (for programmatic backend):
    {
        "mode": "network" | "mesh",
        "path_algorithm": "astar_voxel" | "straight" | "bezier" | "hybrid",
        "waypoint_policy": {"allow_skip": bool, ...},
        "pathfinding_policy": {"voxel_pitch": float, "clearance": float, ...},
        "radius_policy": {...},
        "steps": [{op: ..., ...}, ...]
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
    min_radius: float = 0.0001  # 0.1mm - minimum vessel radius for growth
    step_size: float = 0.0003  # 0.3mm
    backend_params: Dict[str, Any] = field(default_factory=dict)

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

    PATCH 5: Added voxel repair pitch fields for policy-driven repair.
    PATCH 7: Added resolution-aware pitch selection with budget-aware relaxation.

    Controls how vascular networks are converted to triangle meshes.

    JSON Schema:
    {
        "add_node_spheres": bool,
        "cap_ends": bool,
        "radius_clamp_min": float (meters),
        "radius_clamp_max": float (meters),
        "voxel_repair_synthesis": bool,
        "voxel_repair_pitch": float (meters) | null,
        "voxel_repair_auto_adjust": bool,
        "voxel_repair_max_steps": int,
        "voxel_repair_step_factor": float,
        "voxel_repair_max_voxels": int,
        "segments_per_circle": int,
        "mutate_network_in_place": bool,
        "radius_clamp_mode": "copy" | "mutate",
        "use_resolution_policy": bool
    }

    Resolution-aware pitch selection:
    - If voxel_repair_pitch is None and use_resolution_policy is True, pitch is derived
      from ResolutionPolicy.repair_pitch
    - If voxel_repair_max_voxels would be exceeded, pitch is automatically relaxed with warning
    - effective_pitch is recorded in the operation report
    """
    add_node_spheres: bool = True
    cap_ends: bool = True
    radius_clamp_min: Optional[float] = None
    radius_clamp_max: Optional[float] = None
    voxel_repair_synthesis: bool = False
    # PATCH 5: New voxel repair policy fields
    # PATCH 7: Made pitch optional for resolution-aware selection
    voxel_repair_pitch: Optional[float] = 1e-4  # 0.1mm default, None = use resolution policy
    voxel_repair_auto_adjust: bool = True
    voxel_repair_max_steps: int = 4
    voxel_repair_step_factor: float = 1.5
    voxel_repair_max_voxels: int = 100_000_000  # 100M voxels budget
    segments_per_circle: int = 16
    mutate_network_in_place: bool = False
    radius_clamp_mode: Literal["copy", "mutate"] = "copy"
    use_resolution_policy: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MeshSynthesisPolicy":
        return MeshSynthesisPolicy(**{k: v for k, v in d.items() if k in MeshSynthesisPolicy.__dataclass_fields__})


@dataclass
class MeshMergePolicy:
    """
    Policy for mesh merging operations.

    PATCH 7: Added resolution-aware pitch selection with budget-aware relaxation.
    PATCH 8: Added detail loss detection and minimum voxels per diameter control.

    Controls how multiple meshes are combined, with voxel-first strategy.

    JSON Schema:
    {
        "mode": "auto" | "voxel" | "boolean",
        "voxel_pitch": float (meters) | null,
        "auto_adjust_pitch": bool,
        "max_pitch_steps": int,
        "pitch_step_factor": float,
        "fallback_boolean": bool,
        "keep_largest_component": bool,
        "min_component_faces": int,
        "min_component_volume": float (cubic meters),
        "fill_voxels": bool,
        "max_voxels": int,
        "use_resolution_policy": bool,
        "min_voxels_per_diameter": int,
        "min_channel_diameter": float (meters) | null,
        "detail_loss_threshold": float,
        "detail_loss_strictness": "warn" | "fail"
    }

    Resolution-aware pitch selection:
    - If voxel_pitch is None and use_resolution_policy is True, pitch is derived
      from ResolutionPolicy.merge_pitch
    - If max_voxels would be exceeded, pitch is automatically relaxed with warning
    - effective_pitch is recorded in the operation report
    
    Detail loss detection:
    - Compares union result volume/face count to component mesh aggregates
    - If volume loss exceeds detail_loss_threshold, warns or fails based on strictness
    - min_voxels_per_diameter ensures sufficient resolution for smallest channels
    """
    mode: Literal["auto", "voxel", "boolean"] = "auto"
    voxel_pitch: Optional[float] = 5e-5  # 50um, None = use resolution policy
    auto_adjust_pitch: bool = True
    max_pitch_steps: int = 4
    pitch_step_factor: float = 1.5
    fallback_boolean: bool = True
    keep_largest_component: bool = True
    min_component_faces: int = 100
    min_component_volume: float = 1e-12  # 1 cubic mm
    fill_voxels: bool = True
    max_voxels: int = 100_000_000  # 100M voxels budget
    use_resolution_policy: bool = False
    # Detail preservation settings
    min_voxels_per_diameter: int = 4  # Minimum voxels across smallest channel diameter
    min_channel_diameter: Optional[float] = None  # Smallest expected channel diameter (meters)
    detail_loss_threshold: float = 0.5  # Warn/fail if volume loss exceeds 50%
    detail_loss_strictness: Literal["warn", "fail"] = "warn"  # How to handle detail loss

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MeshMergePolicy":
        return MeshMergePolicy(**{k: v for k, v in d.items() if k in MeshMergePolicy.__dataclass_fields__})


@dataclass
class EmbeddingPolicy:
    """
    Policy for embedding voids into domains.

    PATCH 3: Removed "mask" mode - only "recarve" is supported.
    PATCH 7: Added resolution-aware pitch selection with budget-aware relaxation.

    Controls the voxelization and carving process for creating
    domain-with-void meshes.

    JSON Schema:
    {
        "voxel_pitch": float (meters) | null,
        "shell_thickness": float (meters),
        "auto_adjust_pitch": bool,
        "max_pitch_steps": int,
        "pitch_step_factor": float,
        "max_voxels": int,
        "fallback": "auto" | "voxel_subtraction" | "none",
        "preserve_ports_enabled": bool,
        "preserve_mode": "recarve",
        "carve_radius_factor": float,
        "carve_depth": float (meters),
        "use_resolution_policy": bool,
        "output_shell": bool,
        "output_domain_with_void": bool,
        "output_void_mesh": bool
    }

    Resolution-aware pitch selection:
    - If voxel_pitch is None and use_resolution_policy is True, pitch is derived
      from ResolutionPolicy.embed_pitch
    - If max_voxels would be exceeded, pitch is automatically relaxed with warning
    - effective_pitch is recorded in the operation report
    """
    voxel_pitch: Optional[float] = 3e-4  # 0.3mm, None = use resolution policy
    shell_thickness: float = 2e-3  # 2mm
    auto_adjust_pitch: bool = True
    max_pitch_steps: int = 4
    pitch_step_factor: float = 1.5
    max_voxels: int = 100_000_000  # 100M voxels budget
    fallback: Literal["auto", "voxel_subtraction", "none"] = "auto"
    preserve_ports_enabled: bool = True
    preserve_mode: Literal["recarve"] = "recarve"  # PATCH 3: Removed "mask" mode
    carve_radius_factor: float = 1.2
    carve_depth: float = 0.002  # 2mm
    use_resolution_policy: bool = False
    output_shell: bool = False  # Runner contract: output shell mesh
    output_domain_with_void: bool = True  # Runner contract: output domain with void
    output_void_mesh: bool = True  # Runner contract: output void mesh
    
    # Alias field for backward compatibility (not stored, just for constructor)
    preserve_ports_mode: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        import logging
        # Handle alias: preserve_ports_mode -> preserve_mode
        if self.preserve_ports_mode is not None:
            self.preserve_mode = self.preserve_ports_mode
            self.preserve_ports_mode = None  # Clear the alias field
            logging.getLogger(__name__).warning(
                "EmbeddingPolicy: 'preserve_ports_mode' is deprecated, use 'preserve_mode' instead."
            )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Remove alias field from output
        d.pop("preserve_ports_mode", None)
        return d
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "EmbeddingPolicy":
        import logging
        d = dict(d)  # Make a copy to avoid mutating input
        
        # Alias: preserve_ports_mode -> preserve_mode
        if "preserve_ports_mode" in d and "preserve_mode" not in d:
            d["preserve_mode"] = d.pop("preserve_ports_mode")
            logging.getLogger(__name__).warning(
                "EmbeddingPolicy: 'preserve_ports_mode' is deprecated, use 'preserve_mode' instead."
            )
        
        # PATCH 3: Convert legacy "mask" mode to "recarve" with warning
        if d.get("preserve_mode") == "mask":
            logging.getLogger(__name__).warning(
                "EmbeddingPolicy: 'mask' preserve_mode is deprecated and has been "
                "converted to 'recarve'. The 'mask' mode was a no-op and is no longer supported."
            )
            d["preserve_mode"] = "recarve"
        # Filter out None values so dataclass defaults are used instead
        # This prevents explicit null in JSON from overriding defaults
        return EmbeddingPolicy(**{
            k: v for k, v in d.items() 
            if k in EmbeddingPolicy.__dataclass_fields__ and v is not None
        })


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
        "save_reports": bool,
        "output_stl": bool,
        "output_json": bool,
        "output_shell": bool
    }
    """
    output_dir: str = "./output"
    output_units: Literal["mm", "m"] = "mm"
    naming_convention: Literal["default", "timestamped"] = "default"
    save_intermediates: bool = False
    save_reports: bool = True
    output_stl: bool = True  # Runner contract: output STL files
    output_json: bool = True  # Runner contract: output JSON files
    output_shell: bool = False  # Runner contract: output shell mesh

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "OutputPolicy":
        return OutputPolicy(**{k: v for k, v in d.items() if k in OutputPolicy.__dataclass_fields__})


@dataclass
class ProgramPolicy:
    """
    Policy for programmatic network generation.
    
    This is the main configuration object for the ProgrammaticBackend,
    containing the DSL steps and all sub-policies.
    
    JSON Schema:
    {
        "mode": "network" | "mesh",
        "steps": [StepSpec, ...],
        "path_algorithm": "astar_voxel" | "straight" | "bezier" | "hybrid",
        "collision_policy": UnifiedCollisionPolicy,
        "retry_policy": RetryPolicy,
        "waypoint_policy": WaypointPolicy,
        "radius_policy": RadiusPolicy,
        "default_radius": float (meters),
        "default_clearance": float (meters)
    }
    
    Sub-policies use their respective policy classes from aog_policies.collision
    and aog_policies.pathfinding. They are converted to/from dicts for JSON
    serialization using their to_dict/from_dict methods.
    """
    mode: Literal["network", "mesh"] = "network"
    steps: List[Dict[str, Any]] = field(default_factory=list)
    path_algorithm: str = "astar_voxel"
    # Sub-policies use Any type to avoid circular imports; actual types are:
    # collision_policy: UnifiedCollisionPolicy
    # retry_policy: RetryPolicy  
    # waypoint_policy: WaypointPolicy
    # radius_policy: RadiusPolicy
    collision_policy: Any = None
    retry_policy: Any = None
    waypoint_policy: Any = None
    radius_policy: Any = None
    default_radius: float = 0.001  # 1mm
    default_clearance: float = 0.0002  # 0.2mm
    
    def __post_init__(self):
        # Import here to avoid circular imports
        from .collision import UnifiedCollisionPolicy, RadiusPolicy, RetryPolicy
        from .pathfinding import WaypointPolicy
        
        # Initialize sub-policies with defaults if None
        if self.collision_policy is None:
            self.collision_policy = UnifiedCollisionPolicy()
        elif isinstance(self.collision_policy, dict):
            self.collision_policy = UnifiedCollisionPolicy.from_dict(self.collision_policy)
            
        if self.retry_policy is None:
            self.retry_policy = RetryPolicy()
        elif isinstance(self.retry_policy, dict):
            self.retry_policy = RetryPolicy.from_dict(self.retry_policy)
            
        if self.waypoint_policy is None:
            self.waypoint_policy = WaypointPolicy()
        elif isinstance(self.waypoint_policy, dict):
            self.waypoint_policy = WaypointPolicy.from_dict(self.waypoint_policy)
            
        if self.radius_policy is None:
            self.radius_policy = RadiusPolicy()
        elif isinstance(self.radius_policy, dict):
            self.radius_policy = RadiusPolicy.from_dict(self.radius_policy)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "steps": self.steps,
            "path_algorithm": self.path_algorithm,
            "collision_policy": self.collision_policy.to_dict() if hasattr(self.collision_policy, 'to_dict') else self.collision_policy,
            "retry_policy": self.retry_policy.to_dict() if hasattr(self.retry_policy, 'to_dict') else self.retry_policy,
            "waypoint_policy": self.waypoint_policy.to_dict() if hasattr(self.waypoint_policy, 'to_dict') else self.waypoint_policy,
            "radius_policy": self.radius_policy.to_dict() if hasattr(self.radius_policy, 'to_dict') else self.radius_policy,
            "default_radius": self.default_radius,
            "default_clearance": self.default_clearance,
        }
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ProgramPolicy":
        return ProgramPolicy(**{k: v for k, v in d.items() if k in ProgramPolicy.__dataclass_fields__})


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
    "ProgramPolicy",
]
