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
        "allow_boundary_intersections_at_ports": bool,
        "surface_opening_tolerance": float (meters),
        "min_diameter_threshold": float (meters),
        "max_components": int
    }
    
    When check_open_ports is enabled, use OpenPortPolicy to configure
    the open-port validation behavior.
    
    Surface Opening Support:
    When allow_boundary_intersections_at_ports is True, the void_inside_domain
    check will allow void mesh points that are outside the domain but within
    the "port neighborhood" region (defined by port position, radius, and direction).
    This enables "true surface openings" where the void intentionally intersects
    the domain boundary at declared ports.
    """
    check_watertight: bool = True
    check_components: bool = True
    check_min_diameter: bool = True
    check_open_ports: bool = False
    check_bounds: bool = True
    check_void_inside_domain: bool = True  # Added for backward compatibility
    allow_boundary_intersections_at_ports: bool = False  # Enable surface opening semantics
    surface_opening_tolerance: float = 0.001  # 1mm tolerance for port neighborhood
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
    
    PITCH SELECTION STRATEGY:
    The validation pitch determines the resolution of the voxel grid used to check
    port connectivity. A pitch that is too coarse can cause thin connections to
    disappear, leading to false "port sealed" errors.
    
    To avoid false negatives:
    - Set min_voxels_across_radius >= 6 (ensures sufficient resolution)
    - Use adaptive_pitch=True to auto-compute pitch from port radius
    - Increase max_voxels_roi if pitch relaxation warnings appear
    - Reduce local_region_size for smaller ROI at finer pitch
    
    PREFER FINE PITCH POLICY (Task E fix):
    When validation_pitch is null and prefer_fine_pitch is True:
    - Uses resolution.target_pitch or resolution.pathfinding_pitch_fine if available
    - Falls back to adaptive pitch based on port radius
    
    ROI-FIRST REDUCTION (Task E fix):
    When roi_first_reduction is True and voxel budget would be exceeded:
    - First shrinks local_region_size before increasing pitch
    - This preserves fine resolution for small features
    
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
        "min_voxels_across_radius": int,
        "adaptive_pitch": bool,
        "warn_on_pitch_relaxation": bool,
        "require_port_type": bool,
        "prefer_fine_pitch": bool,
        "roi_first_reduction": bool,
        "min_local_region_size": float (meters),
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
    local_region_size: float = 0.004  # 4mm local region (reduced from 5mm for finer pitch)
    max_voxels_roi: int = 2_000_000  # 2M voxels max per port ROI (increased from 1M)
    auto_relax_pitch: bool = True  # Relax pitch if ROI exceeds budget
    
    min_voxels_across_radius: int = 8  # Minimum voxels across port radius (6-10 recommended)
    adaptive_pitch: bool = True  # Auto-compute pitch from port radius
    warn_on_pitch_relaxation: bool = True  # Warn when pitch is relaxed
    require_port_type: bool = False  # Warn if port_type is "unknown"
    
    prefer_fine_pitch: bool = True  # Use resolution.target_pitch when validation_pitch is null
    roi_first_reduction: bool = True  # Shrink ROI before increasing pitch
    min_local_region_size: float = 0.001  # 1mm minimum ROI size for roi_first_reduction
    
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
    
    def compute_adaptive_pitch(self, port_radius: float) -> float:
        """
        Compute the validation pitch based on port radius and min_voxels_across_radius.
        
        This ensures sufficient resolution to detect thin connections.
        
        Parameters
        ----------
        port_radius : float
            Port radius in meters
            
        Returns
        -------
        float
            Computed pitch in meters
        """
        if self.validation_pitch is not None:
            return self.validation_pitch
        
        if self.adaptive_pitch and self.min_voxels_across_radius > 0:
            return port_radius / self.min_voxels_across_radius
        
        return port_radius / 4
    
    def compute_pitch_with_resolution_policy(
        self, 
        port_radius: float, 
        resolution_policy: Optional[Any] = None,
    ) -> float:
        """
        Compute validation pitch using the "prefer fine pitch" policy.
        
        When validation_pitch is null and prefer_fine_pitch is True:
        1. Uses resolution.target_pitch or resolution.pathfinding_pitch_fine if available
        2. Falls back to adaptive pitch based on port radius
        
        Parameters
        ----------
        port_radius : float
            Port radius in meters
        resolution_policy : ResolutionPolicy, optional
            Resolution policy for pitch selection
            
        Returns
        -------
        float
            Computed pitch in meters
        """
        if self.validation_pitch is not None:
            return self.validation_pitch
        
        if self.prefer_fine_pitch and resolution_policy is not None:
            try:
                target_pitch = getattr(resolution_policy, 'target_pitch', None)
                fine_pitch = getattr(resolution_policy, 'pathfinding_pitch_fine', None)
                
                if target_pitch is not None:
                    return target_pitch
                elif fine_pitch is not None:
                    return fine_pitch
            except Exception:
                pass
        
        return self.compute_adaptive_pitch(port_radius)
    
    def compute_roi_with_budget(
        self,
        pitch: float,
        max_voxels: Optional[int] = None,
    ) -> tuple:
        """
        Compute ROI size with ROI-first reduction policy.
        
        When roi_first_reduction is True and voxel budget would be exceeded:
        - First shrinks local_region_size before increasing pitch
        - This preserves fine resolution for small features
        
        Parameters
        ----------
        pitch : float
            Validation pitch in meters
        max_voxels : int, optional
            Maximum voxels budget. Uses self.max_voxels_roi if None.
            
        Returns
        -------
        tuple
            (effective_roi_size, effective_pitch, roi_was_reduced, pitch_was_relaxed)
        """
        if max_voxels is None:
            max_voxels = self.max_voxels_roi
        
        roi_size = self.local_region_size
        current_pitch = pitch
        roi_was_reduced = False
        pitch_was_relaxed = False
        
        import math
        
        def compute_voxels(size: float, p: float) -> int:
            n = max(1, int(math.ceil(size / p)))
            return n * n * n
        
        total_voxels = compute_voxels(roi_size, current_pitch)
        
        if total_voxels <= max_voxels:
            return (roi_size, current_pitch, roi_was_reduced, pitch_was_relaxed)
        
        if self.roi_first_reduction:
            while total_voxels > max_voxels and roi_size > self.min_local_region_size:
                roi_size *= 0.8
                roi_was_reduced = True
                total_voxels = compute_voxels(roi_size, current_pitch)
            
            roi_size = max(roi_size, self.min_local_region_size)
            total_voxels = compute_voxels(roi_size, current_pitch)
        
        if total_voxels > max_voxels and self.auto_relax_pitch:
            while total_voxels > max_voxels:
                current_pitch *= 1.5
                pitch_was_relaxed = True
                total_voxels = compute_voxels(roi_size, current_pitch)
        
        return (roi_size, current_pitch, roi_was_reduced, pitch_was_relaxed)
    
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
            "min_voxels_across_radius": self.min_voxels_across_radius,
            "adaptive_pitch": self.adaptive_pitch,
            "warn_on_pitch_relaxation": self.warn_on_pitch_relaxation,
            "require_port_type": self.require_port_type,
            "prefer_fine_pitch": self.prefer_fine_pitch,
            "roi_first_reduction": self.roi_first_reduction,
            "min_local_region_size": self.min_local_region_size,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OpenPortPolicy":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


__all__ = [
    "ValidationPolicy",
    "RepairPolicy",
    "OpenPortPolicy",
]
