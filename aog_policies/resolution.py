"""
Resolution policy for scale-aware voxel operations.

This module provides ResolutionPolicy as the single source of truth for all
scale-dependent tolerances and pitches. It enables micron-scale work by
controlling where fine pitch applies and preventing hardcoded tolerances.

Target use case: smallest channel diameter = 20 µm, domain size = 5-50 mm
Resolution rule: maintain >= 8 voxels across min diameter
-> target voxel pitch for min-diameter regions: 20 µm / 8 = 2.5 µm pitch

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Literal, Tuple
import math


@dataclass
class PitchLimits:
    """
    Pitch bounds for voxel operations.
    
    Attributes:
        min_pitch: Minimum allowed pitch (meters). Default 1e-6 (1 µm).
        max_pitch: Maximum allowed pitch (meters). Default 1e-3 (1 mm).
    """
    min_pitch: float = 1e-6  # 1 µm
    max_pitch: float = 1e-3  # 1 mm
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PitchLimits":
        return PitchLimits(**{k: v for k, v in d.items() if k in PitchLimits.__dataclass_fields__})


@dataclass
class ResolutionPolicy:
    """
    Policy for scale-aware resolution and tolerance management.
    
    This is the single source of truth for all scale-dependent tolerances.
    It eliminates hardcoded tolerances/pitches and makes micron work feasible
    by controlling where fine pitch applies.
    
    Target use case:
        - Smallest channel diameter: 20 µm
        - Domain size: 5-50 mm
        - Resolution rule: >= 8 voxels across min diameter
        - Target pitch: 20 µm / 8 = 2.5 µm
    
    JSON Schema:
    {
        "input_units": "m" | "mm" | "um",
        "min_channel_diameter": float (in input_units),
        "voxels_across_min_diameter": int,
        "max_voxels": int,
        "pitch_limits": {"min_pitch": float, "max_pitch": float} (meters),
        "auto_relax_pitch": bool,
        "pitch_step_factor": float,
        "rel_epsilon": float,
        "embed_pitch_factor": float,
        "merge_pitch_factor": float,
        "repair_pitch_factor": float,
        "pathfinding_coarse_factor": float,
        "pathfinding_fine_factor": float
    }
    
    Derived outputs (computed properties):
        - target_pitch: min_channel_diameter / voxels_across_min_diameter
        - embed_pitch: target_pitch * embed_pitch_factor
        - merge_pitch: target_pitch * merge_pitch_factor
        - repair_pitch: target_pitch * repair_pitch_factor
        - pathfinding_pitch_coarse: target_pitch * pathfinding_coarse_factor
        - pathfinding_pitch_fine: target_pitch * pathfinding_fine_factor
    """
    
    input_units: Literal["m", "mm", "um"] = "m"
    min_channel_diameter: float = 2e-5  # 20 µm in meters
    voxels_across_min_diameter: int = 8
    max_voxels: int = 100_000_000  # 100M voxels budget per operation
    pitch_limits: PitchLimits = field(default_factory=PitchLimits)
    auto_relax_pitch: bool = True
    pitch_step_factor: float = 1.5
    rel_epsilon: float = 1e-6  # Relative epsilon for scale-aware tolerances
    
    embed_pitch_factor: float = 1.0
    merge_pitch_factor: float = 1.0
    repair_pitch_factor: float = 1.0
    pathfinding_coarse_factor: float = 20.0  # Coarse grid ~50-200 µm for 2.5 µm target
    pathfinding_fine_factor: float = 1.0
    
    max_voxels_embed: Optional[int] = None
    max_voxels_merge: Optional[int] = None
    max_voxels_repair: Optional[int] = None
    max_voxels_pathfinding_coarse: Optional[int] = None
    max_voxels_pathfinding_fine: Optional[int] = None
    
    def __post_init__(self):
        if isinstance(self.pitch_limits, dict):
            self.pitch_limits = PitchLimits.from_dict(self.pitch_limits)
    
    @property
    def min_channel_diameter_m(self) -> float:
        """Get min_channel_diameter in meters."""
        return self._to_meters(self.min_channel_diameter)
    
    @property
    def target_pitch(self) -> float:
        """
        Compute target pitch from min_channel_diameter and voxels_across_min_diameter.
        
        Returns:
            Target pitch in meters.
        """
        diameter_m = self.min_channel_diameter_m
        pitch = diameter_m / self.voxels_across_min_diameter
        return self._clamp_pitch(pitch)
    
    @property
    def embed_pitch(self) -> float:
        """Pitch for embedding operations."""
        return self._clamp_pitch(self.target_pitch * self.embed_pitch_factor)
    
    @property
    def merge_pitch(self) -> float:
        """Pitch for mesh merge operations."""
        return self._clamp_pitch(self.target_pitch * self.merge_pitch_factor)
    
    @property
    def repair_pitch(self) -> float:
        """Pitch for mesh repair operations."""
        return self._clamp_pitch(self.target_pitch * self.repair_pitch_factor)
    
    @property
    def pathfinding_pitch_coarse(self) -> float:
        """Coarse pitch for hierarchical pathfinding (global search)."""
        return self._clamp_pitch(self.target_pitch * self.pathfinding_coarse_factor)
    
    @property
    def pathfinding_pitch_fine(self) -> float:
        """Fine pitch for hierarchical pathfinding (corridor refinement)."""
        return self._clamp_pitch(self.target_pitch * self.pathfinding_fine_factor)
    
    def get_max_voxels_for_operation(self, operation: str) -> int:
        """
        Get max voxels budget for a specific operation.
        
        Args:
            operation: One of "embed", "merge", "repair", "pathfinding_coarse", "pathfinding_fine"
        
        Returns:
            Max voxels budget for the operation.
        """
        op_budgets = {
            "embed": self.max_voxels_embed,
            "merge": self.max_voxels_merge,
            "repair": self.max_voxels_repair,
            "pathfinding_coarse": self.max_voxels_pathfinding_coarse,
            "pathfinding_fine": self.max_voxels_pathfinding_fine,
        }
        return op_budgets.get(operation) or self.max_voxels
    
    def compute_relaxed_pitch(
        self,
        base_pitch: float,
        domain_extents: Tuple[float, float, float],
        max_voxels: Optional[int] = None,
    ) -> Tuple[float, bool, str]:
        """
        Compute a relaxed pitch if the voxel budget would be exceeded.
        
        Args:
            base_pitch: The desired pitch in meters.
            domain_extents: (width, height, depth) of the domain in meters.
            max_voxels: Override max voxels budget. Uses self.max_voxels if None.
        
        Returns:
            Tuple of (effective_pitch, was_relaxed, warning_message).
        """
        if max_voxels is None:
            max_voxels = self.max_voxels
        
        if not self.auto_relax_pitch:
            return base_pitch, False, ""
        
        pitch = base_pitch
        was_relaxed = False
        warning = ""
        
        while True:
            nx = max(1, int(math.ceil(domain_extents[0] / pitch)))
            ny = max(1, int(math.ceil(domain_extents[1] / pitch)))
            nz = max(1, int(math.ceil(domain_extents[2] / pitch)))
            total_voxels = nx * ny * nz
            
            if total_voxels <= max_voxels:
                break
            
            if pitch >= self.pitch_limits.max_pitch:
                warning = (
                    f"Pitch relaxation hit max_pitch limit ({self.pitch_limits.max_pitch:.2e} m). "
                    f"Voxel count {total_voxels:,} exceeds budget {max_voxels:,}."
                )
                break
            
            new_pitch = pitch * self.pitch_step_factor
            new_pitch = min(new_pitch, self.pitch_limits.max_pitch)
            
            if not was_relaxed:
                warning = (
                    f"Pitch relaxed from {base_pitch:.2e} m to {new_pitch:.2e} m "
                    f"to fit voxel budget {max_voxels:,}. "
                    f"Min diameter resolution may be reduced."
                )
            else:
                warning = (
                    f"Pitch relaxed from {base_pitch:.2e} m to {new_pitch:.2e} m "
                    f"to fit voxel budget {max_voxels:,}. "
                    f"Min diameter resolution may be reduced."
                )
            
            pitch = new_pitch
            was_relaxed = True
        
        return pitch, was_relaxed, warning
    
    def eps(self, domain_scale: float) -> float:
        """
        Compute scale-aware epsilon tolerance.
        
        Args:
            domain_scale: Characteristic scale of the domain (e.g., min extent) in meters.
        
        Returns:
            Epsilon tolerance in meters.
        """
        return max(1e-12, domain_scale * self.rel_epsilon)
    
    def _to_meters(self, value: float) -> float:
        """Convert value from input_units to meters."""
        if self.input_units == "m":
            return value
        elif self.input_units == "mm":
            return value * 1e-3
        elif self.input_units == "um":
            return value * 1e-6
        else:
            return value
    
    def _clamp_pitch(self, pitch: float) -> float:
        """Clamp pitch to pitch_limits."""
        return max(self.pitch_limits.min_pitch, min(pitch, self.pitch_limits.max_pitch))
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["pitch_limits"] = self.pitch_limits.to_dict()
        d["target_pitch"] = self.target_pitch
        d["embed_pitch"] = self.embed_pitch
        d["merge_pitch"] = self.merge_pitch
        d["repair_pitch"] = self.repair_pitch
        d["pathfinding_pitch_coarse"] = self.pathfinding_pitch_coarse
        d["pathfinding_pitch_fine"] = self.pathfinding_pitch_fine
        return d
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ResolutionPolicy":
        d = dict(d)
        for key in ["target_pitch", "embed_pitch", "merge_pitch", "repair_pitch", 
                    "pathfinding_pitch_coarse", "pathfinding_pitch_fine"]:
            d.pop(key, None)
        
        if "pitch_limits" in d and isinstance(d["pitch_limits"], dict):
            d["pitch_limits"] = PitchLimits.from_dict(d["pitch_limits"])
        
        return ResolutionPolicy(**{k: v for k, v in d.items() if k in ResolutionPolicy.__dataclass_fields__})


__all__ = [
    "PitchLimits",
    "ResolutionPolicy",
]
