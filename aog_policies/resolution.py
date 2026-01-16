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
from typing import Optional, Dict, Any, Literal, Tuple, Union
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
        "min_voxels_across_feature": int,
        "max_voxels": int,
        "min_pitch": float (meters),
        "max_pitch": float (meters),
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
        - target_pitch: min_channel_diameter / min_voxels_across_feature
        - embed_pitch: target_pitch * embed_pitch_factor
        - merge_pitch: target_pitch * merge_pitch_factor
        - repair_pitch: target_pitch * repair_pitch_factor
        - pathfinding_pitch_coarse: target_pitch * pathfinding_coarse_factor
        - pathfinding_pitch_fine: target_pitch * pathfinding_fine_factor
    """

    input_units: Literal["m", "mm", "um"] = "m"
    min_channel_diameter: float = 2e-5  # 20 µm in meters
    min_voxels_across_feature: int = 8
    max_voxels: int = 100_000_000  # 100M voxels budget per operation
    min_pitch: float = 1e-6  # 1 µm - minimum allowed pitch
    max_pitch: float = 1e-3  # 1 mm - maximum allowed pitch
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
    max_voxels_open_port_roi: Optional[int] = None  # Added for backward compatibility
    
    # Alias field for backward compatibility (not stored, just for constructor)
    voxels_across_min_diameter: Optional[int] = field(default=None, repr=False)

    def __post_init__(self):
        # Handle alias: voxels_across_min_diameter -> min_voxels_across_feature
        if self.voxels_across_min_diameter is not None:
            self.min_voxels_across_feature = self.voxels_across_min_diameter
            self.voxels_across_min_diameter = None  # Clear the alias field
        
        if isinstance(self.pitch_limits, dict):
            self.pitch_limits = PitchLimits.from_dict(self.pitch_limits)
        self.pitch_limits.min_pitch = self.min_pitch
        self.pitch_limits.max_pitch = self.max_pitch

    @property
    def min_channel_diameter_m(self) -> float:
        """Get min_channel_diameter in meters."""
        return self._to_meters(self.min_channel_diameter)

    @property
    def target_pitch(self) -> float:
        """
        Compute target pitch from min_channel_diameter and min_voxels_across_feature.

        Returns:
            Target pitch in meters.
        """
        diameter_m = self.min_channel_diameter_m
        pitch = diameter_m / self.min_voxels_across_feature
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
            operation: One of "embed", "merge", "repair", "pathfinding_coarse", "pathfinding_fine", "open_port_roi"

        Returns:
            Max voxels budget for the operation.
        """
        op_budgets = {
            "embed": self.max_voxels_embed,
            "merge": self.max_voxels_merge,
            "repair": self.max_voxels_repair,
            "pathfinding_coarse": self.max_voxels_pathfinding_coarse,
            "pathfinding_fine": self.max_voxels_pathfinding_fine,
            "open_port_roi": self.max_voxels_open_port_roi,
        }
        return op_budgets.get(operation) or self.max_voxels

    def compute_relaxed_pitch(
        self,
        base_pitch: Optional[float] = None,
        domain_extents: Optional[Tuple[float, float, float]] = None,
        max_voxels: Optional[int] = None,
        *,
        bbox: Optional[Tuple[float, float, float, float, float, float]] = None,
        requested_pitch: Optional[float] = None,
        max_voxels_override: Optional[int] = None,  # Alias for max_voxels
    ) -> Union[float, Tuple[float, bool, Optional[str]]]:
        """
        Compute a relaxed pitch if the voxel budget would be exceeded.

        Supports two calling conventions with different return types:
        1. Legacy: compute_relaxed_pitch(base_pitch, domain_extents, max_voxels)
           Returns: Tuple[float, bool, Optional[str]] - (pitch, was_relaxed, warning)
        2. Runner contract: compute_relaxed_pitch(bbox=..., requested_pitch=...)
           Returns: float - just the effective pitch

        Args:
            base_pitch: (Legacy) The desired pitch in meters.
            domain_extents: (Legacy) (width, height, depth) of the domain in meters.
            max_voxels: Override max voxels budget. Uses self.max_voxels if None.
            bbox: (Runner contract) Bounding box as (x_min, x_max, y_min, y_max, z_min, z_max).
            requested_pitch: (Runner contract) The desired pitch in meters.

        Returns:
            For runner contract mode (bbox=, requested_pitch=): float - effective pitch
            For legacy mode (base_pitch, domain_extents): Tuple[float, bool, Optional[str]]
        """
        # Determine which calling convention is being used
        is_runner_contract_mode = bbox is not None and requested_pitch is not None
        
        if is_runner_contract_mode:
            domain_extents = (
                bbox[1] - bbox[0],
                bbox[3] - bbox[2],
                bbox[5] - bbox[4],
            )
            base_pitch = requested_pitch
        elif base_pitch is None or domain_extents is None:
            raise ValueError(
                "Must provide either (base_pitch, domain_extents) or (bbox=, requested_pitch=)"
            )

        # Handle alias: max_voxels_override -> max_voxels
        if max_voxels_override is not None:
            max_voxels = max_voxels_override
        if max_voxels is None:
            max_voxels = self.max_voxels

        if not self.auto_relax_pitch:
            if is_runner_contract_mode:
                return base_pitch
            return (base_pitch, False, None)

        pitch = base_pitch
        was_relaxed = False
        warning = None

        while True:
            nx = max(1, int(math.ceil(domain_extents[0] / pitch)))
            ny = max(1, int(math.ceil(domain_extents[1] / pitch)))
            nz = max(1, int(math.ceil(domain_extents[2] / pitch)))
            total_voxels = nx * ny * nz

            if total_voxels <= max_voxels:
                break

            new_pitch = pitch * self.pitch_step_factor

            if new_pitch <= pitch:
                break

            pitch = new_pitch
            was_relaxed = True

        if was_relaxed:
            warning = (
                f"Pitch relaxed from {base_pitch:.2e} to {pitch:.2e} to stay within "
                f"voxel budget ({max_voxels} voxels)"
            )

        # Return based on calling convention
        if is_runner_contract_mode:
            return pitch
        return (pitch, was_relaxed, warning)

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
        # Remove alias field from serialization
        d.pop("voxels_across_min_diameter", None)
        d["pitch_limits"] = self.pitch_limits.to_dict()
        d["target_pitch"] = self.target_pitch
        d["embed_pitch"] = self.embed_pitch
        d["merge_pitch"] = self.merge_pitch
        d["repair_pitch"] = self.repair_pitch
        d["pathfinding_pitch_coarse"] = self.pathfinding_pitch_coarse
        d["pathfinding_pitch_fine"] = self.pathfinding_pitch_fine
        # Also include the alias for backward compatibility in serialized output
        d["voxels_across_min_diameter"] = self.min_voxels_across_feature
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ResolutionPolicy":
        d = dict(d)
        
        # Handle alias: voxels_across_min_diameter -> min_voxels_across_feature
        if "voxels_across_min_diameter" in d and "min_voxels_across_feature" not in d:
            d["min_voxels_across_feature"] = d.pop("voxels_across_min_diameter")
        elif "voxels_across_min_diameter" in d:
            d.pop("voxels_across_min_diameter")  # Remove duplicate if both present
        
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
