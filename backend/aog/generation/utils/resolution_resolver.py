"""
Shared resolution resolver for pitch and tolerance computation.

This module provides a single deterministic function that computes effective pitch
from requested pitch + bbox + budgets + min feature constraints. It is the canonical
way to resolve pitch for all voxel-based operations.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import math
import logging

from aog_policies.resolution import ResolutionPolicy
from aog_policies.base import OperationReport

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class ResolutionResult:
    """
    Result of pitch resolution.
    
    Contains the effective pitch, any warnings generated, and metrics
    for reporting in OperationReport.
    
    Supports tuple unpacking for runner contract compatibility:
        effective_pitch, warnings, metrics = resolve_pitch(...)
    """
    effective_pitch: float
    was_relaxed: bool = False
    relax_factor: float = 1.0
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def __iter__(self):
        """Support tuple unpacking: effective_pitch, warnings, metrics = result"""
        return iter((self.effective_pitch, self.warnings, self.metrics))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "effective_pitch": self.effective_pitch,
            "was_relaxed": self.was_relaxed,
            "relax_factor": self.relax_factor,
            "warnings": self.warnings,
            "metrics": self.metrics,
        }


@dataclass
class ToleranceResult:
    """
    Result of tolerance derivation.
    
    Contains scale-aware tolerances derived from pitch and policy.
    """
    eps: float
    projection_tolerance: float
    repair_tolerance: float
    merge_tolerance: float
    collision_tolerance: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "eps": self.eps,
            "projection_tolerance": self.projection_tolerance,
            "repair_tolerance": self.repair_tolerance,
            "merge_tolerance": self.merge_tolerance,
            "collision_tolerance": self.collision_tolerance,
        }


def resolve_pitch(
    op_name: str,
    requested_pitch: Optional[float],
    bbox: Optional[Tuple[float, float, float, float, float, float]],
    resolution_policy: Optional[ResolutionPolicy] = None,
    max_voxels_override: Optional[int] = None,
) -> ResolutionResult:
    """
    Resolve effective pitch for a voxel-based operation.
    
    This is the canonical function for computing effective pitch from:
    - Requested pitch (explicit or None for policy-derived)
    - Bounding box (for voxel budget computation)
    - Resolution policy (for min diameter constraint and budget)
    - Optional max voxels override
    
    The function enforces:
    1. Min diameter voxel constraint (>= 8 voxels across min diameter)
    2. Max voxel budget relaxation (deterministic pitch stepping)
    3. Pitch limits (min/max bounds)
    
    Parameters
    ----------
    op_name : str
        Name of the operation (e.g., "embed", "merge", "pathfinding_coarse").
        Used for operation-specific budget lookup and warning messages.
    requested_pitch : float, optional
        Explicitly requested pitch in meters. If None, uses policy-derived pitch.
    bbox : tuple, optional
        Bounding box as (min_x, max_x, min_y, max_y, min_z, max_z) in meters.
        Required for voxel budget computation.
    resolution_policy : ResolutionPolicy, optional
        Resolution policy. Uses default if None.
    max_voxels_override : int, optional
        Override max voxels budget. Uses policy budget if None.
        
    Returns
    -------
    ResolutionResult
        Contains effective_pitch, was_relaxed, relax_factor, warnings, and metrics.
    """
    if resolution_policy is None:
        resolution_policy = ResolutionPolicy()
    
    warnings = []
    metrics: Dict[str, Any] = {
        "op_name": op_name,
        "requested_pitch": requested_pitch,
    }
    
    # Determine base pitch from policy or request
    if requested_pitch is not None:
        base_pitch = requested_pitch
        metrics["pitch_source"] = "requested"
    else:
        # Use operation-specific pitch from policy
        op_pitch_map = {
            "embed": resolution_policy.embed_pitch,
            "merge": resolution_policy.merge_pitch,
            "repair": resolution_policy.repair_pitch,
            "pathfinding_coarse": resolution_policy.pathfinding_pitch_coarse,
            "pathfinding_fine": resolution_policy.pathfinding_pitch_fine,
            "open_port_validation": resolution_policy.target_pitch,
        }
        base_pitch = op_pitch_map.get(op_name, resolution_policy.target_pitch)
        metrics["pitch_source"] = "policy"
    
    # Clamp to pitch limits
    base_pitch = max(
        resolution_policy.pitch_limits.min_pitch,
        min(base_pitch, resolution_policy.pitch_limits.max_pitch)
    )
    metrics["base_pitch"] = base_pitch
    
    # Determine max voxels budget
    if max_voxels_override is not None:
        max_voxels = max_voxels_override
        metrics["max_voxels_source"] = "override"
    else:
        max_voxels = resolution_policy.get_max_voxels_for_operation(op_name)
        metrics["max_voxels_source"] = "policy"
    metrics["max_voxels"] = max_voxels
    
    # If no bbox, return base pitch without budget check
    if bbox is None:
        return ResolutionResult(
            effective_pitch=base_pitch,
            was_relaxed=False,
            relax_factor=1.0,
            warnings=warnings,
            metrics=metrics,
        )
    
    # Compute domain extents
    extents = (
        bbox[1] - bbox[0],  # x extent
        bbox[3] - bbox[2],  # y extent
        bbox[5] - bbox[4],  # z extent
    )
    metrics["domain_extents"] = extents
    
    # Check min diameter constraint
    min_extent = min(extents)
    min_diameter_m = resolution_policy.min_channel_diameter_m
    voxels_across = resolution_policy.min_voxels_across_feature
    
    # Ensure pitch allows sufficient voxels across min diameter
    max_pitch_for_min_diameter = min_diameter_m / voxels_across
    if base_pitch > max_pitch_for_min_diameter:
        warnings.append(
            f"[{op_name}] Requested pitch {base_pitch:.2e}m exceeds max pitch "
            f"for min diameter constraint ({max_pitch_for_min_diameter:.2e}m). "
            f"Min diameter resolution may be insufficient."
        )
    
    # Compute relaxed pitch if needed (legacy call returns tuple)
    effective_pitch, was_relaxed, relax_warning = resolution_policy.compute_relaxed_pitch(
        base_pitch=base_pitch,
        domain_extents=extents,
        max_voxels=max_voxels,
    )
    if was_relaxed:
        relax_warning = (
            f"Pitch relaxed from {base_pitch:.2e} m to {effective_pitch:.2e} m "
            f"to fit voxel budget {max_voxels:,}. Min diameter resolution may be reduced."
        )
        warnings.append(f"[{op_name}] {relax_warning}")
        logger.warning(f"[{op_name}] {relax_warning}")
    elif effective_pitch > base_pitch:
        # Pitch was clamped to max_pitch (not relaxed in discrete steps)
        clamp_warning = (
            f"Pitch clamped from {base_pitch:.2e} m to {effective_pitch:.2e} m "
            f"(max_pitch limit). Voxel budget {max_voxels:,} cannot be satisfied."
        )
        warnings.append(f"[{op_name}] {clamp_warning}")
        logger.warning(f"[{op_name}] {clamp_warning}")
    
    # Compute relax factor
    relax_factor = effective_pitch / base_pitch if base_pitch > 0 else 1.0
    
    # Compute final voxel count
    nx = max(1, int(math.ceil(extents[0] / effective_pitch)))
    ny = max(1, int(math.ceil(extents[1] / effective_pitch)))
    nz = max(1, int(math.ceil(extents[2] / effective_pitch)))
    total_voxels = nx * ny * nz
    
    metrics["effective_pitch"] = effective_pitch
    metrics["was_relaxed"] = was_relaxed
    metrics["relax_factor"] = relax_factor
    metrics["voxel_count"] = total_voxels
    metrics["voxel_shape"] = (nx, ny, nz)
    
    return ResolutionResult(
        effective_pitch=effective_pitch,
        was_relaxed=was_relaxed,
        relax_factor=relax_factor,
        warnings=warnings,
        metrics=metrics,
    )


def derive_tolerances(
    pitch: float,
    resolution_policy: Optional[ResolutionPolicy] = None,
    domain_scale: Optional[float] = None,
) -> ToleranceResult:
    """
    Derive scale-aware tolerances from pitch and policy.
    
    This standardizes tolerance derivation across all operations:
    - eps: General floating-point tolerance
    - projection_tolerance: For point projection operations
    - repair_tolerance: For mesh repair operations
    - merge_tolerance: For vertex merging operations
    - collision_tolerance: For collision detection
    
    Parameters
    ----------
    pitch : float
        Voxel pitch in meters.
    resolution_policy : ResolutionPolicy, optional
        Resolution policy. Uses default if None.
    domain_scale : float, optional
        Characteristic domain scale in meters. If None, uses pitch * 1000.
        
    Returns
    -------
    ToleranceResult
        Contains all derived tolerances.
    """
    if resolution_policy is None:
        resolution_policy = ResolutionPolicy()
    
    if domain_scale is None:
        domain_scale = pitch * 1000  # Assume domain is ~1000 voxels across
    
    # Base epsilon from policy
    eps = resolution_policy.eps(domain_scale)
    
    # Projection tolerance: k * pitch where k is a small factor
    # This replaces hardcoded 0.001 (1mm) offsets
    projection_k = 0.1  # 10% of pitch
    projection_tolerance = pitch * projection_k
    
    # Repair tolerance: slightly larger than projection
    repair_k = 0.2  # 20% of pitch
    repair_tolerance = pitch * repair_k
    
    # Merge tolerance: for vertex merging, use smaller factor
    merge_k = 0.05  # 5% of pitch
    merge_tolerance = pitch * merge_k
    
    # Collision tolerance: for collision detection
    collision_k = 0.5  # 50% of pitch (half voxel)
    collision_tolerance = pitch * collision_k
    
    return ToleranceResult(
        eps=eps,
        projection_tolerance=projection_tolerance,
        repair_tolerance=repair_tolerance,
        merge_tolerance=merge_tolerance,
        collision_tolerance=collision_tolerance,
    )


def resolve_pitch_for_bbox(
    bbox: Tuple[float, float, float, float, float, float],
    resolution_policy: Optional[ResolutionPolicy] = None,
    op_name: str = "generic",
    max_voxels_override: Optional[int] = None,
) -> Tuple[float, List[str], Dict[str, Any]]:
    """
    Convenience function to resolve pitch from bbox.
    
    This is a simplified interface that returns (pitch, warnings, metrics)
    tuple for easy integration with existing code.
    
    Parameters
    ----------
    bbox : tuple
        Bounding box as (min_x, max_x, min_y, max_y, min_z, max_z) in meters.
    resolution_policy : ResolutionPolicy, optional
        Resolution policy. Uses default if None.
    op_name : str
        Operation name for budget lookup.
    max_voxels_override : int, optional
        Override max voxels budget.
        
    Returns
    -------
    tuple
        (effective_pitch, warnings, metrics)
    """
    result = resolve_pitch(
        op_name=op_name,
        requested_pitch=None,
        bbox=bbox,
        resolution_policy=resolution_policy,
        max_voxels_override=max_voxels_override,
    )
    return result.effective_pitch, result.warnings, result.metrics


def compute_voxel_count(
    bbox: Tuple[float, float, float, float, float, float],
    pitch: float,
) -> Tuple[int, Tuple[int, int, int]]:
    """
    Compute voxel count for a bounding box at a given pitch.
    
    Parameters
    ----------
    bbox : tuple
        Bounding box as (min_x, max_x, min_y, max_y, min_z, max_z) in meters.
    pitch : float
        Voxel pitch in meters.
        
    Returns
    -------
    tuple
        (total_voxels, (nx, ny, nz))
    """
    extents = (
        bbox[1] - bbox[0],
        bbox[3] - bbox[2],
        bbox[5] - bbox[4],
    )
    nx = max(1, int(math.ceil(extents[0] / pitch)))
    ny = max(1, int(math.ceil(extents[1] / pitch)))
    nz = max(1, int(math.ceil(extents[2] / pitch)))
    return nx * ny * nz, (nx, ny, nz)


def create_resolution_report(
    op_name: str,
    resolution_result: ResolutionResult,
    requested_policy: Optional[Dict[str, Any]] = None,
) -> OperationReport:
    """
    Create an OperationReport from a ResolutionResult.
    
    This standardizes how resolution results are reported in OperationReport.
    
    Parameters
    ----------
    op_name : str
        Operation name.
    resolution_result : ResolutionResult
        Result from resolve_pitch().
    requested_policy : dict, optional
        Requested policy dict for the report.
        
    Returns
    -------
    OperationReport
        Report with resolution metrics.
    """
    effective_policy = {
        "effective_pitch": resolution_result.effective_pitch,
        "was_relaxed": resolution_result.was_relaxed,
        "relax_factor": resolution_result.relax_factor,
    }
    
    return OperationReport(
        operation=op_name,
        success=True,
        requested_policy=requested_policy or {},
        effective_policy=effective_policy,
        warnings=resolution_result.warnings,
        metadata=resolution_result.metrics,
    )


__all__ = [
    "ResolutionResult",
    "ToleranceResult",
    "resolve_pitch",
    "derive_tolerances",
    "resolve_pitch_for_bbox",
    "compute_voxel_count",
    "create_resolution_report",
]
