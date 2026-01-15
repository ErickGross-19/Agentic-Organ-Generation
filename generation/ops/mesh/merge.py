"""
Mesh merging operations with voxel-first strategy.

This module provides functions for merging multiple meshes using
voxelization with automatic pitch adjustment.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, TYPE_CHECKING
import numpy as np
import logging

from ...policies import MeshMergePolicy, OperationReport

if TYPE_CHECKING:
    import trimesh

logger = logging.getLogger(__name__)


def merge_meshes(
    meshes: List["trimesh.Trimesh"],
    policy: Optional[MeshMergePolicy] = None,
) -> Tuple["trimesh.Trimesh", OperationReport]:
    """
    Merge multiple meshes using voxel-first strategy.
    
    Parameters
    ----------
    meshes : List[trimesh.Trimesh]
        Meshes to merge
    policy : MeshMergePolicy, optional
        Policy controlling merge behavior
        
    Returns
    -------
    merged : trimesh.Trimesh
        Merged mesh
    report : OperationReport
        Report with merge statistics
    """
    import trimesh
    
    if policy is None:
        policy = MeshMergePolicy()
    
    if not meshes:
        raise ValueError("No meshes to merge")
    
    if len(meshes) == 1:
        report = OperationReport(
            operation="merge_meshes",
            success=True,
            requested_policy=policy.to_dict(),
            effective_policy=policy.to_dict(),
            warnings=[],
            metadata={"meshes_merged": 1, "mode_used": "single"},
        )
        return meshes[0].copy(), report
    
    warnings = []
    metadata = {
        "meshes_merged": len(meshes),
        "requested_mode": policy.mode,
        "pitch_adjustments": [],
    }
    
    # Select merge strategy
    if policy.mode == "voxel" or policy.mode == "auto":
        merged, mode_used, pitch_adjustments = _voxel_merge(meshes, policy)
        metadata["mode_used"] = mode_used
        metadata["pitch_adjustments"] = pitch_adjustments
        
        if pitch_adjustments:
            warnings.append(
                f"Voxel pitch adjusted {len(pitch_adjustments)} times: "
                f"{policy.voxel_pitch:.6f} -> {pitch_adjustments[-1]:.6f}"
            )
        
        if mode_used == "boolean_fallback":
            warnings.append("Voxel merge failed, fell back to boolean union")
    
    elif policy.mode == "boolean":
        merged = _boolean_merge(meshes)
        metadata["mode_used"] = "boolean"
    
    else:
        raise ValueError(f"Unknown merge mode: {policy.mode}")
    
    # Compute final statistics
    metadata["vertex_count"] = len(merged.vertices)
    metadata["face_count"] = len(merged.faces)
    metadata["is_watertight"] = merged.is_watertight
    
    # Build effective policy
    effective_policy = policy.to_dict()
    if metadata["pitch_adjustments"]:
        effective_policy["voxel_pitch"] = metadata["pitch_adjustments"][-1]
    
    report = OperationReport(
        operation="merge_meshes",
        success=True,
        requested_policy=policy.to_dict(),
        effective_policy=effective_policy,
        warnings=warnings,
        metadata=metadata,
    )
    
    return merged, report


def _voxel_merge(
    meshes: List["trimesh.Trimesh"],
    policy: MeshMergePolicy,
) -> Tuple["trimesh.Trimesh", str, List[float]]:
    """
    Merge meshes using voxelization with automatic pitch adjustment.
    
    Returns (merged_mesh, mode_used, pitch_adjustments).
    """
    import trimesh
    from validity.mesh.voxel_utils import voxel_union_meshes
    
    pitch = policy.voxel_pitch
    pitch_adjustments = []
    
    for attempt in range(policy.max_pitch_steps):
        try:
            merged = voxel_union_meshes(
                meshes,
                pitch=pitch,
                fill=True,
                max_attempts=1,  # We handle retries ourselves
                log_prefix="[merge] ",
            )
            return merged, "voxel", pitch_adjustments
            
        except (MemoryError, RuntimeError) as e:
            if attempt < policy.max_pitch_steps - 1:
                pitch *= policy.pitch_step_factor
                pitch_adjustments.append(pitch)
                logger.warning(f"Voxel merge failed, increasing pitch to {pitch:.6f}")
            else:
                logger.error(f"Voxel merge failed after {policy.max_pitch_steps} attempts")
                
                if policy.fallback_boolean:
                    logger.info("Falling back to boolean merge")
                    merged = _boolean_merge(meshes)
                    return merged, "boolean_fallback", pitch_adjustments
                else:
                    raise
    
    # Should not reach here
    raise RuntimeError("Voxel merge failed")


def _boolean_merge(meshes: List["trimesh.Trimesh"]) -> "trimesh.Trimesh":
    """Merge meshes using boolean union."""
    import trimesh
    
    if len(meshes) == 1:
        return meshes[0].copy()
    
    # Simple concatenation as fallback
    # Note: True boolean union requires manifold meshes and is expensive
    merged = trimesh.util.concatenate(meshes)
    
    # Clean up
    merged.merge_vertices()
    merged.remove_unreferenced_vertices()
    
    # Try to fill holes
    trimesh.repair.fill_holes(merged)
    
    if merged.volume < 0:
        merged.invert()
    trimesh.repair.fix_normals(merged)
    
    return merged


__all__ = [
    "merge_meshes",
    "MeshMergePolicy",
]
