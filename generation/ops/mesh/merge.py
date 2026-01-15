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
        result_mesh = meshes[0].copy()
        # Apply component filtering even for single mesh
        if policy.keep_largest_component:
            result_mesh, comp_meta = _filter_components(result_mesh, policy)
        report = OperationReport(
            operation="merge_meshes",
            success=True,
            requested_policy=policy.to_dict(),
            effective_policy=policy.to_dict(),
            warnings=[],
            metadata={"meshes_merged": 1, "mode_used": "single"},
        )
        return result_mesh, report
    
    warnings = []
    metadata = {
        "meshes_merged": len(meshes),
        "requested_mode": policy.mode,
        "pitch_adjustments": [],
        "voxel_estimate": None,
        "pitch_auto_adjusted": False,
    }
    
    # Preflight voxel estimate and auto-adjust pitch if needed
    effective_pitch = policy.voxel_pitch
    if policy.mode in ("voxel", "auto") and policy.auto_adjust_pitch:
        estimated_voxels, adjusted_pitch = _preflight_voxel_estimate(meshes, policy)
        metadata["voxel_estimate"] = estimated_voxels
        
        if adjusted_pitch != policy.voxel_pitch:
            metadata["pitch_auto_adjusted"] = True
            effective_pitch = adjusted_pitch
            warnings.append(
                f"Pitch auto-adjusted from {policy.voxel_pitch:.6f} to {adjusted_pitch:.6f} "
                f"to stay within {policy.max_voxels} voxel budget (estimated {estimated_voxels})"
            )
    
    # Select merge strategy
    if policy.mode == "voxel" or policy.mode == "auto":
        merged, mode_used, pitch_adjustments = _voxel_merge(meshes, policy, effective_pitch)
        metadata["mode_used"] = mode_used
        metadata["pitch_adjustments"] = pitch_adjustments
        
        if pitch_adjustments:
            warnings.append(
                f"Voxel pitch adjusted {len(pitch_adjustments)} times: "
                f"{effective_pitch:.6f} -> {pitch_adjustments[-1]:.6f}"
            )
        
        if mode_used == "boolean_fallback":
            warnings.append("Voxel merge failed, fell back to boolean union")
    
    elif policy.mode == "boolean":
        merged = _boolean_merge(meshes)
        metadata["mode_used"] = "boolean"
    
    else:
        raise ValueError(f"Unknown merge mode: {policy.mode}")
    
    # Apply component filtering
    if policy.keep_largest_component:
        merged, comp_meta = _filter_components(merged, policy)
        metadata["component_filtering"] = comp_meta
        if comp_meta.get("components_removed", 0) > 0:
            warnings.append(
                f"Removed {comp_meta['components_removed']} small components "
                f"(kept {comp_meta['components_kept']})"
            )
    
    # Compute final statistics
    metadata["vertex_count"] = len(merged.vertices)
    metadata["face_count"] = len(merged.faces)
    metadata["is_watertight"] = merged.is_watertight
    
    # Build effective policy
    effective_policy = policy.to_dict()
    if metadata["pitch_adjustments"]:
        effective_policy["voxel_pitch"] = metadata["pitch_adjustments"][-1]
    elif metadata["pitch_auto_adjusted"]:
        effective_policy["voxel_pitch"] = effective_pitch
    
    report = OperationReport(
        operation="merge_meshes",
        success=True,
        requested_policy=policy.to_dict(),
        effective_policy=effective_policy,
        warnings=warnings,
        metadata=metadata,
    )
    
    return merged, report


def _preflight_voxel_estimate(
    meshes: List["trimesh.Trimesh"],
    policy: MeshMergePolicy,
) -> Tuple[int, float]:
    """
    Estimate voxel count and auto-adjust pitch if needed.
    
    Returns (estimated_voxels, adjusted_pitch).
    """
    import trimesh
    
    # Compute combined bounding box
    all_bounds = np.vstack([m.bounds for m in meshes])
    min_bounds = all_bounds[:, 0].min(axis=0) if len(all_bounds) > 0 else np.zeros(3)
    max_bounds = all_bounds[:, 1].max(axis=0) if len(all_bounds) > 0 else np.zeros(3)
    
    # Handle case where bounds are from individual meshes
    min_bounds = np.array([m.bounds[0] for m in meshes]).min(axis=0)
    max_bounds = np.array([m.bounds[1] for m in meshes]).max(axis=0)
    
    extents = max_bounds - min_bounds
    
    pitch = policy.voxel_pitch
    
    # Estimate voxel count
    voxels_per_dim = extents / pitch
    estimated_voxels = int(np.prod(voxels_per_dim))
    
    # Auto-adjust pitch if over budget
    if estimated_voxels > policy.max_voxels and policy.auto_adjust_pitch:
        # Calculate required pitch to stay within budget
        # V = (ex/p) * (ey/p) * (ez/p) = ex*ey*ez / p^3
        # p^3 = ex*ey*ez / V
        # p = (ex*ey*ez / V)^(1/3)
        volume = np.prod(extents)
        required_pitch = (volume / policy.max_voxels) ** (1/3)
        
        # Round up to next step factor multiple
        steps_needed = int(np.ceil(np.log(required_pitch / policy.voxel_pitch) / np.log(policy.pitch_step_factor)))
        pitch = policy.voxel_pitch * (policy.pitch_step_factor ** steps_needed)
        
        # Recalculate estimate
        voxels_per_dim = extents / pitch
        estimated_voxels = int(np.prod(voxels_per_dim))
    
    return estimated_voxels, pitch


def _filter_components(
    mesh: "trimesh.Trimesh",
    policy: MeshMergePolicy,
) -> Tuple["trimesh.Trimesh", Dict[str, Any]]:
    """
    Filter mesh components based on policy criteria.
    
    Returns (filtered_mesh, metadata).
    """
    import trimesh
    
    meta = {
        "components_total": 0,
        "components_kept": 0,
        "components_removed": 0,
        "largest_component_faces": 0,
        "largest_component_volume": 0.0,
    }
    
    # Split into connected components
    try:
        components = mesh.split(only_watertight=False)
    except Exception:
        # If split fails, return original mesh
        meta["components_total"] = 1
        meta["components_kept"] = 1
        return mesh, meta
    
    if len(components) == 0:
        return mesh, meta
    
    meta["components_total"] = len(components)
    
    if len(components) == 1:
        meta["components_kept"] = 1
        meta["largest_component_faces"] = len(components[0].faces)
        try:
            meta["largest_component_volume"] = abs(components[0].volume)
        except Exception:
            meta["largest_component_volume"] = 0.0
        return components[0], meta
    
    # Filter components based on criteria
    kept_components = []
    for comp in components:
        face_count = len(comp.faces)
        try:
            volume = abs(comp.volume)
        except Exception:
            volume = 0.0
        
        # Check minimum criteria
        if face_count >= policy.min_component_faces and volume >= policy.min_component_volume:
            kept_components.append((comp, face_count, volume))
    
    if not kept_components:
        # If all components filtered out, keep the largest by face count
        largest = max(components, key=lambda c: len(c.faces))
        meta["components_kept"] = 1
        meta["components_removed"] = len(components) - 1
        meta["largest_component_faces"] = len(largest.faces)
        try:
            meta["largest_component_volume"] = abs(largest.volume)
        except Exception:
            meta["largest_component_volume"] = 0.0
        return largest, meta
    
    # Sort by volume (largest first)
    kept_components.sort(key=lambda x: x[2], reverse=True)
    
    meta["components_kept"] = len(kept_components)
    meta["components_removed"] = len(components) - len(kept_components)
    meta["largest_component_faces"] = kept_components[0][1]
    meta["largest_component_volume"] = kept_components[0][2]
    
    # If keep_largest_component is True, only keep the largest
    if policy.keep_largest_component:
        return kept_components[0][0], meta
    
    # Otherwise, combine all kept components
    if len(kept_components) == 1:
        return kept_components[0][0], meta
    
    combined = trimesh.util.concatenate([c[0] for c in kept_components])
    return combined, meta


def _voxel_merge(
    meshes: List["trimesh.Trimesh"],
    policy: MeshMergePolicy,
    effective_pitch: Optional[float] = None,
) -> Tuple["trimesh.Trimesh", str, List[float]]:
    """
    Merge meshes using voxelization with automatic pitch adjustment.
    
    Parameters
    ----------
    meshes : List[trimesh.Trimesh]
        Meshes to merge
    policy : MeshMergePolicy
        Policy controlling merge behavior
    effective_pitch : float, optional
        Pre-computed effective pitch (from preflight estimate).
        If None, uses policy.voxel_pitch.
    
    Returns (merged_mesh, mode_used, pitch_adjustments).
    """
    import trimesh
    from validity.mesh.voxel_utils import voxel_union_meshes
    
    pitch = effective_pitch if effective_pitch is not None else policy.voxel_pitch
    pitch_adjustments = []
    
    for attempt in range(policy.max_pitch_steps):
        try:
            merged = voxel_union_meshes(
                meshes,
                pitch=pitch,
                fill=policy.fill_voxels,
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
