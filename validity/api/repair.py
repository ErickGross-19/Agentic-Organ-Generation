"""
Public API for mesh repair operations.

This module provides the main entry points for repairing vascular
network meshes with policy-controlled repair steps.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, TYPE_CHECKING
import logging
import numpy as np

if TYPE_CHECKING:
    import trimesh

logger = logging.getLogger(__name__)


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
class RepairReport:
    """Report from mesh repair operation."""
    success: bool
    operations_applied: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    requested_policy: Dict[str, Any] = field(default_factory=dict)
    effective_policy: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def repair_mesh(
    mesh: "trimesh.Trimesh",
    policy: Optional[RepairPolicy] = None,
) -> tuple:
    """
    Repair a mesh using the specified policy.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to repair
    policy : RepairPolicy, optional
        Policy controlling repair operations
        
    Returns
    -------
    repaired : trimesh.Trimesh
        Repaired mesh
    report : RepairReport
        Report with repair statistics
    """
    import trimesh
    
    if policy is None:
        policy = RepairPolicy()
    
    operations_applied = []
    warnings = []
    errors = []
    metadata = {
        "original_vertex_count": len(mesh.vertices),
        "original_face_count": len(mesh.faces),
        "original_watertight": mesh.is_watertight,
    }
    
    repaired = mesh.copy()
    
    # Apply voxel repair
    if policy.voxel_repair_enabled:
        try:
            repaired, voxel_meta = _voxel_repair(repaired, policy.voxel_pitch)
            operations_applied.append("voxel_repair")
            metadata["voxel_repair"] = voxel_meta
        except Exception as e:
            warnings.append(f"Voxel repair failed: {e}")
            logger.warning(f"Voxel repair failed: {e}")
    
    # Remove small components
    if policy.remove_small_components_enabled:
        try:
            repaired, component_meta = _remove_small_components(
                repaired, policy.min_component_faces
            )
            operations_applied.append("remove_small_components")
            metadata["remove_small_components"] = component_meta
        except Exception as e:
            warnings.append(f"Remove small components failed: {e}")
            logger.warning(f"Remove small components failed: {e}")
    
    # Fill holes
    if policy.fill_holes_enabled:
        try:
            trimesh.repair.fill_holes(repaired)
            operations_applied.append("fill_holes")
        except Exception as e:
            warnings.append(f"Fill holes failed: {e}")
            logger.warning(f"Fill holes failed: {e}")
    
    # Smooth
    if policy.smooth_enabled:
        try:
            repaired = _smooth_mesh(repaired, policy.smooth_iterations)
            operations_applied.append("smooth")
        except Exception as e:
            warnings.append(f"Smooth failed: {e}")
            logger.warning(f"Smooth failed: {e}")
    
    # Final cleanup
    repaired.merge_vertices()
    repaired.remove_unreferenced_vertices()
    
    if repaired.volume < 0:
        repaired.invert()
    trimesh.repair.fix_normals(repaired)
    
    # Final statistics
    metadata["final_vertex_count"] = len(repaired.vertices)
    metadata["final_face_count"] = len(repaired.faces)
    metadata["final_watertight"] = repaired.is_watertight
    
    report = RepairReport(
        success=True,
        operations_applied=operations_applied,
        warnings=warnings,
        errors=errors,
        requested_policy=policy.to_dict(),
        effective_policy=policy.to_dict(),
        metadata=metadata,
    )
    
    return repaired, report


def _voxel_repair(mesh: "trimesh.Trimesh", pitch: float) -> tuple:
    """Apply voxel-based repair to a mesh."""
    import trimesh
    
    voxels = mesh.voxelized(pitch)
    voxels = voxels.fill()
    repaired = voxels.marching_cubes
    
    # Check for coordinate system issues
    in_extent = float(np.max(mesh.extents))
    out_extent = float(np.max(repaired.extents))
    
    transform_applied = False
    if in_extent > 0 and out_extent / in_extent > 50:
        repaired.apply_transform(voxels.transform)
        transform_applied = True
    
    repaired.merge_vertices()
    repaired.remove_unreferenced_vertices()
    
    if repaired.volume < 0:
        repaired.invert()
    trimesh.repair.fix_normals(repaired)
    
    meta = {
        "pitch": pitch,
        "transform_applied": transform_applied,
    }
    
    return repaired, meta


def _remove_small_components(mesh: "trimesh.Trimesh", min_faces: int) -> tuple:
    """Remove small disconnected components from a mesh."""
    parts = mesh.split(only_watertight=False)
    
    if len(parts) <= 1:
        return mesh, {"components_removed": 0}
    
    # Keep largest component and any above threshold
    scored = [(len(p.faces), p) for p in parts]
    scored.sort(key=lambda t: t[0], reverse=True)
    
    kept = []
    removed = 0
    
    for idx, (faces, p) in enumerate(scored):
        if idx == 0 or faces >= min_faces:
            kept.append(p)
        else:
            removed += 1
    
    if not kept:
        return scored[0][1], {"components_removed": 0}
    
    import trimesh
    out = trimesh.util.concatenate(kept)
    out.merge_vertices()
    out.remove_unreferenced_vertices()
    
    if out.volume < 0:
        out.invert()
    trimesh.repair.fix_normals(out)
    
    meta = {
        "components_removed": removed,
        "components_kept": len(kept),
    }
    
    return out, meta


def _smooth_mesh(mesh: "trimesh.Trimesh", iterations: int) -> "trimesh.Trimesh":
    """Apply Laplacian smoothing to a mesh."""
    import trimesh
    
    # Simple Laplacian smoothing
    smoothed = mesh.copy()
    
    for _ in range(iterations):
        # Get vertex neighbors
        adjacency = smoothed.vertex_neighbors
        new_vertices = smoothed.vertices.copy()
        
        for i, neighbors in enumerate(adjacency):
            if neighbors:
                neighbor_positions = smoothed.vertices[neighbors]
                new_vertices[i] = neighbor_positions.mean(axis=0)
        
        smoothed.vertices = new_vertices
    
    return smoothed


__all__ = [
    "repair_mesh",
    "RepairPolicy",
    "RepairReport",
]
