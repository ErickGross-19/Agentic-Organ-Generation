"""
Mesh synthesis from vascular networks.

This module provides functions for converting vascular networks to
triangle meshes with policy-controlled options.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, TYPE_CHECKING
import numpy as np
import logging

from ...policies import MeshSynthesisPolicy, OperationReport

if TYPE_CHECKING:
    import trimesh
    from ...core.network import VascularNetwork

logger = logging.getLogger(__name__)


def synthesize_mesh(
    network: "VascularNetwork",
    policy: Optional[MeshSynthesisPolicy] = None,
) -> Tuple["trimesh.Trimesh", OperationReport]:
    """
    Synthesize a triangle mesh from a vascular network.
    
    This wraps the mesh_adapter.to_trimesh function with policy controls.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to convert to mesh
    policy : MeshSynthesisPolicy, optional
        Policy controlling synthesis options
        
    Returns
    -------
    mesh : trimesh.Trimesh
        Synthesized mesh
    report : OperationReport
        Report with synthesis statistics
    """
    import trimesh
    from ...adapters.mesh_adapter import to_trimesh
    
    if policy is None:
        policy = MeshSynthesisPolicy()
    
    warnings = []
    metadata = {
        "add_node_spheres": policy.add_node_spheres,
        "cap_ends": policy.cap_ends,
        "segments_per_circle": policy.segments_per_circle,
    }
    
    # Apply radius clamping if specified
    effective_policy = MeshSynthesisPolicy(
        add_node_spheres=policy.add_node_spheres,
        cap_ends=policy.cap_ends,
        radius_clamp_min=policy.radius_clamp_min,
        radius_clamp_max=policy.radius_clamp_max,
        voxel_repair_synthesis=policy.voxel_repair_synthesis,
        segments_per_circle=policy.segments_per_circle,
    )
    
    clamped_count = 0
    if policy.radius_clamp_min is not None or policy.radius_clamp_max is not None:
        for segment in network.segments.values():
            if hasattr(segment, 'start_radius') and segment.start_radius:
                original = segment.start_radius
                if policy.radius_clamp_min and segment.start_radius < policy.radius_clamp_min:
                    segment.start_radius = policy.radius_clamp_min
                    clamped_count += 1
                if policy.radius_clamp_max and segment.start_radius > policy.radius_clamp_max:
                    segment.start_radius = policy.radius_clamp_max
                    clamped_count += 1
            
            if hasattr(segment, 'end_radius') and segment.end_radius:
                if policy.radius_clamp_min and segment.end_radius < policy.radius_clamp_min:
                    segment.end_radius = policy.radius_clamp_min
                    clamped_count += 1
                if policy.radius_clamp_max and segment.end_radius > policy.radius_clamp_max:
                    segment.end_radius = policy.radius_clamp_max
                    clamped_count += 1
        
        if clamped_count > 0:
            warnings.append(f"Clamped {clamped_count} radii to policy limits")
            metadata["radii_clamped"] = clamped_count
    
    # Synthesize mesh using adapter
    try:
        mesh = to_trimesh(
            network,
            add_node_spheres=policy.add_node_spheres,
            cap_ends=policy.cap_ends,
            segments_per_circle=policy.segments_per_circle,
        )
        
        # Apply voxel repair if requested
        if policy.voxel_repair_synthesis:
            mesh = _voxel_repair(mesh)
            metadata["voxel_repair_applied"] = True
        
        metadata["vertex_count"] = len(mesh.vertices)
        metadata["face_count"] = len(mesh.faces)
        metadata["is_watertight"] = mesh.is_watertight
        
        success = True
        
    except Exception as e:
        logger.error(f"Mesh synthesis failed: {e}")
        mesh = trimesh.Trimesh()
        success = False
        metadata["error"] = str(e)
    
    report = OperationReport(
        operation="synthesize_mesh",
        success=success,
        requested_policy=policy.to_dict(),
        effective_policy=effective_policy.to_dict(),
        warnings=warnings,
        metadata=metadata,
    )
    
    return mesh, report


def _voxel_repair(mesh: "trimesh.Trimesh", pitch: float = 1e-4) -> "trimesh.Trimesh":
    """Apply voxel-based repair to a mesh."""
    import trimesh
    
    try:
        voxels = mesh.voxelized(pitch)
        voxels = voxels.fill()
        repaired = voxels.marching_cubes
        
        # Check for coordinate system issues
        in_extent = float(np.max(mesh.extents))
        out_extent = float(np.max(repaired.extents))
        
        if in_extent > 0 and out_extent / in_extent > 50:
            repaired.apply_transform(voxels.transform)
        
        repaired.merge_vertices()
        repaired.remove_unreferenced_vertices()
        
        if repaired.volume < 0:
            repaired.invert()
        trimesh.repair.fix_normals(repaired)
        
        return repaired
        
    except Exception as e:
        logger.warning(f"Voxel repair failed: {e}, returning original mesh")
        return mesh


__all__ = [
    "synthesize_mesh",
    "MeshSynthesisPolicy",
]
