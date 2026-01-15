"""
Voxel-based mesh repair operations.

This module provides voxel-based repair functions that use voxelization
and marching cubes to fix mesh issues.
"""

from typing import Dict, Any, Optional, TYPE_CHECKING
import numpy as np
import logging

if TYPE_CHECKING:
    import trimesh

logger = logging.getLogger(__name__)


def voxel_repair_mesh(
    mesh: "trimesh.Trimesh",
    pitch: float = 1e-4,
    fill: bool = True,
    max_attempts: int = 4,
    pitch_factor: float = 1.5,
) -> tuple:
    """
    Repair a mesh using voxelization and marching cubes.
    
    This converts the mesh to a voxel grid, fills it, and reconstructs
    using marching cubes. This can fix many mesh issues including
    non-manifold edges, holes, and self-intersections.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to repair
    pitch : float
        Voxel pitch in meters (default: 0.1mm)
    fill : bool
        Whether to fill the voxel grid (default: True)
    max_attempts : int
        Maximum retry attempts with increasing pitch
    pitch_factor : float
        Factor to multiply pitch on retry
        
    Returns
    -------
    repaired : trimesh.Trimesh
        Repaired mesh
    meta : dict
        Metadata about the repair operation
    """
    import trimesh
    
    current_pitch = pitch
    meta = {
        "requested_pitch": pitch,
        "final_pitch": pitch,
        "attempts": 0,
        "transform_applied": False,
    }
    
    for attempt in range(max_attempts):
        meta["attempts"] = attempt + 1
        
        try:
            # Voxelize
            voxels = mesh.voxelized(current_pitch)
            
            # Fill if requested
            if fill:
                voxels = voxels.fill()
            
            # Reconstruct with marching cubes
            repaired = voxels.marching_cubes
            
            # Check for coordinate system issues
            in_extent = float(np.max(mesh.extents))
            out_extent = float(np.max(repaired.extents))
            
            if in_extent > 0 and out_extent / in_extent > 50:
                repaired.apply_transform(voxels.transform)
                meta["transform_applied"] = True
            
            # Clean up
            repaired.merge_vertices()
            repaired.remove_unreferenced_vertices()
            
            if repaired.volume < 0:
                repaired.invert()
            trimesh.repair.fix_normals(repaired)
            
            meta["final_pitch"] = current_pitch
            
            return repaired, meta
            
        except MemoryError:
            logger.warning(
                f"Voxel repair memory error at pitch {current_pitch:.6f}, "
                f"increasing pitch (attempt {attempt + 1}/{max_attempts})"
            )
            current_pitch *= pitch_factor
            
        except Exception as e:
            logger.warning(
                f"Voxel repair failed at pitch {current_pitch:.6f}: {e}, "
                f"increasing pitch (attempt {attempt + 1}/{max_attempts})"
            )
            current_pitch *= pitch_factor
    
    # All attempts failed, return original
    logger.error(f"Voxel repair failed after {max_attempts} attempts")
    meta["error"] = "All attempts failed"
    
    return mesh, meta


__all__ = ["voxel_repair_mesh"]
