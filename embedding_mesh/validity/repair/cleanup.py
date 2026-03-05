"""
Mesh cleanup operations.

This module provides cleanup functions for removing small components
and filling holes in meshes.
"""

from typing import Dict, Any, Optional, TYPE_CHECKING
import numpy as np
import logging

if TYPE_CHECKING:
    import trimesh

logger = logging.getLogger(__name__)


def remove_small_components(
    mesh: "trimesh.Trimesh",
    min_faces: int = 500,
    min_diagonal: Optional[float] = None,
    keep_largest: bool = True,
) -> "trimesh.Trimesh":
    """
    Remove small disconnected components from a mesh.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to clean
    min_faces : int
        Minimum number of faces for a component to be kept
    min_diagonal : float, optional
        Minimum bounding box diagonal for a component to be kept
    keep_largest : bool
        Always keep the largest component regardless of thresholds
        
    Returns
    -------
    trimesh.Trimesh
        Cleaned mesh with small components removed
    """
    import trimesh
    
    parts = mesh.split(only_watertight=False)
    
    if len(parts) <= 1:
        return mesh
    
    # Estimate min_diagonal if not provided
    if min_diagonal is None:
        avg_edge = np.mean([np.linalg.norm(mesh.extents) / max(len(mesh.faces), 1) ** 0.5])
        min_diagonal = 8.0 * avg_edge
    
    # Score and sort components
    scored = []
    for p in parts:
        diag = float(np.linalg.norm(p.extents))
        scored.append((len(p.faces), diag, p))
    
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    
    # Keep components above thresholds
    kept = []
    for idx, (faces, diag, p) in enumerate(scored):
        if idx == 0 and keep_largest:
            kept.append(p)
            continue
        if faces >= min_faces and diag >= min_diagonal:
            kept.append(p)
    
    if not kept:
        return scored[0][2]
    
    # Combine kept components
    out = trimesh.util.concatenate(kept)
    out.merge_vertices()
    out.remove_unreferenced_vertices()
    
    if out.volume < 0:
        out.invert()
    trimesh.repair.fix_normals(out)
    
    return out


def fill_holes(mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
    """
    Fill holes in a mesh.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to repair
        
    Returns
    -------
    trimesh.Trimesh
        Mesh with holes filled
    """
    import trimesh
    
    repaired = mesh.copy()
    trimesh.repair.fill_holes(repaired)
    
    return repaired


__all__ = ["remove_small_components", "fill_holes"]
