"""
Voxelization utilities with memory error handling.

This module provides helper functions for voxelization that automatically
retry with larger voxel pitch if memory errors occur, as well as robust
voxel-based mesh union operations.
"""

import numpy as np
import trimesh
from typing import Optional, List


def voxelized_with_retry(
    mesh: trimesh.Trimesh,
    pitch: float,
    method: Optional[str] = None,
    max_attempts: int = 4,
    factor: float = 1.5,
    log_prefix: str = "",
):
    """
    Voxelize a mesh with automatic retry on memory errors.
    
    If voxelization fails due to memory constraints, the pitch is increased
    by the specified factor and retried up to max_attempts times.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh to voxelize
    pitch : float
        Initial voxel pitch (size)
    method : str, optional
        Voxelization method to pass to trimesh.voxelized()
    max_attempts : int
        Maximum number of retry attempts (default: 4)
    factor : float
        Factor to multiply pitch by on each retry (default: 1.5)
    log_prefix : str
        Prefix for log messages (default: "")
        
    Returns
    -------
    trimesh.voxel.VoxelGrid
        The voxelized mesh
        
    Raises
    ------
    RuntimeError
        If voxelization fails after all retry attempts
    """
    cur = float(pitch)
    last_exc = None
    
    for attempt in range(max_attempts):
        try:
            if method is None:
                return mesh.voxelized(cur)
            else:
                return mesh.voxelized(cur, method=method)
        except MemoryError as e:
            last_exc = e
            print(
                f"{log_prefix}voxelized(pitch={cur:.4g}) raised MemoryError, "
                f"increasing pitch (attempt {attempt + 1}/{max_attempts})..."
            )
            cur *= factor
        except ValueError as e:
            last_exc = e
            print(
                f"{log_prefix}voxelized(pitch={cur:.4g}) failed ({e}), "
                f"increasing pitch (attempt {attempt + 1}/{max_attempts})..."
            )
            cur *= factor
    
    raise RuntimeError(
        f"{log_prefix}voxelization failed after {max_attempts} attempts; "
        f"final pitch={cur:.4g}, original pitch={pitch:.4g}"
    ) from last_exc


def voxel_union_meshes(
    meshes: List[trimesh.Trimesh],
    pitch: float,
    fill: bool = True,
    max_attempts: int = 4,
    retry_factor: float = 1.5,
    log_prefix: str = "",
) -> trimesh.Trimesh:
    """
    Union multiple meshes using voxelization and marching cubes.
    
    This is a robust overlap-based merge strategy: overlapping volumes are
    automatically merged during voxelization. Uses filled voxel grids to
    produce solid volumes (not surface shells).
    
    IMPORTANT: Uses trimesh's native VoxelGrid.marching_cubes property instead of
    skimage.measure.marching_cubes to avoid axis/transform seam bugs that can
    create disconnected mesh components.
    
    Parameters
    ----------
    meshes : List[trimesh.Trimesh]
        List of meshes to union
    pitch : float
        Voxel pitch (size) in the same units as the meshes
    fill : bool
        If True (default), fill voxel grids to get solid volumes.
        Without fill, voxelized() returns surface occupancy which produces
        fragile non-manifold results.
    max_attempts : int
        Maximum retry attempts for voxelization (default: 4)
    retry_factor : float
        Factor to multiply pitch by on each retry (default: 1.5)
    log_prefix : str
        Prefix for log messages (default: "")
        
    Returns
    -------
    trimesh.Trimesh
        The unioned mesh
        
    Raises
    ------
    ValueError
        If no meshes are provided
    RuntimeError
        If voxelization fails after all retry attempts
    """
    if not meshes:
        raise ValueError("No meshes to union")
    
    if len(meshes) == 1:
        return meshes[0].copy()
    
    combined = trimesh.util.concatenate(meshes)
    
    voxels = voxelized_with_retry(
        combined,
        pitch,
        max_attempts=max_attempts,
        factor=retry_factor,
        log_prefix=log_prefix,
    )
    
    if fill:
        voxels = voxels.fill()
    
    result = voxels.marching_cubes
    
    try:
        in_extent = float(np.max(combined.extents))
        out_extent = float(np.max(result.extents))
        if in_extent > 0:
            ratio = out_extent / in_extent
            if ratio > 50.0:
                print(
                    f"{log_prefix}Detected marching_cubes in voxel coordinates "
                    f"(out/in={ratio:.1f}). Applying voxels.transform to convert to world units."
                )
                result.apply_transform(voxels.transform)
    except Exception as e:
        print(f"{log_prefix}Warning: unit sanity-check/transform failed: {e}")
    
    result.merge_vertices()
    result.remove_unreferenced_vertices()
    
    trimesh.repair.fill_holes(result)
    
    if result.volume < 0:
        result.invert()
    trimesh.repair.fix_normals(result)
    
    return result


def remove_small_components(
    mesh: trimesh.Trimesh,
    min_faces: int = 500,
    min_diagonal: Optional[float] = None,
    keep_largest: bool = True,
) -> trimesh.Trimesh:
    """
    Remove tiny disconnected components ("floaters") from a mesh.
    
    Useful after voxelization + marching cubes operations which can produce
    small disconnected fragments. Keeps the largest component and any other
    component above the specified thresholds.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh to clean
    min_faces : int
        Minimum number of faces for a component to be kept (default: 500)
    min_diagonal : float, optional
        Minimum bounding box diagonal for a component to be kept.
        If None, defaults to 8× the estimated voxel pitch based on mesh resolution.
    keep_largest : bool
        If True (default), always keep the largest component regardless of thresholds
        
    Returns
    -------
    trimesh.Trimesh
        The cleaned mesh with small components removed
    """
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh
    
    if min_diagonal is None:
        avg_edge = np.mean([np.linalg.norm(mesh.extents) / max(len(mesh.faces), 1) ** 0.5 
                          for _ in [1]])
        min_diagonal = 8.0 * avg_edge
    
    scored = []
    for p in parts:
        diag = float(np.linalg.norm(p.extents))
        scored.append((len(p.faces), diag, p))
    
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    
    kept = []
    for idx, (faces, diag, p) in enumerate(scored):
        if idx == 0 and keep_largest:
            kept.append(p)
            continue
        if faces >= min_faces and diag >= min_diagonal:
            kept.append(p)
    
    if not kept:
        return scored[0][2]
    
    out = trimesh.util.concatenate(kept)
    out.merge_vertices()
    out.remove_unreferenced_vertices()
    if out.volume < 0:
        out.invert()
    trimesh.repair.fix_normals(out)
    return out
