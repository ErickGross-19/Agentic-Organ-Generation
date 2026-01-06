"""
Voxelization utilities with memory error handling.

This module provides helper functions for voxelization that automatically
retry with larger voxel pitch if memory errors occur.
"""

import trimesh
from typing import Optional


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
