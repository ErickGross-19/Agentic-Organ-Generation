import numpy as np
import trimesh
from scipy import ndimage
from typing import Tuple, Dict, Any

from ..mesh.voxel_utils import voxelized_with_retry


def mesh_to_fluid_mask(mesh: trimesh.Trimesh, pitch: float = 0.1):
    """
    Voxelize a watertight fluid mesh into a 3D boolean array.

    We ONLY use the VoxelGrid for occupancy; for coordinates we
    map indices to the *mesh bounding box* ourselves, to avoid
    any surprises with VoxelGrid.transform/origin.

    Returns
    -------
    fluid_mask : (nx, ny, nz) bool
        Boolean array of voxel occupancy
    bbox_min   : (3,) float
        Lower corner of mesh bounding box
    spacing    : (3,) float
        Per-axis voxel spacing (dx, dy, dz)
    """
    vox = voxelized_with_retry(
        mesh,
        pitch=pitch,
        max_attempts=4,
        factor=1.5,
        log_prefix="[mesh_to_fluid_mask] ",
    )
    vox_filled = vox.fill()
    fluid_mask = vox_filled.matrix.astype(bool)

    if not fluid_mask.any():
        raise RuntimeError("Voxelization produced an empty fluid mask.")

    # Axis-aligned bounding box of the mesh
    bbox_min, bbox_max = mesh.bounds  # shape (2,3)
    bbox_min = np.asarray(bbox_min, dtype=float)
    bbox_max = np.asarray(bbox_max, dtype=float)

    nx, ny, nz = fluid_mask.shape
    dims = np.array([nx, ny, nz], dtype=float)

    # Physical spacing per voxel in each direction
    spacing = (bbox_max - bbox_min) / dims

    return fluid_mask, bbox_min, spacing


def analyze_connectivity_voxel(
    mesh: trimesh.Trimesh,
    pitch: float = 0.1,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """
    Analyze connectivity in voxel space:
      - number of fluid components
      - components that touch the bounding box ("ports")
      - trapped fluid components and volume fraction

    Returns
    -------
    connectivity_info : dict
        Connectivity analysis results with both new and legacy keys
    fluid_mask        : (nx, ny, nz) bool
        Boolean array of voxel occupancy
    bbox_min          : (3,) float
        Lower corner of physical domain
    spacing           : (3,) float
        Voxel spacing (dx, dy, dz)
    """
    fluid_mask, bbox_min, spacing = mesh_to_fluid_mask(mesh, pitch=pitch)
    nx, ny, nz = fluid_mask.shape
    num_fluid_voxels = int(fluid_mask.sum())
    if num_fluid_voxels == 0:
        raise RuntimeError("Fluid mask has no voxels; check voxelization or mesh.")

    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    labels, num_labels = ndimage.label(fluid_mask, structure=structure)

    component_sizes = ndimage.sum(fluid_mask, labels, index=range(1, num_labels + 1))
    component_sizes = [int(s) for s in component_sizes]

    port_labels_set = set()
    for face_slice in [
        labels[0, :, :],
        labels[-1, :, :],
        labels[:, 0, :],
        labels[:, -1, :],
        labels[:, :, 0],
        labels[:, :, -1],
    ]:
        port_labels_set.update(np.unique(face_slice))
    port_labels_set.discard(0)
    port_labels = sorted(list(port_labels_set))

    if len(port_labels) > 0:
        reachable_voxels = sum(component_sizes[l - 1] for l in port_labels)
        reachable_fraction = reachable_voxels / num_fluid_voxels
    else:
        reachable_voxels = 0
        reachable_fraction = 0.0

    all_labels = np.arange(1, num_labels + 1)
    trapped_labels = sorted(list(set(all_labels) - set(port_labels)))
    trapped_sizes = [component_sizes[l - 1] for l in trapped_labels]

    pitch_mean = float(np.mean(spacing))

    connectivity_info = {
        "pitch_requested": float(pitch),
        "grid_shape": (nx, ny, nz),
        "bbox_min": bbox_min.tolist(),
        "spacing": spacing.tolist(),
        "pitch": pitch_mean,
        "shape": (nx, ny, nz),
        "num_fluid_voxels": num_fluid_voxels,
        "num_fluid_components": int(num_labels),
        "component_sizes": component_sizes,
        "num_port_components": len(port_labels),
        "port_component_labels": port_labels,
        "reachable_fraction": float(reachable_fraction),
        "num_trapped_components": len(trapped_labels),
        "trapped_component_labels": [int(l) for l in trapped_labels],
        "trapped_component_sizes": trapped_sizes,
    }

    return connectivity_info, fluid_mask, bbox_min, spacing
