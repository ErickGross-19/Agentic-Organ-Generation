import numpy as np
import trimesh

from ..models import MeshDiagnostics, SurfaceQuality
from .voxel_utils import voxelized_with_retry


def count_degenerate_faces(mesh: trimesh.Trimesh, area_eps: float = 1e-18) -> int:
    """
    Count faces whose area is effectively zero.
    """
    areas = mesh.area_faces
    return int(np.sum(areas <= area_eps))


def estimate_voxel_volume(
    mesh: trimesh.Trimesh,
    pitch: float = 0.05,
    method: str = "ray",
) -> float | None:
    """
    Estimate volume by voxelization:
      - voxelize mesh at given pitch
      - fill interior
      - count voxels * pitch^3

    Returns None if voxelization fails.
    
    Automatically retries with larger pitch on memory errors.
    """
    try:
        vox = voxelized_with_retry(
            mesh,
            pitch=pitch,
            method=method,
            max_attempts=4,
            factor=1.5,
            log_prefix="[estimate_voxel_volume] ",
        )
        vox_filled = vox.fill()
        mask = vox_filled.matrix.astype(bool)
        num_voxels = int(mask.sum())
        return float(num_voxels * (pitch ** 3))
    except Exception:
        return None


def compute_diagnostics(
    mesh: trimesh.Trimesh,
    volume_pitch: float = 0.05,
    max_voxels: float = 3e7,
) -> MeshDiagnostics:
    """
    Compute core mesh health/topology metrics, including approximate volume.
    """
    watertight = mesh.is_watertight
    euler = mesh.euler_number

    num_vertices = int(mesh.vertices.shape[0])
    num_faces = int(mesh.faces.shape[0])

    components = mesh.split(only_watertight=False)
    num_components = len(components)

    try:
        if mesh.edges_unique is not None and mesh.edges_unique_inverse is not None:
            counts = np.bincount(mesh.edges_unique_inverse)
            non_manifold_edges = int(np.sum(counts > 2))
        else:
            non_manifold_edges = 0
    except AttributeError:
        non_manifold_edges = 0

    degenerate_faces = count_degenerate_faces(mesh)

    extents = mesh.extents.tolist()

    # Volume: if watertight use mesh.volume (fast and accurate), else voxel estimate
    if watertight:
        try:
            volume = float(mesh.volume)
            volume_source = "mesh"
        except Exception:
            volume = estimate_voxel_volume(mesh, pitch=volume_pitch)
            volume_source = "voxel_estimate"
    else:
        volume = estimate_voxel_volume(mesh, pitch=volume_pitch)
        volume_source = "voxel_estimate"

    return MeshDiagnostics(
        watertight=watertight,
        euler_number=int(euler),
        num_vertices=num_vertices,
        num_faces=num_faces,
        num_components=num_components,
        non_manifold_edges=non_manifold_edges,
        degenerate_faces=degenerate_faces,
        bounding_box_extents=extents,
        volume=volume,
        volume_source=volume_source,
    )


def compute_surface_quality(mesh: trimesh.Trimesh) -> SurfaceQuality:
    """
    Compute basic surface quality metrics:
      - face areas
      - edge lengths
      - triangle aspect ratios
    """
    v = mesh.vertices
    f = mesh.faces

    areas = mesh.area_faces
    min_face_area = float(areas.min())
    max_face_area = float(areas.max())
    mean_face_area = float(areas.mean())

    edges = mesh.edges_unique
    lengths = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)
    min_edge_length = float(lengths.min())
    max_edge_length = float(lengths.max())
    mean_edge_length = float(lengths.mean())

    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]

    e01 = np.linalg.norm(v1 - v0, axis=1)
    e12 = np.linalg.norm(v2 - v1, axis=1)
    e20 = np.linalg.norm(v0 - v2, axis=1)

    longest_edge = np.maximum(e01, np.maximum(e12, e20))
    h = 2.0 * areas / (longest_edge + 1e-16)
    aspect_ratio = longest_edge / (h + 1e-16)

    max_aspect_ratio = float(aspect_ratio.max())
    frac_over_10 = float(np.mean(aspect_ratio > 10.0))

    return SurfaceQuality(
        min_face_area=min_face_area,
        max_face_area=max_face_area,
        mean_face_area=mean_face_area,
        min_edge_length=min_edge_length,
        max_edge_length=max_edge_length,
        mean_edge_length=mean_edge_length,
        max_aspect_ratio=max_aspect_ratio,
        frac_aspect_ratio_over_10=frac_over_10,
    )
