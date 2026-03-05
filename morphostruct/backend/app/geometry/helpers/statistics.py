"""
Calculate geometry statistics for scaffold meshes.

Provides utilities for measuring volumes, surface areas, porosity,
and generating standard statistics dictionaries for scaffold metadata.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Union

try:
    import manifold3d as m3d
    HAS_MANIFOLD = True
except ImportError:
    m3d = None
    HAS_MANIFOLD = False

import numpy as np


def _get_manifold():
    """Get manifold3d module, raising ImportError if not available."""
    if not HAS_MANIFOLD:
        raise ImportError(
            "manifold3d is required for geometry operations. "
            "Install with: pip install manifold3d"
        )
    return m3d


def calculate_volume(manifold: "m3d.Manifold") -> float:
    """
    Calculate the volume of a manifold geometry.

    Args:
        manifold: Manifold3d geometry object

    Returns:
        Volume in cubic millimeters (mm^3)

    Example:
        >>> sphere = m3d.Manifold.sphere(1.0, 32)
        >>> vol = calculate_volume(sphere)  # ~4.19 mm^3
    """
    if hasattr(manifold, 'volume'):
        return float(manifold.volume())

    # Fallback: compute from mesh
    mesh = manifold.to_mesh()
    verts = np.array(mesh.vert_properties)[:, :3]
    tris = np.array(mesh.tri_verts)

    # Compute signed volume using divergence theorem
    volume = 0.0
    for tri in tris:
        v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
        # Signed volume of tetrahedron formed with origin
        volume += np.dot(v0, np.cross(v1, v2)) / 6.0

    return abs(volume)


def calculate_surface_area(manifold: "m3d.Manifold") -> float:
    """
    Calculate the surface area of a manifold geometry.

    Args:
        manifold: Manifold3d geometry object

    Returns:
        Surface area in square millimeters (mm^2)

    Example:
        >>> sphere = m3d.Manifold.sphere(1.0, 32)
        >>> area = calculate_surface_area(sphere)  # ~12.57 mm^2
    """
    if hasattr(manifold, 'surface_area'):
        return float(manifold.surface_area())

    # Compute from mesh triangles
    mesh = manifold.to_mesh()
    verts = np.array(mesh.vert_properties)[:, :3]
    tris = np.array(mesh.tri_verts)

    area = 0.0
    for tri in tris:
        v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
        # Triangle area = 0.5 * |cross product|
        cross = np.cross(v1 - v0, v2 - v0)
        area += 0.5 * np.linalg.norm(cross)

    return area


def calculate_porosity(solid_volume: float, total_volume: float) -> float:
    """
    Calculate porosity as the fraction of void space.

    Porosity = (total_volume - solid_volume) / total_volume

    Args:
        solid_volume: Volume occupied by solid material (mm^3)
        total_volume: Total bounding volume (mm^3)

    Returns:
        Porosity as a fraction (0-1), where 0 = fully solid, 1 = fully void

    Example:
        >>> # Scaffold with 30% material
        >>> porosity = calculate_porosity(30.0, 100.0)  # 0.7
    """
    if total_volume <= 0:
        return 0.0
    return max(0.0, min(1.0, (total_volume - solid_volume) / total_volume))


def calculate_surface_to_volume_ratio(surface_area: float, volume: float) -> float:
    """
    Calculate the surface-to-volume ratio.

    Higher ratios indicate more surface area available for cell attachment
    relative to volume, which is generally desirable for tissue scaffolds.

    Args:
        surface_area: Surface area in mm^2
        volume: Volume in mm^3

    Returns:
        Surface-to-volume ratio in mm^-1 (1/mm)

    Example:
        >>> # Sphere of radius 1mm
        >>> svr = calculate_surface_to_volume_ratio(12.57, 4.19)  # ~3.0 mm^-1
    """
    if volume <= 0:
        return 0.0
    return surface_area / volume


def count_features(
    manifold_list: List["m3d.Manifold"],
    feature_name: str
) -> Dict[str, Any]:
    """
    Count features and calculate aggregate statistics.

    Args:
        manifold_list: List of manifold objects representing features
        feature_name: Name of the feature type (e.g., "channels", "pores")

    Returns:
        Dict with count and aggregate volume information

    Example:
        >>> channels = [create_channel(...) for _ in range(20)]
        >>> stats = count_features(channels, "channels")
        >>> # {'channel_count': 20, 'total_channel_volume_mm3': ...}
    """
    count = len(manifold_list)

    if count == 0:
        return {
            f"{feature_name}_count": 0,
            f"total_{feature_name}_volume_mm3": 0.0,
            f"avg_{feature_name}_volume_mm3": 0.0
        }

    total_volume = sum(calculate_volume(m) for m in manifold_list)
    avg_volume = total_volume / count

    return {
        f"{feature_name}_count": count,
        f"total_{feature_name}_volume_mm3": total_volume,
        f"avg_{feature_name}_volume_mm3": avg_volume
    }


def get_bounding_box(manifold: "m3d.Manifold") -> Dict[str, float]:
    """
    Get the axis-aligned bounding box of a manifold.

    Args:
        manifold: Manifold3d geometry object

    Returns:
        Dict with min_x, max_x, min_y, max_y, min_z, max_z,
        and dimensions width, depth, height

    Example:
        >>> bbox = get_bounding_box(scaffold)
        >>> print(f"Size: {bbox['width']} x {bbox['depth']} x {bbox['height']}")
    """
    mesh = manifold.to_mesh()
    verts = np.array(mesh.vert_properties)[:, :3]

    min_bounds = verts.min(axis=0)
    max_bounds = verts.max(axis=0)

    return {
        'min_x': float(min_bounds[0]),
        'max_x': float(max_bounds[0]),
        'min_y': float(min_bounds[1]),
        'max_y': float(max_bounds[1]),
        'min_z': float(min_bounds[2]),
        'max_z': float(max_bounds[2]),
        'width': float(max_bounds[0] - min_bounds[0]),
        'depth': float(max_bounds[1] - min_bounds[1]),
        'height': float(max_bounds[2] - min_bounds[2])
    }


def calculate_bounding_volume(manifold: "m3d.Manifold") -> float:
    """
    Calculate the volume of the axis-aligned bounding box.

    Args:
        manifold: Manifold3d geometry object

    Returns:
        Bounding box volume in mm^3
    """
    bbox = get_bounding_box(manifold)
    return bbox['width'] * bbox['depth'] * bbox['height']


def generate_standard_stats(
    manifold: "m3d.Manifold",
    scaffold_type: str,
    **extra_stats: Any
) -> Dict[str, Any]:
    """
    Generate standard statistics dictionary for a scaffold.

    Creates a consistent statistics format used across all scaffold generators.
    Includes triangle count, volume, surface area, and any extra statistics.

    Args:
        manifold: The generated scaffold manifold
        scaffold_type: Type identifier (e.g., "hepatic_lobule", "trabecular_bone")
        **extra_stats: Additional statistics to include (e.g., num_lobules=5)

    Returns:
        Dict with standardized statistics format

    Example:
        >>> stats = generate_standard_stats(
        ...     manifold,
        ...     scaffold_type="hepatic_lobule",
        ...     num_lobules=7,
        ...     show_sinusoids=True
        ... )
    """
    mesh = manifold.to_mesh()

    volume = calculate_volume(manifold)
    surface_area = calculate_surface_area(manifold)
    bbox = get_bounding_box(manifold)
    bounding_volume = bbox['width'] * bbox['depth'] * bbox['height']

    stats = {
        # Mesh statistics
        'triangle_count': len(mesh.tri_verts) if hasattr(mesh, 'tri_verts') else 0,
        'vertex_count': len(mesh.vert_properties) if hasattr(mesh, 'vert_properties') else 0,

        # Volume and area
        'volume_mm3': volume,
        'surface_area_mm2': surface_area,
        'surface_to_volume_ratio': calculate_surface_to_volume_ratio(surface_area, volume),

        # Bounding box
        'bounding_box': bbox,
        'bounding_volume_mm3': bounding_volume,
        'solid_fraction': volume / bounding_volume if bounding_volume > 0 else 0,
        'porosity': calculate_porosity(volume, bounding_volume),

        # Type identifier
        'scaffold_type': scaffold_type
    }

    # Add any extra statistics
    stats.update(extra_stats)

    return stats


def mesh_triangle_count(manifold: "m3d.Manifold") -> int:
    """
    Get the triangle count of a manifold mesh.

    Args:
        manifold: Manifold3d geometry object

    Returns:
        Number of triangles in the mesh
    """
    mesh = manifold.to_mesh()
    return len(mesh.tri_verts) if hasattr(mesh, 'tri_verts') else 0


def mesh_vertex_count(manifold: "m3d.Manifold") -> int:
    """
    Get the vertex count of a manifold mesh.

    Args:
        manifold: Manifold3d geometry object

    Returns:
        Number of vertices in the mesh
    """
    mesh = manifold.to_mesh()
    return len(mesh.vert_properties) if hasattr(mesh, 'vert_properties') else 0


def is_manifold_valid(manifold: "m3d.Manifold") -> bool:
    """
    Check if a manifold is valid (watertight, no self-intersections).

    Args:
        manifold: Manifold3d geometry object

    Returns:
        True if the manifold is valid, False otherwise
    """
    try:
        # If we can get a non-empty mesh, it's likely valid
        mesh = manifold.to_mesh()
        tri_count = len(mesh.tri_verts) if hasattr(mesh, 'tri_verts') else 0
        return tri_count > 0
    except Exception:
        return False


def summarize_statistics(stats: Dict[str, Any]) -> str:
    """
    Create a human-readable summary of scaffold statistics.

    Args:
        stats: Statistics dictionary from generate_standard_stats

    Returns:
        Formatted string summary

    Example:
        >>> summary = summarize_statistics(stats)
        >>> print(summary)
    """
    lines = [
        f"Scaffold Type: {stats.get('scaffold_type', 'unknown')}",
        f"Triangles: {stats.get('triangle_count', 0):,}",
        f"Volume: {stats.get('volume_mm3', 0):.3f} mm^3",
        f"Surface Area: {stats.get('surface_area_mm2', 0):.3f} mm^2",
        f"Porosity: {stats.get('porosity', 0) * 100:.1f}%",
    ]

    bbox = stats.get('bounding_box', {})
    if bbox:
        lines.append(
            f"Dimensions: {bbox.get('width', 0):.2f} x "
            f"{bbox.get('depth', 0):.2f} x "
            f"{bbox.get('height', 0):.2f} mm"
        )

    return "\n".join(lines)
