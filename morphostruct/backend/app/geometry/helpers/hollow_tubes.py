"""
Functions for creating hollow tubular structures.

Provides utilities for generating hollow tubes, sinusoids, and vessel networks
using manifold3d. All functions work without pyvista dependencies.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np

try:
    import manifold3d as m3d
    HAS_MANIFOLD = True
except ImportError:
    m3d = None
    HAS_MANIFOLD = False


def _get_manifold():
    """Get manifold3d module, raising ImportError if not available."""
    if not HAS_MANIFOLD:
        raise ImportError(
            "manifold3d is required for geometry operations. "
            "Install with: pip install manifold3d"
        )
    return m3d


def create_hollow_vertical_tube(
    x: float,
    y: float,
    height: float,
    radius: float,
    wall_thickness: float,
    resolution: int = 16,
    cap_bottom: bool = True,
    cap_top: bool = True
) -> "m3d.Manifold":
    """
    Create a hollow vertical tube at position (x, y).

    Args:
        x: X-coordinate of tube center
        y: Y-coordinate of tube center
        height: Height of tube
        radius: Outer radius
        wall_thickness: Wall thickness (inner radius = radius - wall_thickness)
        resolution: Number of sides for cylinder
        cap_bottom: If True, close the bottom end (Z=0)
        cap_top: If True, close the top end (Z=height)

    Returns:
        Hollow tube manifold with specified end caps

    Raises:
        ImportError: If manifold3d is not available
    """
    m3d = _get_manifold()

    inner_radius = radius - wall_thickness
    if inner_radius <= 0:
        # If wall is too thick, just return solid cylinder
        return m3d.Manifold.cylinder(height, radius, radius, resolution).translate([x, y, 0])

    # Create outer cylinder
    outer = m3d.Manifold.cylinder(height, radius, radius, resolution)

    # Inner cylinder - extend beyond ends to ensure clean subtraction
    inner_ext = 0.1  # Extension amount
    inner_height = height
    inner_z_offset = 0

    if not cap_bottom:
        inner_height += inner_ext
        inner_z_offset = -inner_ext
    if not cap_top:
        inner_height += inner_ext

    inner = m3d.Manifold.cylinder(inner_height, inner_radius, inner_radius, resolution)
    inner = inner.translate([0, 0, inner_z_offset])

    # Subtract inner from outer to create hollow tube
    hollow = outer - inner

    return hollow.translate([x, y, 0])


def create_hollow_sinusoid(
    start_pos: Tuple[float, float, float],
    end_pos: Tuple[float, float, float],
    radius: float,
    wall_thickness: float,
    resolution: int = 8
) -> Optional["m3d.Manifold"]:
    """
    Create a hollow sinusoid channel between two points with both ends open.

    Args:
        start_pos: (x, y, z) start position
        end_pos: (x, y, z) end position
        radius: Outer radius
        wall_thickness: Wall thickness (inner radius = radius - wall_thickness)
        resolution: Number of sides for cylinder

    Returns:
        Hollow tube manifold with both ends open, or None if length < 0.01

    Raises:
        ImportError: If manifold3d is not available
    """
    m3d = _get_manifold()

    x1, y1, z1 = start_pos
    x2, y2, z2 = end_pos

    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    length = np.sqrt(dx*dx + dy*dy + dz*dz)

    if length < 0.01:
        return None

    inner_radius = radius - wall_thickness
    if inner_radius <= 0:
        # Wall too thick, return solid cylinder
        cyl = m3d.Manifold.cylinder(length, radius, radius, resolution)
        # Rotate to align with direction
        h = np.sqrt(dx*dx + dy*dy)
        if h > 0.001 or abs(dz) > 0.001:
            tilt = np.arctan2(h, dz) * 180 / np.pi
            azim = np.arctan2(dy, dx) * 180 / np.pi
            cyl = cyl.rotate([0, tilt, 0]).rotate([0, 0, azim])
        return cyl.translate([x1, y1, z1])

    # Create outer cylinder
    outer = m3d.Manifold.cylinder(length, radius, radius, resolution)

    # Create inner cylinder - extend beyond both ends to ensure open ends
    inner_ext = 0.1
    inner = m3d.Manifold.cylinder(length + 2 * inner_ext, inner_radius, inner_radius, resolution)
    inner = inner.translate([0, 0, -inner_ext])

    # Hollow tube
    hollow = outer - inner

    # Rotate to align with direction
    h = np.sqrt(dx*dx + dy*dy)
    if h > 0.001 or abs(dz) > 0.001:
        tilt = np.arctan2(h, dz) * 180 / np.pi
        azim = np.arctan2(dy, dx) * 180 / np.pi
        hollow = hollow.rotate([0, tilt, 0]).rotate([0, 0, azim])

    return hollow.translate([x1, y1, z1])


def create_solid_tube_manifold(
    points: List[Tuple[float, float, float]],
    radius_start: float,
    radius_end: float,
    segments: int = 16
) -> Optional["m3d.Manifold"]:
    """
    Create a SOLID tapered tube along a path using manifold3d.

    The tube follows the given points with linear interpolation of radius
    from radius_start to radius_end. Junction spheres are added at each
    intermediate point to ensure smooth connections.

    Args:
        points: List of (x, y, z) points defining the path
        radius_start: Radius at the start of the tube
        radius_end: Radius at the end of the tube
        segments: Number of segments for tube cross-section

    Returns:
        Solid tube manifold, or None if fewer than 2 points

    Raises:
        ImportError: If manifold3d is not available
    """
    if len(points) < 2:
        return None

    m3d = _get_manifold()
    from .mesh_utils import tree_union

    pts = np.array(points)
    n = len(pts)
    radii = np.linspace(radius_start, radius_end, n)

    solids = []
    for i in range(n - 1):
        p1 = pts[i]
        p2 = pts[i + 1]
        r1 = radii[i]
        r2 = radii[i + 1]

        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 1e-6:
            continue

        direction = direction / length
        cyl = m3d.Manifold.cylinder(length, r1, r2, segments)

        z_axis = np.array([0, 0, 1])
        dot = np.dot(z_axis, direction)

        if dot < -0.9999:
            cyl = cyl.rotate([180, 0, 0])
        elif dot < 0.9999:
            axis = np.cross(z_axis, direction)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(np.clip(dot, -1, 1)) * 180 / np.pi

            c = np.cos(np.radians(angle))
            s = np.sin(np.radians(angle))
            ux, uy, uz = axis

            rot_mat = np.array([
                [c + ux*ux*(1-c), ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s, 0],
                [uy*ux*(1-c) + uz*s, c + uy*uy*(1-c), uy*uz*(1-c) - ux*s, 0],
                [uz*ux*(1-c) - uy*s, uz*uy*(1-c) + ux*s, c + uz*uz*(1-c), 0]
            ])
            cyl = cyl.transform(rot_mat)

        cyl = cyl.translate(list(p1))
        solids.append(cyl)

        # Add junction sphere at intermediate points
        if i < n - 2:
            sphere = m3d.Manifold.sphere(r2 * 0.99, segments)
            sphere = sphere.translate(list(p2))
            solids.append(sphere)

    if not solids:
        return None

    return tree_union(solids)


def create_hollow_collector_manifold(
    points: List[Tuple[float, float, float]],
    radius_start: float,
    radius_end: float,
    wall_fraction: float = 0.15,
    segments: int = 16
) -> Optional["m3d.Manifold"]:
    """
    Create a HOLLOW tapered tube by boolean subtracting inner from outer.

    Args:
        points: List of (x, y, z) points defining the path
        radius_start: Outer radius at the start
        radius_end: Outer radius at the end
        wall_fraction: Wall thickness as fraction of radius (default 0.15 = 15%)
        segments: Number of segments for tube cross-section

    Returns:
        Hollow tube manifold, or None if fewer than 2 points

    Raises:
        ImportError: If manifold3d is not available
    """
    outer = create_solid_tube_manifold(points, radius_start, radius_end, segments)
    if outer is None:
        return None

    inner_r_start = radius_start * (1 - wall_fraction)
    inner_r_end = radius_end * (1 - wall_fraction)
    inner = create_solid_tube_manifold(points, inner_r_start, inner_r_end, segments)

    if inner is None:
        return outer

    return outer - inner


def create_hollow_vessel_network(
    tube_specs: List[Tuple[List[Tuple[float, float, float]], float, float]],
    wall_thickness: float = 0.02,
    segments: int = 16,
    junction_spheres: Optional[List[Tuple[Tuple[float, float, float], float]]] = None
) -> Optional["m3d.Manifold"]:
    """
    Create a unified hollow vessel network from multiple connected tubes.

    Extends tube paths slightly at endpoints to ensure overlap and proper
    connection even when bezier curve endpoints don't perfectly align.

    Args:
        tube_specs: List of (points, radius_start, radius_end) tuples where
                    points is a list of (x, y, z) coordinates
        wall_thickness: Thickness of tube walls (absolute, not fraction)
        segments: Number of segments for tube cross-section
        junction_spheres: Optional list of (position, radius) for explicit
                          junction smoothing at connection points

    Returns:
        Manifold3d object representing the hollow network with proper connections,
        or None if no valid tubes provided

    Raises:
        ImportError: If manifold3d is not available
    """
    if not tube_specs:
        return None

    m3d = _get_manifold()
    from .mesh_utils import tree_union

    exterior_solids = []
    interior_solids = []

    for points, r_start, r_end in tube_specs:
        pts = np.array(points)
        if len(pts) < 2:
            continue

        # Extend tube at both ends along the tangent direction
        # This ensures tubes overlap at junctions even if endpoints don't match exactly
        extend_amount = max(r_start, r_end) * 0.5

        # Extend at start
        tangent_start = pts[1] - pts[0]
        tangent_start = tangent_start / (np.linalg.norm(tangent_start) + 1e-10)
        new_start = pts[0] - tangent_start * extend_amount

        # Extend at end
        tangent_end = pts[-1] - pts[-2]
        tangent_end = tangent_end / (np.linalg.norm(tangent_end) + 1e-10)
        new_end = pts[-1] + tangent_end * extend_amount

        # Create extended path
        extended_pts = [new_start] + list(pts) + [new_end]

        # Exterior tube with extended path
        exterior = create_solid_tube_manifold(extended_pts, r_start, r_end, segments)
        if exterior:
            exterior_solids.append(exterior)

        # Interior tube (smaller by wall_thickness)
        inner_r_start = max(r_start - wall_thickness, r_start * 0.5)
        inner_r_end = max(r_end - wall_thickness, r_end * 0.5)
        interior = create_solid_tube_manifold(extended_pts, inner_r_start, inner_r_end, segments)
        if interior:
            interior_solids.append(interior)

    # Add explicit junction spheres if provided
    if junction_spheres:
        for pos, radius in junction_spheres:
            # Exterior sphere
            sphere_out = m3d.Manifold.sphere(radius, segments)
            sphere_out = sphere_out.translate(list(pos))
            exterior_solids.append(sphere_out)

            # Interior sphere (hollow)
            inner_r = max(radius - wall_thickness, radius * 0.5)
            sphere_in = m3d.Manifold.sphere(inner_r, segments)
            sphere_in = sphere_in.translate(list(pos))
            interior_solids.append(sphere_in)

    if not exterior_solids:
        return None

    # Union all exteriors
    exterior_union = tree_union(exterior_solids)

    # Union all interiors and subtract
    if interior_solids:
        interior_union = tree_union(interior_solids)
        return exterior_union - interior_union
    else:
        return exterior_union
