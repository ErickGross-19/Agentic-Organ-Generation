"""
Pattern generation functions for biological structures.

Provides utilities for generating honeycomb, voronoi, fibonacci spiral,
and other biologically-inspired spatial patterns.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import numpy as np


def generate_honeycomb_centers(
    num_units: int,
    radius: float
) -> List[Tuple[float, float]]:
    """
    Generate center positions for hexagons in a honeycomb pattern.

    Adjacent hexagons share edges (and thus 2 corners each).
    Starts with center at origin, then adds neighbors in rings using
    axial coordinate BFS traversal.

    Args:
        num_units: Number of hexagonal units to generate
        radius: Radius of each hexagonal unit (distance from center to corner)

    Returns:
        List of (x, y) center positions

    Example:
        >>> centers = generate_honeycomb_centers(7, radius=1.0)
        >>> # Returns center hex + 6 surrounding hexes
    """
    if num_units <= 0:
        return []
    if num_units == 1:
        return [(0.0, 0.0)]

    hex_spacing = radius * np.sqrt(3)

    centers = [(0.0, 0.0)]
    added = {(0, 0)}

    # Axial coordinate directions for hexagon neighbors
    axial_dirs = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]

    queue = [(0, 0)]
    while len(centers) < num_units and queue:
        q, r = queue.pop(0)

        for dq, dr in axial_dirs:
            nq, nr = q + dq, r + dr
            if (nq, nr) not in added and len(centers) < num_units:
                added.add((nq, nr))
                x = hex_spacing * (nq + nr * 0.5)
                y = hex_spacing * nr * (np.sqrt(3) / 2)
                centers.append((x, y))
                queue.append((nq, nr))

    return centers[:num_units]


def generate_voronoi_pattern(
    bounding_box: Tuple[float, float, float, float],
    seed_density: float,
    seed: Optional[int] = None
) -> Tuple[List[Tuple[float, float]], List[List[int]]]:
    """
    Generate a 2D Voronoi pattern within a bounding box.

    Creates randomly distributed seed points and computes the Voronoi
    tessellation. Useful for creating organic cellular patterns.

    Args:
        bounding_box: (min_x, min_y, max_x, max_y) defining the region
        seed_density: Number of seed points per unit area
        seed: Random seed for reproducibility

    Returns:
        Tuple of:
        - List of (x, y) seed point positions
        - List of region vertex indices (for each seed)

    Example:
        >>> seeds, regions = generate_voronoi_pattern(
        ...     bounding_box=(0, 0, 10, 10),
        ...     seed_density=0.5  # ~50 cells in 100 sq units
        ... )
    """
    try:
        from scipy.spatial import Voronoi
    except ImportError:
        raise ImportError("scipy is required for Voronoi pattern generation")

    min_x, min_y, max_x, max_y = bounding_box
    width = max_x - min_x
    height = max_y - min_y
    area = width * height

    rng = np.random.default_rng(seed)

    # Generate seed points
    num_seeds = max(4, int(area * seed_density))
    seed_x = rng.uniform(min_x, max_x, num_seeds)
    seed_y = rng.uniform(min_y, max_y, num_seeds)
    seeds = list(zip(seed_x, seed_y))

    # Compute Voronoi diagram
    points = np.array(seeds)
    vor = Voronoi(points)

    # Extract regions (list of vertex indices for each seed)
    regions = []
    for region_idx in vor.point_region:
        region = vor.regions[region_idx]
        if -1 not in region and len(region) > 0:
            regions.append(region)
        else:
            regions.append([])

    return seeds, regions


def generate_fibonacci_spiral(
    num_points: int,
    radius: float,
    z_height: float = 0.0
) -> List[Tuple[float, float, float]]:
    """
    Generate points in a Fibonacci spiral (golden angle distribution).

    Creates a naturally uniform distribution of points that follows
    the golden angle pattern seen in sunflower seed heads, pinecones, etc.
    This is the most uniform way to distribute points in a circle.

    Args:
        num_points: Number of points to generate
        radius: Radius of the pattern
        z_height: Z-coordinate for all points (default 0)

    Returns:
        List of (x, y, z) positions in Fibonacci spiral

    Example:
        >>> # Generate 100 uniformly distributed points
        >>> points = generate_fibonacci_spiral(100, radius=5.0)
    """
    if num_points <= 0:
        return []

    golden_angle = np.pi * (3.0 - np.sqrt(5.0))  # ~137.5 degrees
    points = []

    for i in range(num_points):
        # Distance from center increases as sqrt to maintain uniform density
        r = radius * np.sqrt(i / num_points)
        theta = i * golden_angle

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append((x, y, z_height))

    return points


def get_corner_positions(
    center_x: float,
    center_y: float,
    radius: float,
    rotation_offset: float = np.radians(30)
) -> List[Tuple[float, float]]:
    """
    Get the 6 corner positions of a hexagon at given center.

    Args:
        center_x: X-coordinate of hexagon center
        center_y: Y-coordinate of hexagon center
        radius: Distance from center to corners
        rotation_offset: Angular offset for first corner (default 30 degrees)

    Returns:
        List of 6 (x, y) corner positions

    Example:
        >>> corners = get_corner_positions(0, 0, 1.0)
        >>> # Returns 6 corner positions of unit hexagon at origin
    """
    angles = np.linspace(0, 2 * np.pi, 7)[:-1] + rotation_offset
    return [
        (center_x + radius * np.cos(a), center_y + radius * np.sin(a))
        for a in angles
    ]


def find_unique_corners(
    centers: List[Tuple[float, float]],
    radius: float,
    tolerance: float = 0.05
) -> Tuple[List[Tuple[float, float]], Dict[int, List[int]]]:
    """
    Find all unique corner positions across multiple hexagons.

    When hexagons are arranged in a honeycomb pattern, adjacent hexagons
    share corners. This function identifies unique corners and tracks
    which hexagons share each corner.

    Args:
        centers: List of (x, y) hexagon center positions
        radius: Radius of hexagons (center to corner distance)
        tolerance: Distance threshold for considering corners identical

    Returns:
        Tuple of:
        - unique_corners: List of (x, y) unique corner positions
        - corner_to_units: Dict mapping corner index to list of unit indices
                           that share this corner

    Example:
        >>> centers = generate_honeycomb_centers(7, radius=1.0)
        >>> unique_corners, corner_to_units = find_unique_corners(centers, 1.0)
        >>> # Center corners are shared by up to 3 hexagons
    """
    all_corners = []

    for unit_idx, (cx, cy) in enumerate(centers):
        corners = get_corner_positions(cx, cy, radius)
        for corner_x, corner_y in corners:
            all_corners.append((corner_x, corner_y, unit_idx))

    unique_corners = []
    corner_to_units = {}

    for cx, cy, unit_idx in all_corners:
        found = False
        for unique_idx, (ux, uy) in enumerate(unique_corners):
            if abs(cx - ux) < tolerance and abs(cy - uy) < tolerance:
                if unit_idx not in corner_to_units[unique_idx]:
                    corner_to_units[unique_idx].append(unit_idx)
                found = True
                break

        if not found:
            unique_idx = len(unique_corners)
            unique_corners.append((cx, cy))
            corner_to_units[unique_idx] = [unit_idx]

    return unique_corners, corner_to_units


def generate_grid_centers(
    num_x: int,
    num_y: int,
    spacing_x: float,
    spacing_y: float,
    centered: bool = True
) -> List[Tuple[float, float]]:
    """
    Generate center positions in a rectangular grid pattern.

    Args:
        num_x: Number of positions in X direction
        num_y: Number of positions in Y direction
        spacing_x: Spacing between positions in X direction
        spacing_y: Spacing between positions in Y direction
        centered: If True, center the grid on origin

    Returns:
        List of (x, y) grid positions

    Example:
        >>> # 3x3 grid with 2mm spacing
        >>> centers = generate_grid_centers(3, 3, 2.0, 2.0)
    """
    centers = []

    for i in range(num_x):
        for j in range(num_y):
            x = i * spacing_x
            y = j * spacing_y
            centers.append((x, y))

    if centered and centers:
        # Calculate centroid and offset
        cx = sum(p[0] for p in centers) / len(centers)
        cy = sum(p[1] for p in centers) / len(centers)
        centers = [(p[0] - cx, p[1] - cy) for p in centers]

    return centers


def generate_radial_positions(
    num_rings: int,
    points_per_ring: int,
    ring_spacing: float,
    start_radius: float = 0.0,
    angular_offset_per_ring: float = 0.0
) -> List[Tuple[float, float]]:
    """
    Generate positions in concentric rings.

    Args:
        num_rings: Number of concentric rings
        points_per_ring: Number of points on each ring
        ring_spacing: Radial distance between rings
        start_radius: Radius of innermost ring (default 0 = include center)
        angular_offset_per_ring: Angular offset between successive rings

    Returns:
        List of (x, y) positions

    Example:
        >>> # 5 rings with 8 points each, alternating angles
        >>> positions = generate_radial_positions(
        ...     num_rings=5, points_per_ring=8, ring_spacing=1.0,
        ...     angular_offset_per_ring=np.pi/8
        ... )
    """
    positions = []

    # Include center point if start_radius is 0
    if start_radius == 0:
        positions.append((0.0, 0.0))
        actual_start = ring_spacing
    else:
        actual_start = start_radius

    for ring_idx in range(num_rings):
        radius = actual_start + ring_idx * ring_spacing
        base_offset = ring_idx * angular_offset_per_ring

        for point_idx in range(points_per_ring):
            angle = base_offset + (2 * np.pi * point_idx / points_per_ring)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            positions.append((x, y))

    return positions


def generate_hexagonal_packing(
    bounding_radius: float,
    unit_radius: float
) -> List[Tuple[float, float]]:
    """
    Generate positions using optimal hexagonal packing (circle packing).

    Hexagonal packing is the densest way to pack circles in 2D.
    Different from honeycomb_centers which places hex CELL centers;
    this places CIRCLE centers for optimal packing density.

    Args:
        bounding_radius: Radius of circular bounding region
        unit_radius: Radius of circles being packed

    Returns:
        List of (x, y) circle center positions

    Example:
        >>> # Pack 1mm spheres into a 10mm circle
        >>> positions = generate_hexagonal_packing(10.0, 1.0)
    """
    positions = [(0.0, 0.0)]

    # Row spacing for hexagonal packing
    row_height = unit_radius * np.sqrt(3)
    col_width = unit_radius * 2

    # Generate positions row by row
    row = 0
    while True:
        y = row * row_height
        if y > bounding_radius + unit_radius:
            break

        # Offset every other row
        x_offset = (unit_radius if row % 2 else 0)

        col = 0
        while True:
            x = x_offset + col * col_width
            if x > bounding_radius + unit_radius:
                break

            # Check if position is within bounding circle
            if np.sqrt(x*x + y*y) <= bounding_radius - unit_radius:
                if x != 0 or y != 0:  # Don't duplicate origin
                    positions.append((x, y))
                if x != 0:  # Mirror across Y axis
                    positions.append((-x, y))
                if y != 0:  # Mirror across X axis
                    positions.append((x, -y))
                if x != 0 and y != 0:  # Mirror across both
                    positions.append((-x, -y))

            col += 1
        row += 1

    return positions
