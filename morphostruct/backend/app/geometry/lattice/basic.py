"""
Lattice scaffold generator for cubic and BCC (body-centered cubic) structures.

Provides parametric generation of lattice-based scaffolds with:
- Cubic unit cells (12 edge struts per cell)
- BCC unit cells (12 edge struts + 8 diagonal struts per cell)
- FCC unit cells (face-centered cubic)
- Configurable cell size, strut diameter, and resolution
- Strut tapering for biomimetic gradients
- Multiple cross-section profiles (circular, square, hexagonal)
- Density gradients along any axis
- Node filleting and spheres for smooth transitions
"""

from __future__ import annotations
import manifold3d as m3d
import numpy as np
from dataclasses import dataclass
from typing import Literal
from ..core import batch_union


# =============================================================================
# Helper Functions for Advanced Strut Features
# =============================================================================

def _create_tapered_strut(
    p1: np.ndarray,
    p2: np.ndarray,
    radius_node: float,
    taper: float,
    resolution: int,
    segments: int = 6
) -> m3d.Manifold:
    """
    Create a tapered strut between two points.

    The strut is thicker at the nodes (endpoints) and thinner at the midpoint.
    This biomimetic design mimics natural bone trabeculae which are thicker at
    junctions for better load distribution.

    Formula: radius(t) = r_mid + (r_node - r_mid) * |2t - 1|
    where t ∈ [0,1], r_mid = r_node * (1 - taper)

    Args:
        p1: Starting point (x, y, z)
        p2: Ending point (x, y, z)
        radius_node: Radius at the node (endpoint)
        taper: Taper factor (0 = no taper, 1 = full taper to zero at midpoint)
        resolution: Number of segments around cylinder
        segments: Number of segments along the strut length

    Returns:
        Manifold representing the tapered strut
    """
    direction = p2 - p1
    length = np.linalg.norm(direction)

    if length < 1e-6:
        return m3d.Manifold()

    # Calculate midpoint radius
    radius_mid = radius_node * (1.0 - min(taper, 0.9))  # Cap taper at 0.9 to avoid zero radius

    # Create strut as series of cone frustums
    strut_parts = []
    for i in range(segments):
        t_start = i / segments
        t_end = (i + 1) / segments
        t_mid_start = abs(2 * t_start - 1)
        t_mid_end = abs(2 * t_end - 1)

        r_start = radius_mid + (radius_node - radius_mid) * t_mid_start
        r_end = radius_mid + (radius_node - radius_mid) * t_mid_end

        seg_start = t_start * length
        seg_end = t_end * length
        seg_height = seg_end - seg_start

        # Create cone frustum along Z axis
        frustum = m3d.Manifold.cylinder(seg_height, r_start, r_end, resolution)
        frustum = frustum.translate([0, 0, seg_start])
        strut_parts.append(frustum)

    if not strut_parts:
        return m3d.Manifold()

    # Union all frustum segments
    strut = strut_parts[0]
    for part in strut_parts[1:]:
        strut = strut + part

    # Rotate to align with direction vector
    dx, dy, dz = direction
    h = np.sqrt(dx*dx + dy*dy)
    if h > 0.001 or abs(dz) > 0.001:
        angle_y = np.arctan2(h, dz) * 180 / np.pi
        angle_z = np.arctan2(dy, dx) * 180 / np.pi
        strut = strut.rotate([0, angle_y, 0])
        strut = strut.rotate([0, 0, angle_z])

    return strut.translate([p1[0], p1[1], p1[2]])


def _create_square_strut(
    p1: np.ndarray,
    p2: np.ndarray,
    radius: float,
    taper: float = 0.0,
    segments: int = 6
) -> m3d.Manifold:
    """
    Create a square cross-section strut between two points.

    Args:
        p1: Starting point
        p2: Ending point
        radius: Half-width of the square (equivalent to cylinder radius)
        taper: Taper factor (0 = no taper)
        segments: Number of segments for tapering

    Returns:
        Manifold representing the square strut
    """
    direction = p2 - p1
    length = np.linalg.norm(direction)

    if length < 1e-6:
        return m3d.Manifold()

    # Side length from equivalent radius
    side = radius * 2

    if taper > 0:
        # Tapered square strut
        side_mid = side * (1.0 - min(taper, 0.9))
        strut_parts = []

        for i in range(segments):
            t_start = i / segments
            t_end = (i + 1) / segments
            t_mid_start = abs(2 * t_start - 1)
            t_mid_end = abs(2 * t_end - 1)

            s_start = side_mid + (side - side_mid) * t_mid_start
            s_end = side_mid + (side - side_mid) * t_mid_end

            seg_start = t_start * length
            seg_end = t_end * length
            seg_height = seg_end - seg_start

            # Create prism using extrusion of cross-section polygon
            # Use hull of bottom and top squares for frustum effect
            half_s_start = s_start / 2
            half_s_end = s_end / 2

            # Bottom square
            bottom = m3d.Manifold.cube([s_start, s_start, seg_height * 0.01])
            bottom = bottom.translate([-half_s_start, -half_s_start, seg_start])

            # Top square
            top = m3d.Manifold.cube([s_end, s_end, seg_height * 0.01])
            top = top.translate([-half_s_end, -half_s_end, seg_end - seg_height * 0.01])

            # For tapered squares, we'll use a simple box with average size
            avg_side = (s_start + s_end) / 2
            box = m3d.Manifold.cube([avg_side, avg_side, seg_height])
            box = box.translate([-avg_side/2, -avg_side/2, seg_start])
            strut_parts.append(box)

        strut = strut_parts[0]
        for part in strut_parts[1:]:
            strut = strut + part
    else:
        # Simple square strut
        strut = m3d.Manifold.cube([side, side, length])
        strut = strut.translate([-radius, -radius, 0])

    # Rotate to align with direction
    dx, dy, dz = direction
    h = np.sqrt(dx*dx + dy*dy)
    if h > 0.001 or abs(dz) > 0.001:
        angle_y = np.arctan2(h, dz) * 180 / np.pi
        angle_z = np.arctan2(dy, dx) * 180 / np.pi
        strut = strut.rotate([0, angle_y, 0])
        strut = strut.rotate([0, 0, angle_z])

    return strut.translate([p1[0], p1[1], p1[2]])


def _create_hexagonal_strut(
    p1: np.ndarray,
    p2: np.ndarray,
    radius: float,
    resolution: int = 6
) -> m3d.Manifold:
    """
    Create a hexagonal cross-section strut between two points.

    The hexagonal profile provides better packing efficiency and is commonly
    used in honeycomb structures.

    Args:
        p1: Starting point
        p2: Ending point
        radius: Circumradius of the hexagon
        resolution: Ignored (always 6 sides for hexagon)

    Returns:
        Manifold representing the hexagonal strut
    """
    direction = p2 - p1
    length = np.linalg.norm(direction)

    if length < 1e-6:
        return m3d.Manifold()

    # Create hexagonal cylinder (6-sided polygon extruded)
    strut = m3d.Manifold.cylinder(length, radius, radius, 6)

    # Rotate to align with direction
    dx, dy, dz = direction
    h = np.sqrt(dx*dx + dy*dy)
    if h > 0.001 or abs(dz) > 0.001:
        angle_y = np.arctan2(h, dz) * 180 / np.pi
        angle_z = np.arctan2(dy, dx) * 180 / np.pi
        strut = strut.rotate([0, angle_y, 0])
        strut = strut.rotate([0, 0, angle_z])

    return strut.translate([p1[0], p1[1], p1[2]])


def _create_elliptical_strut(
    p1: np.ndarray,
    p2: np.ndarray,
    radius: float,
    resolution: int = 16
) -> m3d.Manifold:
    """
    Create an elliptical cross-section strut between two points.

    The elliptical profile provides directional stiffness and can be used
    to create anisotropic mechanical properties.

    Args:
        p1: Starting point
        p2: Ending point
        radius: Minor axis radius (major axis will be 1.5x larger)
        resolution: Number of segments around the ellipse

    Returns:
        Manifold representing the elliptical strut
    """
    direction = p2 - p1
    length = np.linalg.norm(direction)

    if length < 1e-6:
        return m3d.Manifold()

    # Create ellipse cross-section with aspect ratio 1.5:1
    major_radius = radius * 1.5
    minor_radius = radius

    # Create elliptical cylinder by scaling a circular cylinder
    strut = m3d.Manifold.cylinder(length, radius, radius, resolution)
    # Scale in X to create ellipse
    strut = strut.scale([major_radius / radius, 1.0, 1.0])

    # Rotate to align with direction
    dx, dy, dz = direction
    h = np.sqrt(dx*dx + dy*dy)
    if h > 0.001 or abs(dz) > 0.001:
        angle_y = np.arctan2(h, dz) * 180 / np.pi
        angle_z = np.arctan2(dy, dx) * 180 / np.pi
        strut = strut.rotate([0, angle_y, 0])
        strut = strut.rotate([0, 0, angle_z])

    return strut.translate([p1[0], p1[1], p1[2]])


def _calculate_gradient_diameter(
    position: np.ndarray,
    base_diameter: float,
    gradient_axis: str,
    gradient_start: float,
    gradient_end: float,
    bbox: tuple[float, float, float]
) -> float:
    """
    Calculate strut diameter based on position along gradient axis.

    Implements a linear density gradient where strut diameter varies
    based on the normalized position along the specified axis.

    Args:
        position: Midpoint position of the strut
        base_diameter: Base strut diameter (mm)
        gradient_axis: Axis for gradient ('x', 'y', or 'z')
        gradient_start: Density factor at start (0-1)
        gradient_end: Density factor at end (0-1)
        bbox: Bounding box dimensions (x, y, z)

    Returns:
        Adjusted strut diameter based on gradient position
    """
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map.get(gradient_axis.lower(), 2)

    # Get normalized position along axis (0 to 1)
    axis_extent = bbox[axis_idx]
    if axis_extent < 1e-6:
        return base_diameter

    normalized_pos = position[axis_idx] / axis_extent
    normalized_pos = max(0.0, min(1.0, normalized_pos))

    # Linear interpolation of density factor
    density_factor = gradient_start + (gradient_end - gradient_start) * normalized_pos

    # Diameter scales with square root of density for constant mass per unit length
    # But for visual effect, linear scaling works better
    return base_diameter * density_factor


def _create_node_sphere(
    position: np.ndarray,
    radius: float,
    resolution: int
) -> m3d.Manifold:
    """
    Create a spherical node at a junction point.

    Spherical nodes provide:
    - Smooth stress distribution at junctions
    - Better mechanical properties
    - More natural appearance similar to trabecular bone

    Args:
        position: Center position of the sphere
        radius: Sphere radius
        resolution: Number of segments for sphere

    Returns:
        Manifold representing the node sphere
    """
    sphere = m3d.Manifold.sphere(radius, resolution)
    return sphere.translate([position[0], position[1], position[2]])


def _create_fillet_sphere(
    position: np.ndarray,
    radius: float,
    resolution: int
) -> m3d.Manifold:
    """
    Create a fillet sphere at a junction for smooth transitions.

    Fillets reduce stress concentrations at strut junctions by
    creating smooth transitions between intersecting struts.

    Args:
        position: Center position of the fillet
        radius: Fillet sphere radius
        resolution: Number of segments

    Returns:
        Manifold representing the fillet sphere
    """
    return _create_node_sphere(position, radius, resolution)


def _collect_node_positions(
    strut_endpoints: set,
    cell_size: float
) -> set[tuple[float, float, float]]:
    """
    Collect all unique node positions from strut endpoints.

    Args:
        strut_endpoints: Set of strut endpoint tuples
        cell_size: Unit cell size for rounding tolerance

    Returns:
        Set of unique node position tuples
    """
    nodes = set()
    tolerance = cell_size * 0.01  # 1% of cell size for grouping

    for (p1_tuple, p2_tuple) in strut_endpoints:
        # Round to tolerance for deduplication
        p1_rounded = tuple(round(c / tolerance) * tolerance for c in p1_tuple)
        p2_rounded = tuple(round(c / tolerance) * tolerance for c in p2_tuple)
        nodes.add(p1_rounded)
        nodes.add(p2_rounded)

    return nodes


@dataclass
class LatticeParams:
    """
    Parameters for basic lattice scaffold generation with biologically realistic defaults.

    Supports cubic, BCC (body-centered cubic), and FCC (face-centered cubic) unit cells.
    These lattice types provide predictable mechanical properties and are widely used
    in bone tissue engineering.

    Literature references:
    - Strut diameter: 0.3-0.8mm for bone scaffolds
    - Unit cell size: 1.5-3.0mm for cell infiltration
    - Porosity: 60-80% for trabecular bone-like structures
    - Pore size: 300-800 μm for bone ingrowth

    Attributes:
        # === Basic Geometry ===
        bounding_box_x_mm: Scaffold X dimension in millimeters
        bounding_box_y_mm: Scaffold Y dimension in millimeters
        bounding_box_z_mm: Scaffold Z dimension in millimeters
        unit_cell_size_mm: Unit cell size (1.5-3.0mm)

        # === Lattice Type ===
        lattice_type: Unit cell type ('cubic', 'bcc', 'fcc')

        # === Strut Properties ===
        strut_diameter_mm: Strut diameter (0.3-0.8mm)
        strut_taper: Taper ratio from center to joints (0-0.3)
        strut_profile: Cross-section profile ('circular', 'square', 'hexagonal', 'elliptical')

        # === Porosity Control ===
        porosity: Target void fraction (0.6-0.8)
        pore_size_um: Target pore size (300-800 μm)

        # === Gradient Features ===
        enable_gradient_density: Enable density gradient
        gradient_axis: Axis for gradient ('x', 'y', or 'z')
        gradient_start_density: Strut density at gradient start
        gradient_end_density: Strut density at gradient end

        # === Node Features ===
        enable_node_filleting: Add fillets at strut joints
        fillet_radius_factor: Fillet radius as fraction of strut radius
        enable_node_spheres: Add spherical nodes at joints

        # === Quality & Generation ===
        resolution: Cylinder segments (6-16)
        seed: Random seed for stochastic features
    """
    # Basic Geometry
    bounding_box_x_mm: float = 10.0
    bounding_box_y_mm: float = 10.0
    bounding_box_z_mm: float = 10.0
    unit_cell_size_mm: float = 2.0  # 1.5-3.0mm for bone scaffolds

    # Lattice Type
    lattice_type: Literal['cubic', 'bcc', 'fcc'] = 'cubic'

    # Strut Properties
    strut_diameter_mm: float = 0.5  # 0.3-0.8mm for bone
    strut_taper: float = 0.0
    strut_profile: str = 'circular'  # 'circular', 'square', 'elliptical'

    # Porosity Control
    porosity: float = 0.7  # 60-80% for trabecular-like
    pore_size_um: float = 500.0  # 300-800 μm for bone ingrowth

    # Gradient Features
    enable_gradient_density: bool = False
    gradient_axis: str = 'z'
    gradient_start_density: float = 0.5
    gradient_end_density: float = 1.0

    # Node Features
    enable_node_filleting: bool = False
    fillet_radius_factor: float = 0.3
    enable_node_spheres: bool = False
    node_sphere_factor: float = 1.1

    # Quality & Generation
    resolution: int = 8
    seed: int | None = None

    # Legacy compatibility
    bounding_box_mm: tuple[float, float, float] | None = None
    unit_cell: Literal['cubic', 'bcc'] | None = None  # Maps to lattice_type
    cell_size_mm: float | None = None  # Maps to unit_cell_size_mm


def make_strut(
    p1: np.ndarray,
    p2: np.ndarray,
    radius: float,
    resolution: int,
    taper: float = 0.0,
    profile: str = 'circular'
) -> m3d.Manifold:
    """
    Create a strut between two points with configurable profile and taper.

    Args:
        p1: Starting point (x, y, z)
        p2: Ending point (x, y, z)
        radius: Strut radius (or equivalent for non-circular profiles)
        resolution: Number of segments around cylinder
        taper: Taper factor (0 = no taper, >0 = thicker at ends)
        profile: Cross-section profile ('circular', 'square', 'hexagonal', 'elliptical')

    Returns:
        Manifold representing the strut
    """
    # Handle tapered circular struts
    if taper > 0 and profile == 'circular':
        return _create_tapered_strut(p1, p2, radius, taper, resolution)

    # Handle square profile
    if profile == 'square':
        return _create_square_strut(p1, p2, radius, taper)

    # Handle hexagonal profile
    if profile == 'hexagonal':
        return _create_hexagonal_strut(p1, p2, radius, resolution)

    # Handle elliptical profile
    if profile == 'elliptical':
        return _create_elliptical_strut(p1, p2, radius, resolution)

    # Default: circular profile without taper
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    length = np.sqrt(dx*dx + dy*dy + dz*dz)

    if length < 1e-6:
        return m3d.Manifold()

    # Create cylinder along Z axis (from z=0 to z=length)
    strut = m3d.Manifold.cylinder(length, radius, radius, resolution)

    # Calculate rotation angles to align with direction vector (p1 -> p2)
    h = np.sqrt(dx*dx + dy*dy)  # horizontal distance
    if h > 0.001 or abs(dz) > 0.001:
        # Rotate to align with direction vector
        # First rotate around Y to tilt from vertical
        angle_y = np.arctan2(h, dz) * 180 / np.pi
        # Then rotate around Z to point in correct horizontal direction
        angle_z = np.arctan2(dy, dx) * 180 / np.pi

        strut = strut.rotate([0, angle_y, 0])
        strut = strut.rotate([0, 0, angle_z])

    # Translate to starting position
    return strut.translate([p1[0], p1[1], p1[2]])


def get_cubic_struts(cell_origin: np.ndarray, cell_size: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Get strut endpoints for a cubic unit cell.

    A cubic cell has 8 corner nodes connected by 12 edge struts:
    - 4 struts along X direction
    - 4 struts along Y direction
    - 4 struts along Z direction

    Args:
        cell_origin: Origin point (x, y, z) of the unit cell
        cell_size: Size of the unit cell

    Returns:
        List of (p1, p2) tuples defining strut endpoints
    """
    # 8 corners of the cube
    corners = []
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                corners.append(cell_origin + np.array([i, j, k]) * cell_size)

    # 12 edges connecting adjacent corners
    struts = []

    # X-direction edges (4 struts)
    for j in [0, 1]:
        for k in [0, 1]:
            p1 = cell_origin + np.array([0, j, k]) * cell_size
            p2 = cell_origin + np.array([1, j, k]) * cell_size
            struts.append((p1, p2))

    # Y-direction edges (4 struts)
    for i in [0, 1]:
        for k in [0, 1]:
            p1 = cell_origin + np.array([i, 0, k]) * cell_size
            p2 = cell_origin + np.array([i, 1, k]) * cell_size
            struts.append((p1, p2))

    # Z-direction edges (4 struts)
    for i in [0, 1]:
        for j in [0, 1]:
            p1 = cell_origin + np.array([i, j, 0]) * cell_size
            p2 = cell_origin + np.array([i, j, 1]) * cell_size
            struts.append((p1, p2))

    return struts


def get_bcc_struts(cell_origin: np.ndarray, cell_size: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Get strut endpoints for BCC (body-centered cubic) unit cell.

    A BCC cell has all cubic struts plus a center node connected by 8 diagonal struts
    to the corner nodes.

    Args:
        cell_origin: Origin point (x, y, z) of the unit cell
        cell_size: Size of the unit cell

    Returns:
        List of (p1, p2) tuples defining strut endpoints (12 edges + 8 diagonals = 20 struts)
    """
    # Start with cubic edge struts
    struts = get_cubic_struts(cell_origin, cell_size)

    # Add center node and diagonal connections to corners
    center = cell_origin + np.array([0.5, 0.5, 0.5]) * cell_size

    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                corner = cell_origin + np.array([i, j, k]) * cell_size
                struts.append((center, corner))

    return struts


def get_fcc_struts(cell_origin: np.ndarray, cell_size: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Get strut endpoints for FCC (face-centered cubic) unit cell.

    An FCC cell has corner nodes plus face center nodes, with struts connecting
    each face center to its 4 adjacent corners.

    Args:
        cell_origin: Origin point (x, y, z) of the unit cell
        cell_size: Size of the unit cell

    Returns:
        List of (p1, p2) tuples defining strut endpoints
    """
    L = cell_size
    o = cell_origin
    struts = []

    # Face centers
    face_centers = [
        o + np.array([0.5, 0.5, 0]) * L,    # bottom
        o + np.array([0.5, 0.5, 1]) * L,    # top
        o + np.array([0.5, 0, 0.5]) * L,    # front
        o + np.array([0.5, 1, 0.5]) * L,    # back
        o + np.array([0, 0.5, 0.5]) * L,    # left
        o + np.array([1, 0.5, 0.5]) * L,    # right
    ]

    # Connect each face center to its 4 corners
    # Bottom face (z=0)
    for i in [0, 1]:
        for j in [0, 1]:
            corner = o + np.array([i, j, 0]) * L
            struts.append((face_centers[0], corner))

    # Top face (z=1)
    for i in [0, 1]:
        for j in [0, 1]:
            corner = o + np.array([i, j, 1]) * L
            struts.append((face_centers[1], corner))

    # Front face (y=0)
    for i in [0, 1]:
        for k in [0, 1]:
            corner = o + np.array([i, 0, k]) * L
            struts.append((face_centers[2], corner))

    # Back face (y=1)
    for i in [0, 1]:
        for k in [0, 1]:
            corner = o + np.array([i, 1, k]) * L
            struts.append((face_centers[3], corner))

    # Left face (x=0)
    for j in [0, 1]:
        for k in [0, 1]:
            corner = o + np.array([0, j, k]) * L
            struts.append((face_centers[4], corner))

    # Right face (x=1)
    for j in [0, 1]:
        for k in [0, 1]:
            corner = o + np.array([1, j, k]) * L
            struts.append((face_centers[5], corner))

    return struts


def generate_lattice(params: LatticeParams) -> tuple[m3d.Manifold, dict]:
    """
    Generate a lattice scaffold with advanced features.

    Supports:
    - Multiple lattice types (cubic, BCC, FCC)
    - Strut tapering (biomimetic - thicker at nodes)
    - Multiple strut profiles (circular, square, hexagonal)
    - Density gradients along any axis
    - Node spheres for smooth junctions
    - Filleting at strut intersections

    Args:
        params: LatticeParams specifying geometry and structure

    Returns:
        Tuple of (manifold, stats_dict)
        - manifold: The 3D geometry
        - stats_dict: Dictionary with comprehensive statistics

    Raises:
        ValueError: If no struts are generated
    """
    # Handle both legacy tuple format and new individual params
    if params.bounding_box_mm is not None:
        bx, by, bz = params.bounding_box_mm
    else:
        bx = params.bounding_box_x_mm
        by = params.bounding_box_y_mm
        bz = params.bounding_box_z_mm

    bbox = (bx, by, bz)

    # Handle legacy cell_size_mm or new unit_cell_size_mm
    cell = params.cell_size_mm if params.cell_size_mm is not None else params.unit_cell_size_mm
    base_radius = params.strut_diameter_mm / 2

    # Extract strut parameters
    strut_taper = params.strut_taper
    strut_profile = params.strut_profile.lower() if params.strut_profile else 'circular'

    # Validate strut profile
    valid_profiles = ('circular', 'square', 'hexagonal', 'elliptical')
    if strut_profile not in valid_profiles:
        strut_profile = 'circular'

    # Extract gradient parameters
    enable_gradient = params.enable_gradient_density
    gradient_axis = params.gradient_axis.lower() if params.gradient_axis else 'z'
    gradient_start = params.gradient_start_density
    gradient_end = params.gradient_end_density

    # Extract node feature parameters
    enable_node_spheres = params.enable_node_spheres
    node_sphere_factor = params.node_sphere_factor
    enable_filleting = params.enable_node_filleting
    fillet_factor = params.fillet_radius_factor

    # Calculate number of cells in each dimension
    nx = max(1, int(bx / cell))
    ny = max(1, int(by / cell))
    nz = max(1, int(bz / cell))

    # Get strut function based on unit cell type
    lattice_type = params.unit_cell if params.unit_cell is not None else params.lattice_type
    if lattice_type == 'bcc':
        get_struts = get_bcc_struts
    elif lattice_type == 'fcc':
        get_struts = get_fcc_struts
    else:
        get_struts = get_cubic_struts

    # Collect all unique struts (avoid duplicates at cell boundaries)
    all_strut_endpoints = set()

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                origin = np.array([i, j, k], dtype=float) * cell
                struts = get_struts(origin, cell)
                for p1, p2 in struts:
                    # Normalize order for deduplication (smaller tuple first)
                    key = tuple(sorted([tuple(p1.round(6)), tuple(p2.round(6))]))
                    all_strut_endpoints.add(key)

    # Create strut manifolds with advanced features
    strut_manifolds = []
    for (p1_tuple, p2_tuple) in all_strut_endpoints:
        p1 = np.array(p1_tuple)
        p2 = np.array(p2_tuple)

        # Calculate strut midpoint for gradient
        midpoint = (p1 + p2) / 2

        # Apply gradient if enabled
        if enable_gradient:
            adjusted_diameter = _calculate_gradient_diameter(
                midpoint,
                params.strut_diameter_mm,
                gradient_axis,
                gradient_start,
                gradient_end,
                bbox
            )
            radius = adjusted_diameter / 2
        else:
            radius = base_radius

        # Create strut with taper and profile
        strut = make_strut(
            p1, p2,
            radius,
            params.resolution,
            taper=strut_taper,
            profile=strut_profile
        )

        if strut.num_vert() > 0:
            strut_manifolds.append(strut)

    # Union all struts
    if not strut_manifolds:
        raise ValueError("No struts generated")

    result = batch_union(strut_manifolds)

    # Add node features (spheres and/or fillets)
    if enable_node_spheres or enable_filleting:
        node_positions = _collect_node_positions(all_strut_endpoints, cell)
        node_manifolds = []

        for pos_tuple in node_positions:
            pos = np.array(pos_tuple)

            # Calculate local radius based on gradient if enabled
            if enable_gradient:
                local_diameter = _calculate_gradient_diameter(
                    pos,
                    params.strut_diameter_mm,
                    gradient_axis,
                    gradient_start,
                    gradient_end,
                    bbox
                )
                local_radius = local_diameter / 2
            else:
                local_radius = base_radius

            # Add node sphere
            if enable_node_spheres:
                sphere_radius = local_radius * node_sphere_factor
                sphere = _create_node_sphere(pos, sphere_radius, params.resolution)
                if sphere.num_vert() > 0:
                    node_manifolds.append(sphere)

            # Add fillet sphere (smaller than node sphere)
            if enable_filleting and not enable_node_spheres:
                # Fillet is added only if node spheres are not enabled
                # (since node spheres already provide smooth transitions)
                fillet_radius = local_radius * fillet_factor
                fillet = _create_fillet_sphere(pos, fillet_radius, params.resolution)
                if fillet.num_vert() > 0:
                    node_manifolds.append(fillet)

        # Union node features with struts
        if node_manifolds:
            node_union = batch_union(node_manifolds)
            result = result + node_union

    # Clip to bounding box (intersection, not XOR)
    clip_box = m3d.Manifold.cube([bx, by, bz])
    result = result & clip_box

    # Calculate statistics
    mesh = result.to_mesh()
    volume = result.volume() if hasattr(result, 'volume') else 0
    solid_volume = bx * by * bz
    relative_density = volume / solid_volume if solid_volume > 0 else 0
    porosity = 1.0 - relative_density

    # Estimate pore size based on geometry
    estimated_pore_size_um = (cell - params.strut_diameter_mm) * 1000 / 2

    # Count node features
    node_count = len(_collect_node_positions(all_strut_endpoints, cell)) if (enable_node_spheres or enable_filleting) else 0

    stats = {
        'triangle_count': len(mesh.tri_verts) // 3 if hasattr(mesh, 'tri_verts') else 0,
        'vertex_count': len(mesh.vert_properties) if hasattr(mesh, 'vert_properties') else 0,
        'volume_mm3': volume,
        'relative_density': relative_density,
        'porosity': porosity,
        'target_porosity': params.porosity,
        'cell_count': nx * ny * nz,
        'cells_x': nx,
        'cells_y': ny,
        'cells_z': nz,
        'strut_count': len(all_strut_endpoints),
        'unit_cell_size_mm': cell,
        'strut_diameter_mm': params.strut_diameter_mm,
        'strut_taper': strut_taper,
        'strut_profile': strut_profile,
        'lattice_type': lattice_type,
        'pore_size_um': params.pore_size_um,
        'estimated_pore_size_um': estimated_pore_size_um,
        # Gradient features
        'gradient_density_enabled': enable_gradient,
        'gradient_axis': gradient_axis if enable_gradient else None,
        'gradient_start_density': gradient_start if enable_gradient else None,
        'gradient_end_density': gradient_end if enable_gradient else None,
        # Node features
        'node_spheres_enabled': enable_node_spheres,
        'node_sphere_factor': node_sphere_factor if enable_node_spheres else None,
        'node_filleting_enabled': enable_filleting,
        'fillet_radius_factor': fillet_factor if enable_filleting else None,
        'node_count': node_count,
        # Generation
        'seed': params.seed,
        'scaffold_type': 'lattice'
    }

    return result, stats


def generate_lattice_from_dict(params: dict) -> tuple[m3d.Manifold, dict]:
    """
    Generate lattice from dictionary parameters (for API compatibility).

    Supports both new biological parameters and legacy parameter names
    for backward compatibility.

    Args:
        params: Dictionary with keys matching LatticeParams fields

    Returns:
        Tuple of (manifold, stats_dict)
    """
    # Handle bounding box - support multiple formats
    bbox = params.get('bounding_box_mm', params.get('bounding_box'))
    if isinstance(bbox, dict):
        bx = bbox.get('x', 10.0)
        by = bbox.get('y', 10.0)
        bz = bbox.get('z', 10.0)
    elif isinstance(bbox, (list, tuple)):
        bx, by, bz = bbox[0], bbox[1], bbox[2]
    else:
        bx = params.get('bounding_box_x_mm', 10.0)
        by = params.get('bounding_box_y_mm', 10.0)
        bz = params.get('bounding_box_z_mm', 10.0)

    # Handle unit cell - legacy and new
    unit_cell_size = params.get('unit_cell_size_mm', params.get('cell_size_mm', 2.0))
    lattice_type = params.get('lattice_type', params.get('unit_cell', 'cubic'))

    return generate_lattice(LatticeParams(
        # Basic Geometry
        bounding_box_x_mm=bx,
        bounding_box_y_mm=by,
        bounding_box_z_mm=bz,
        unit_cell_size_mm=unit_cell_size,

        # Lattice Type
        lattice_type=lattice_type,

        # Strut Properties
        strut_diameter_mm=params.get('strut_diameter_mm', 0.5),
        strut_taper=params.get('strut_taper', 0.0),
        strut_profile=params.get('strut_profile', 'circular'),

        # Porosity Control
        porosity=params.get('porosity', 0.7),
        pore_size_um=params.get('pore_size_um', 500.0),

        # Gradient Features
        enable_gradient_density=params.get('enable_gradient_density', False),
        gradient_axis=params.get('gradient_axis', 'z'),
        gradient_start_density=params.get('gradient_start_density', 0.5),
        gradient_end_density=params.get('gradient_end_density', 1.0),

        # Node Features
        enable_node_filleting=params.get('enable_node_filleting', False),
        fillet_radius_factor=params.get('fillet_radius_factor', 0.3),
        enable_node_spheres=params.get('enable_node_spheres', False),
        node_sphere_factor=params.get('node_sphere_factor', 1.1),

        # Quality & Generation
        resolution=params.get('resolution', 8),
        seed=params.get('seed'),
    ))
